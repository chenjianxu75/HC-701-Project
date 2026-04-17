"""
TTT/TTN 可行性验证 Demo v2
===========================
双模型 (v8n + v11s) × 三方法 (TTN + TTT + Pseudo-TTT) = 6组对比实验
基线: v8n→39.5%, v11s→77.8% on ETIS-Larib
"""
import os, glob, json, shutil, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2


ROOT = Path(__file__).resolve().parents[2]


def preprocess_batch(paths, imgsz=640, device='cpu'):
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None: continue
        img = cv2.resize(img, (imgsz, imgsz))
        img = img[:, :, ::-1].copy().transpose(2, 0, 1).astype(np.float32) / 255.0
        imgs.append(torch.from_numpy(img))
    return torch.stack(imgs).to(device) if imgs else None


def run_baseline(YOLO, model_path, etis_cfg, bs):
    """Step 0: 基线评估"""
    model = YOLO(model_path)
    m = model.val(data=etis_cfg, split='test', imgsz=640, batch=bs,
                  workers=0, plots=False, name='ttt_baseline', exist_ok=True)
    return m.seg.map50


def run_ttn(YOLO, model_path, etis_cfg, etis_imgs, bs, device, alpha=0.5):
    """
    方法1: TTN — 渐进式BN统计量混合
    不完全重置，而是将目标域统计量与源域统计量按 alpha 比例混合:
      new_stat = (1-alpha) * source_stat + alpha * target_stat
    """
    model = YOLO(model_path)
    model.to(device)
    tm = model.model

    # 1) 保存源域BN统计量
    source_stats = {}
    for name, mod in tm.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            source_stats[name] = {
                'mean': mod.running_mean.clone(),
                'var': mod.running_var.clone()
            }

    # 2) 重置并收集目标域统计量
    for mod in tm.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.zero_()
            mod.running_var.fill_(1)
            mod.num_batches_tracked.zero_()
            mod.momentum = None  # 使用累积移动平均

    tm.train()
    with torch.no_grad():
        for i in range(0, len(etis_imgs), bs):
            b = preprocess_batch(etis_imgs[i:i + bs], device=device)
            if b is not None:
                tm(b)

    # 3) 混合: new = (1-alpha)*source + alpha*target
    for name, mod in tm.named_modules():
        if isinstance(mod, nn.BatchNorm2d) and name in source_stats:
            src = source_stats[name]
            mod.running_mean.copy_(
                (1 - alpha) * src['mean'] + alpha * mod.running_mean
            )
            mod.running_var.copy_(
                (1 - alpha) * src['var'] + alpha * mod.running_var
            )

    tm.eval()

    # 4) 评估
    m = model.val(data=etis_cfg, split='test', imgsz=640, batch=bs,
                  workers=0, plots=False, name='ttt_ttn', exist_ok=True)
    return m.seg.map50


def run_ttt(YOLO, model_path, etis_cfg, etis_imgs, bs, device, n_steps=5):
    """
    方法2: TTT — 增强一致性BN仿射参数微调
    仅微调BN的 weight/bias, 用翻转一致性作为自监督信号
    """
    import random
    random.seed(42)
    np.random.seed(42)

    model = YOLO(model_path)
    model.to(device)
    tm = model.model

    # 冻结所有参数
    for p in tm.parameters():
        p.requires_grad = False
    # 解冻BN仿射参数
    bn_params = []
    for mod in tm.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.weight.requires_grad_(True)
            mod.bias.requires_grad_(True)
            bn_params += [mod.weight, mod.bias]

    optimizer = torch.optim.Adam(bn_params, lr=0.001)

    # hook 取中间层特征
    features = {}
    def hook_fn(module, inp, out):
        features['feat'] = out
    # 挂在第9层 (SPPF前)
    hook_layer = list(tm.model.children())[9]
    handle = hook_layer.register_forward_hook(hook_fn)
    tm.train()

    for step in range(n_steps):
        idxs = np.random.choice(len(etis_imgs), min(bs, len(etis_imgs)), replace=False)
        batch = preprocess_batch([etis_imgs[i] for i in idxs], device=device)
        if batch is None:
            continue
        # 原图
        _ = tm(batch)
        feat_orig = features['feat'].clone()
        # 水平翻转
        batch_flip = torch.flip(batch, [3])
        _ = tm(batch_flip)
        feat_flip = torch.flip(features['feat'], [3])
        # 一致性损失
        loss = nn.functional.mse_loss(feat_flip, feat_orig)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"      step {step + 1}/{n_steps}  loss={loss.item():.6f}")

    handle.remove()
    tm.eval()

    m = model.val(data=etis_cfg, split='test', imgsz=640, batch=bs,
                  workers=0, plots=False, name='ttt_ttt', exist_ok=True)
    return m.seg.map50


def run_pseudo_ttt(YOLO, model_path, etis_cfg, etis_imgs, bs, model_tag):
    """
    方法3: Pseudo-Label Self-Training
    用模型自身高置信预测作为伪标注, 极小学习率微调2 epoch
    """
    pseudo_base = f'datasets/etis_pseudo_{model_tag}'
    pseudo_img_dir = os.path.join(pseudo_base, 'images', 'train')
    pseudo_lbl_dir = os.path.join(pseudo_base, 'labels', 'train')
    # 清理旧数据
    if os.path.exists(pseudo_base):
        shutil.rmtree(pseudo_base)
    os.makedirs(pseudo_img_dir, exist_ok=True)
    os.makedirs(pseudo_lbl_dir, exist_ok=True)

    # 生成伪标注
    model = YOLO(model_path)
    model.predict(source='datasets/etis_larib/images/test',
                  conf=0.5, save=False, save_txt=True, save_conf=False,
                  project=pseudo_base, name='pred', exist_ok=True, imgsz=640)

    pred_lbl_dir = os.path.join(pseudo_base, 'pred', 'labels')
    count = 0
    for ip in etis_imgs:
        name = os.path.basename(ip)
        lname = os.path.splitext(name)[0] + '.txt'
        lpath = os.path.join(pred_lbl_dir, lname)
        if os.path.exists(lpath) and os.path.getsize(lpath) > 0:
            shutil.copy2(ip, os.path.join(pseudo_img_dir, name))
            shutil.copy2(lpath, os.path.join(pseudo_lbl_dir, lname))
            count += 1

    print(f"      生成 {count}/{len(etis_imgs)} 张伪标注图像")

    if count < 5:
        print("      伪标注数量过少，跳过此方法")
        return None

    # 数据集配置
    pseudo_yaml = f'configs/etis_pseudo_{model_tag}.yaml'
    with open(pseudo_yaml, 'w') as f:
        f.write(f"path: ./{pseudo_base}\ntrain: images/train\nval: images/train\n\nnames:\n  0: polyp\n")

    # 微调
    model = YOLO(model_path)
    model.train(data=pseudo_yaml, epochs=2, imgsz=640, batch=bs,
                workers=0, lr0=0.00005, lrf=1.0, warmup_epochs=0,
                name=f'ttt_pseudo_{model_tag}', exist_ok=True,
                plots=False, patience=100)

    wt = f'runs/segment/ttt_pseudo_{model_tag}/weights/best.pt'
    if not os.path.exists(wt):
        wt = f'runs/segment/ttt_pseudo_{model_tag}/weights/last.pt'
    model = YOLO(wt)
    m = model.val(data=etis_cfg, split='test', imgsz=640, batch=bs,
                  workers=0, plots=False, name=f'ttt_pseudo_{model_tag}_eval',
                  exist_ok=True)
    return m.seg.map50


def main():
    os.chdir(ROOT)
    from ultralytics import YOLO

    ETIS_CFG = 'configs/etis_larib.yaml'
    ETIS_IMGS = sorted(glob.glob('datasets/etis_larib/images/test/*'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    BS = 16

    # 两个模型配置
    models = {
        'v8n': 'runs/segment/v8n_100ep/weights/best.pt',
        'v11s': 'runs/segment/main_v11s_100ep/weights/best.pt',
    }

    all_results = {}

    for tag, model_path in models.items():
        print("\n" + "#" * 70)
        print(f"#  模型: {tag.upper()} — {model_path}")
        print("#" * 70)

        res = {}

        # 0. Baseline
        print(f"\n  [0/3] BASELINE")
        res['baseline'] = run_baseline(YOLO, model_path, ETIS_CFG, BS)
        print(f"    ✓ Baseline = {res['baseline'] * 100:.2f}%")

        # 1. TTN
        print(f"\n  [1/3] TTN (alpha=0.5 混合)")
        res['ttn'] = run_ttn(YOLO, model_path, ETIS_CFG, ETIS_IMGS, BS, device, alpha=0.5)
        d = (res['ttn'] - res['baseline']) * 100
        print(f"    ✓ TTN = {res['ttn'] * 100:.2f}%  (Δ {d:+.2f}%)")

        # 2. TTT
        print(f"\n  [2/3] TTT (5步增强一致性)")
        res['ttt'] = run_ttt(YOLO, model_path, ETIS_CFG, ETIS_IMGS, BS, device, n_steps=5)
        d = (res['ttt'] - res['baseline']) * 100
        print(f"    ✓ TTT = {res['ttt'] * 100:.2f}%  (Δ {d:+.2f}%)")

        # 3. Pseudo-TTT
        print(f"\n  [3/3] Pseudo-Label TTT")
        res['pseudo_ttt'] = run_pseudo_ttt(YOLO, model_path, ETIS_CFG, ETIS_IMGS, BS, tag)
        if res['pseudo_ttt'] is not None:
            d = (res['pseudo_ttt'] - res['baseline']) * 100
            print(f"    ✓ Pseudo-TTT = {res['pseudo_ttt'] * 100:.2f}%  (Δ {d:+.2f}%)")

        all_results[tag] = res

    # ====== 最终汇总 ======
    print("\n" + "=" * 70)
    print("TTT/TTN 可行性验证 — 最终对比表")
    print("=" * 70)

    def fmt(v, base):
        if v is None: return "N/A"
        d = (v - base) * 100
        return f"{v * 100:.2f}% ({d:+.2f}%)"

    print(f"\n| 模型 | Baseline | +TTN | +TTT | +Pseudo-TTT |")
    print(f"| :--- | :---: | :---: | :---: | :---: |")
    for tag, res in all_results.items():
        b = res['baseline']
        print(f"| {tag} | {b * 100:.2f}% | {fmt(res.get('ttn'), b)} | {fmt(res.get('ttt'), b)} | {fmt(res.get('pseudo_ttt'), b)} |")

    output_path = ROOT / 'results' / 'summary' / 'ttt_demo_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print("DONE!")


if __name__ == '__main__':
    main()
