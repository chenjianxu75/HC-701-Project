"""补跑 Pseudo-Label TTT (修复路径问题)"""
import os, glob, json, shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

def main():
    os.chdir(ROOT)
    from ultralytics import YOLO

    ETIS_CFG = 'configs/etis_larib.yaml'
    ETIS_IMGS = sorted(glob.glob('datasets/etis_larib/images/test/*'))
    BS = 16
    summary_path = ROOT / 'results' / 'summary' / 'ttt_demo_results.json'
    results = json.load(open(summary_path))

    for tag, model_path in [('v8n', 'runs/segment/v8n_100ep/weights/best.pt'),
                             ('v11s', 'runs/segment/main_v11s_100ep/weights/best.pt')]:
        print(f"\n{'='*60}")
        print(f"Pseudo-TTT 补测: {tag}")
        print(f"{'='*60}")

        # 从错误路径拿到已生成的伪标签
        wrong_lbl_dir = f'runs/segment/datasets/etis_pseudo_{tag}/pred/labels'
        pseudo_base = f'datasets/etis_pseudo_{tag}'
        pseudo_img = os.path.join(pseudo_base, 'images', 'train')
        pseudo_lbl = os.path.join(pseudo_base, 'labels', 'train')

        if os.path.exists(pseudo_base):
            shutil.rmtree(pseudo_base)
        os.makedirs(pseudo_img, exist_ok=True)
        os.makedirs(pseudo_lbl, exist_ok=True)

        count = 0
        for ip in ETIS_IMGS:
            name = os.path.basename(ip)
            lname = os.path.splitext(name)[0] + '.txt'
            lpath = os.path.join(wrong_lbl_dir, lname)
            if os.path.exists(lpath) and os.path.getsize(lpath) > 0:
                shutil.copy2(ip, os.path.join(pseudo_img, name))
                shutil.copy2(lpath, os.path.join(pseudo_lbl, lname))
                count += 1
        print(f"  迁移 {count} 张伪标注图像到正确路径")

        # 数据集配置
        pseudo_yaml = f'configs/etis_pseudo_{tag}.yaml'
        with open(pseudo_yaml, 'w') as f:
            f.write(f"path: ./{pseudo_base}\ntrain: images/train\nval: images/train\n\nnames:\n  0: polyp\n")

        # 微调 2 epoch
        model = YOLO(model_path)
        model.train(data=pseudo_yaml, epochs=2, imgsz=640, batch=BS,
                    workers=0, lr0=0.00005, lrf=1.0, warmup_epochs=0,
                    name=f'ttt_pseudo_{tag}', exist_ok=True,
                    plots=False, patience=100)

        wt = f'runs/segment/ttt_pseudo_{tag}/weights/best.pt'
        if not os.path.exists(wt):
            wt = f'runs/segment/ttt_pseudo_{tag}/weights/last.pt'

        model = YOLO(wt)
        m = model.val(data=ETIS_CFG, split='test', imgsz=640, batch=BS,
                      workers=0, plots=False, name=f'ttt_pseudo_{tag}_eval', exist_ok=True)

        results[tag]['pseudo_ttt'] = m.seg.map50
        d = (m.seg.map50 - results[tag]['baseline']) * 100
        print(f"  Pseudo-TTT = {m.seg.map50*100:.2f}%  (Δ {d:+.2f}%)")

    # 保存更新后的结果
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 打印完整表格
    print(f"\n{'='*60}")
    print("完整对比表")
    print(f"{'='*60}")
    def fmt(v, base):
        if v is None: return "N/A"
        d = (v - base) * 100
        return f"{v*100:.2f}% ({d:+.2f}%)"
    print(f"\n| 模型 | Baseline | +TTN | +TTT | +Pseudo-TTT |")
    print(f"| :--- | :---: | :---: | :---: | :---: |")
    for tag, res in results.items():
        b = res['baseline']
        print(f"| {tag} | {b*100:.2f}% | {fmt(res.get('ttn'), b)} | {fmt(res.get('ttt'), b)} | {fmt(res.get('pseudo_ttt'), b)} |")
    print("\nDONE!")

if __name__ == '__main__':
    main()
