"""
消融实验: 少样本目标域微调 vs TTT/TTN
============================================================
实验 A: Kvasir 训练权重  → 小量 CVC 数据微调   → 评估 CVC + ETIS 泛化
实验 B: 实验 A 权重      → 小量 ETIS 数据微调  → 评估 CVC + ETIS 泛化

对照 (已有结果, 直接嵌入):
  - Kvasir Baseline (无自适应)
  - +TTN  (测试时归一化, 无需标注)
  - +TTT  (测试时训练, 无需标注)

预期:
  CVC  泛化: Baseline < Exp A < TTT ≈ Exp B
  ETIS 泛化: Baseline < Exp A < TTT < Exp B

用法:
  cd <PROJECT_ROOT>
  python scripts/train/run_ablation_finetune.py                 # 全部实验
  python scripts/train/run_ablation_finetune.py --models yolo    # 仅 YOLO
  python scripts/train/run_ablation_finetune.py --models rtdetr  # 仅 RT-DETR
  python scripts/train/run_ablation_finetune.py --skip-data-prep # 跳过数据准备(已准备好)
"""

import os
import json
import shutil
import random
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  超参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CVC_FINETUNE_N  = 50      # CVC 微调使用的图片数  (~8% of 612)
ETIS_FINETUNE_N = 20      # ETIS 微调使用的图片数 (~10% of 196)
FT_EPOCHS       = 20      # 微调训练轮数
YOLO_FT_LR      = 0.001   # YOLO 微调初始学习率 (低于默认 0.01)
RTDETR_FT_LR    = 0.0001  # RT-DETR 微调学习率
SEED            = 42

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Kvasir 训练的基线权重
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOLO_WEIGHTS = {
    'YOLOv8n-seg':  'runs/segment/v8n_100ep/weights/best.pt',
    'YOLOv11s-seg': 'runs/segment/main_v11s_100ep/weights/best.pt',
}
RTDETR_WEIGHT = 'runs/detect/main_rtdetr_100ep_server3/weights/best.pt'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  已有 Baseline / TTN / TTT 结果 (%)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXISTING = {
    'YOLOv8n-seg': {
        'baseline': {'cvc': 43.3, 'etis': 39.5},
        'ttn':      {'cvc': 72.5, 'etis': 67.8},
        'ttt':      {'cvc': 80.7, 'etis': 76.3},
    },
    'YOLOv11s-seg': {
        'baseline': {'cvc': 82.4, 'etis': 77.8},
        'ttn':      {'cvc': 85.7, 'etis': 81.2},
        'ttt':      {'cvc': 91.6, 'etis': 87.1},
    },
    'RT-DETR-L': {
        'baseline': {'cvc': 87.7, 'etis': 87.9},
        'ttn':      {'cvc': 88.2, 'etis': 86.1},   # RT-DETR+SAM Mask
        'ttt':      {'cvc': 93.4, 'etis': 91.8},   # RT-DETR+SAM Mask
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  数据源 / 评估配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SRC_DATASETS = {
    'cvc': {
        'img_dir':  'datasets/cvc_clinicdb/images/test',
        'lbl_dir':  'datasets/cvc_clinicdb/labels/test',
        'n_total':  612,
    },
    'etis': {
        'img_dir':  'datasets/etis_larib/images/test',
        'lbl_dir':  'datasets/etis_larib/labels/test',
        'n_total':  196,
    },
}

# YOLO seg 评估用配置 (已有, 包含完整 test split)
EVAL_YAML_SEG = {
    'cvc':  'configs/cvc_clinicdb.yaml',
    'etis': 'configs/etis_larib.yaml',
}
# RT-DETR det 评估用配置
EVAL_YAML_DET = {
    'cvc':  'configs/cvc_clinicdb_det.yaml',
    'etis': 'configs/etis_larib_det.yaml',
}

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  辅助函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def polygon_line_to_box(line: str):
    """将 YOLO 多边形标注转为 bounding-box 标注."""
    parts = line.strip().split()
    if not parts:
        return None
    cls_id = parts[0]
    coords = [float(v) for v in parts[1:]]
    if len(coords) == 4:
        return line.strip()          # 已经是 box 格式
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None
    xs, ys = coords[0::2], coords[1::2]
    xc = (min(xs) + max(xs)) / 2.0
    yc = (min(ys) + max(ys)) / 2.0
    w  = max(0.0, max(xs) - min(xs))
    h  = max(0.0, max(ys) - min(ys))
    return f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def sample_and_copy(src_img_dir, src_lbl_dir, dst_base, n_samples,
                    seed=42, convert_to_box=False):
    """
    从源 test 目录随机采样 n_samples 张图, 拆成 train/val 复制到 dst_base.
    convert_to_box=True 时将 polygon label 转为 box label (给 RT-DETR 用).
    返回 (n_train, n_val).
    """
    random.seed(seed)
    src_img_dir = Path(src_img_dir)
    src_lbl_dir = Path(src_lbl_dir)
    dst_base    = Path(dst_base)

    # 收集 (image, label) 配对
    pairs = []
    for lbl in sorted(src_lbl_dir.glob("*.txt")):
        for ext in IMAGE_EXTS:
            img = src_img_dir / (lbl.stem + ext)
            if img.exists():
                pairs.append((img, lbl))
                break

    if not pairs:
        raise RuntimeError(f"No image-label pairs found in {src_img_dir}")

    n_samples = min(n_samples, len(pairs))
    sampled = random.sample(pairs, n_samples)

    # 80/20 分 train/val
    n_val   = max(1, n_samples // 5)
    n_train = n_samples - n_val
    train_pairs = sampled[:n_train]
    val_pairs   = sampled[n_train:]

    for split_name, split_pairs in [('train', train_pairs), ('val', val_pairs)]:
        img_dst = dst_base / 'images' / split_name
        lbl_dst = dst_base / 'labels' / split_name
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for img, lbl in split_pairs:
            shutil.copy2(img, img_dst / img.name)
            if convert_to_box:
                box_lines = []
                for line in lbl.read_text(encoding='utf-8').splitlines():
                    b = polygon_line_to_box(line)
                    if b:
                        box_lines.append(b)
                (lbl_dst / lbl.name).write_text('\n'.join(box_lines),
                                                encoding='utf-8')
            else:
                shutil.copy2(lbl, lbl_dst / lbl.name)

    return n_train, n_val


def write_yaml(path, dataset_path):
    """写一个最简 YOLO YAML config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"# Auto-generated for ablation fine-tuning\n"
        f"path: {dataset_path}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"  0: polyp\n"
    )
    path.write_text(content, encoding='utf-8')
    print(f"  [CONFIG] wrote {path}")


def fmt(v):
    """格式化 mAP50 值 (百分比字符串)."""
    if v is None:
        return "N/A"
    if isinstance(v, float) and v <= 1.0:
        return f"{v*100:.1f}%"
    return f"{v:.1f}%"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  训练/评估 封装
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def finetune_yolo(weight_path, data_yaml, run_name, lr=0.001):
    """微调 YOLO seg 模型, 返回 best.pt 路径."""
    from ultralytics import YOLO
    print(f"\n{'='*60}")
    print(f"[FINETUNE-YOLO] weight={weight_path}")
    print(f"[FINETUNE-YOLO] data={data_yaml}  epochs={FT_EPOCHS}  lr={lr}")
    print(f"{'='*60}")

    project_abs = str(ROOT / 'runs' / 'segment')
    model = YOLO(weight_path)
    results = model.train(
        data=data_yaml,
        epochs=FT_EPOCHS,
        imgsz=640,
        batch=16,
        workers=0,           # Windows 兼容
        name=run_name,
        project=project_abs,
        exist_ok=True,
        patience=FT_EPOCHS,  # 不要早停
        seed=0,
        lr0=lr,
        lrf=0.1,
        plots=True,
    )
    # 从 trainer 获取实际保存路径
    save_dir = Path(getattr(results, 'save_dir', project_abs + '/' + run_name))
    best = save_dir / 'weights' / 'best.pt'
    if not best.exists():
        best = save_dir / 'weights' / 'last.pt'
    print(f"[FINETUNE-YOLO] saved → {best}")
    return str(best)


def finetune_rtdetr(weight_path, data_yaml, run_name, lr=0.0001):
    """微调 RT-DETR 检测模型, 返回 best.pt 路径."""
    from ultralytics import YOLO
    print(f"\n{'='*60}")
    print(f"[FINETUNE-RTDETR] weight={weight_path}")
    print(f"[FINETUNE-RTDETR] data={data_yaml}  epochs={FT_EPOCHS}  lr={lr}")
    print(f"{'='*60}")

    project_abs = str(ROOT / 'runs' / 'detect')
    model = YOLO(weight_path)
    results = model.train(
        data=data_yaml,
        epochs=FT_EPOCHS,
        imgsz=640,
        batch=16,
        workers=0,
        name=run_name,
        project=project_abs,
        exist_ok=True,
        patience=FT_EPOCHS,
        seed=0,
        lr0=lr,
        amp=False,            # RT-DETR + AMP 可能导致 NaN
        plots=True,
    )
    save_dir = Path(getattr(results, 'save_dir', project_abs + '/' + run_name))
    best = save_dir / 'weights' / 'best.pt'
    if not best.exists():
        best = save_dir / 'weights' / 'last.pt'
    print(f"[FINETUNE-RTDETR] saved → {best}")
    return str(best)


def eval_yolo_seg(weight_path, eval_yaml, split='test', name_suffix=''):
    """评估 YOLO seg 模型, 返回 Mask mAP@0.5 (0~1)."""
    from ultralytics import YOLO
    print(f"  [EVAL-SEG] {weight_path} on {eval_yaml} split={split}")
    model = YOLO(weight_path)
    metrics = model.val(
        data=eval_yaml,
        split=split,
        imgsz=640,
        batch=16,
        workers=0,
        plots=False,
        save_json=False,
        name=f'ablation_eval_{name_suffix}',
        project=str(ROOT / 'runs' / 'segment'),
        exist_ok=True,
    )
    try:
        val = float(metrics.seg.map50)
    except Exception:
        val = None
    print(f"  [EVAL-SEG] Mask mAP@0.5 = {fmt(val)}")
    return val


def eval_rtdetr_box(weight_path, eval_yaml, split='test', name_suffix=''):
    """评估 RT-DETR 检测模型, 返回 Box mAP@0.5 (0~1)."""
    from ultralytics import YOLO
    print(f"  [EVAL-DET] {weight_path} on {eval_yaml} split={split}")
    model = YOLO(weight_path)
    metrics = model.val(
        data=eval_yaml,
        split=split,
        imgsz=640,
        batch=16,
        workers=0,
        plots=False,
        save_json=False,
        name=f'ablation_eval_{name_suffix}',
        project=str(ROOT / 'runs' / 'detect'),
        exist_ok=True,
        amp=False,
    )
    try:
        val = float(metrics.box.map50)
    except Exception:
        val = None
    print(f"  [EVAL-DET] Box mAP@0.5 = {fmt(val)}")
    return val


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  数据准备
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def prepare_all_finetune_data():
    """为 CVC / ETIS 创建微调训练子集 (seg + det 两套)."""
    print("\n" + "="*80)
    print("  STEP 0: 准备微调数据子集")
    print("="*80)

    specs = [
        # (name, n_samples, convert_to_box, dst_suffix)
        ('cvc',  CVC_FINETUNE_N,  False, 'cvc_finetune_seg'),
        ('cvc',  CVC_FINETUNE_N,  True,  'cvc_finetune_det'),
        ('etis', ETIS_FINETUNE_N, False, 'etis_finetune_seg'),
        ('etis', ETIS_FINETUNE_N, True,  'etis_finetune_det'),
    ]

    for ds_name, n, to_box, dst_suffix in specs:
        src = SRC_DATASETS[ds_name]
        dst = ROOT / 'datasets' / dst_suffix

        if dst.exists():
            print(f"  [DATA] {dst_suffix}: 已存在, 跳过 (若需重建请先删除)")
            continue

        n_tr, n_va = sample_and_copy(
            ROOT / src['img_dir'], ROOT / src['lbl_dir'],
            dst, n, seed=SEED, convert_to_box=to_box
        )
        print(f"  [DATA] {dst_suffix}: train={n_tr}, val={n_va}  "
              f"({'box' if to_box else 'polygon'} labels)")

    # 写 YAML configs
    write_yaml(ROOT / 'configs' / 'cvc_finetune_seg.yaml',
               './datasets/cvc_finetune_seg')
    write_yaml(ROOT / 'configs' / 'etis_finetune_seg.yaml',
               './datasets/etis_finetune_seg')
    write_yaml(ROOT / 'configs' / 'cvc_finetune_det.yaml',
               './datasets/cvc_finetune_det')
    write_yaml(ROOT / 'configs' / 'etis_finetune_det.yaml',
               './datasets/etis_finetune_det')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  打印最终对比表
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_comparison_table(all_results):
    """合并已有结果和新实验结果, 打印消融对比表."""
    print("\n" + "="*80)
    print("  ★ 消融实验结果: 少样本目标域微调 vs TTT/TTN")
    print("="*80)

    header = (f"| {'模型':<16} | {'方法':<28} | {'CVC mAP50':>10} "
              f"| {'ETIS mAP50':>11} | {'额外标注':>8} | {'额外训练':>8} |")
    sep = (f"|:{'-'*15} |:{'-'*27} |{'-'*11}:|"
           f"{'-'*12}:|{'-'*9}:|{'-'*9}:|")

    print(f"\n{header}")
    print(sep)

    for model_name in all_results:
        ex = EXISTING.get(model_name, {})
        res = all_results[model_name]

        # Baseline
        bl_cvc  = ex.get('baseline', {}).get('cvc', '—')
        bl_etis = ex.get('baseline', {}).get('etis', '—')
        print(f"| {model_name:<16} | {'Kvasir Baseline':<28} "
              f"| {fmt(bl_cvc):>10} | {fmt(bl_etis):>11} | {'❌':>8} | {'❌':>8} |")

        # TTN
        ttn_cvc  = ex.get('ttn', {}).get('cvc')
        ttn_etis = ex.get('ttn', {}).get('etis')
        if ttn_cvc is not None:
            print(f"| {'':<16} | {'+ TTN':<28} "
                  f"| {fmt(ttn_cvc):>10} | {fmt(ttn_etis):>11} | {'❌':>8} | {'❌':>8} |")

        # TTT
        ttt_cvc  = ex.get('ttt', {}).get('cvc')
        ttt_etis = ex.get('ttt', {}).get('etis')
        if ttt_cvc is not None:
            print(f"| {'':<16} | {'**+ TTT**':<28} "
                  f"| {fmt(ttt_cvc):>10} | {fmt(ttt_etis):>11} | {'❌':>8} | {'❌':>8} |")

        # Exp A
        ea = res.get('exp_a', {})
        ea_cvc  = ea.get('cvc')
        ea_etis = ea.get('etis')
        cvc_n = CVC_FINETUNE_N
        print(f"| {'':<16} | {'Exp A: +CVC微调(' + str(cvc_n) + '张)':<28} "
              f"| {fmt(ea_cvc):>10} | {fmt(ea_etis):>11} "
              f"| {'✅ ' + str(cvc_n) + '张':>8} | {'✅ ' + str(FT_EPOCHS) + 'ep':>8} |")

        # Exp B
        eb = res.get('exp_b', {})
        eb_cvc  = eb.get('cvc')
        eb_etis = eb.get('etis')
        etis_n = ETIS_FINETUNE_N
        print(f"| {'':<16} | {'Exp B: +CVC+ETIS微调(+' + str(etis_n) + '张)':<28} "
              f"| {fmt(eb_cvc):>10} | {fmt(eb_etis):>11} "
              f"| {'✅ +' + str(etis_n) + '张':>8} | {'✅ +' + str(FT_EPOCHS) + 'ep':>8} |")

        print(f"| {'':<16} | {'':<28} | {'':<10} "
              f"| {'':<11} | {'':<8} | {'':<8} |")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    parser = argparse.ArgumentParser(description="消融实验: 少样本微调 vs TTT/TTN")
    parser.add_argument('--models', choices=['all', 'yolo', 'rtdetr'],
                        default='all', help='运行哪些模型的实验')
    parser.add_argument('--skip-data-prep', action='store_true',
                        help='跳过数据准备步骤 (已经准备好)')
    return parser.parse_args()


def run_yolo_ablation():
    """运行所有 YOLO 模型的消融实验."""
    results = {}

    for model_name, base_weight in YOLO_WEIGHTS.items():
        safe_name = model_name.replace('-', '_').lower()
        print(f"\n{'#'*80}")
        print(f"  MODEL: {model_name}")
        print(f"{'#'*80}")

        results[model_name] = {'exp_a': {}, 'exp_b': {}}

        # ── 实验 A: 在 CVC 子集上微调 ──────────────────
        print(f"\n{'─'*60}")
        print(f"  实验 A: {model_name} → CVC 微调 ({CVC_FINETUNE_N} 张)")
        print(f"{'─'*60}")

        exp_a_run = f'ablation_expA_{safe_name}_cvc_ft'
        exp_a_weight = finetune_yolo(
            base_weight,
            'configs/cvc_finetune_seg.yaml',
            run_name=exp_a_run,
            lr=YOLO_FT_LR,
        )

        # 评估 Exp A → CVC 全量
        exp_a_cvc = eval_yolo_seg(
            exp_a_weight,
            EVAL_YAML_SEG['cvc'],
            split='test',
            name_suffix=f'{safe_name}_expA_cvc',
        )
        results[model_name]['exp_a']['cvc'] = exp_a_cvc

        # 评估 Exp A → ETIS 全量
        exp_a_etis = eval_yolo_seg(
            exp_a_weight,
            EVAL_YAML_SEG['etis'],
            split='test',
            name_suffix=f'{safe_name}_expA_etis',
        )
        results[model_name]['exp_a']['etis'] = exp_a_etis

        # ── 实验 B: 在 Exp A 基础上, 继续微调 ETIS 子集 ──
        print(f"\n{'─'*60}")
        print(f"  实验 B: {model_name} (Exp A 权重) → ETIS 微调 ({ETIS_FINETUNE_N} 张)")
        print(f"{'─'*60}")

        exp_b_run = f'ablation_expB_{safe_name}_etis_ft'
        exp_b_weight = finetune_yolo(
            exp_a_weight,           # 注意: 从 Exp A 权重继续
            'configs/etis_finetune_seg.yaml',
            run_name=exp_b_run,
            lr=YOLO_FT_LR,
        )

        # 评估 Exp B → CVC 全量
        exp_b_cvc = eval_yolo_seg(
            exp_b_weight,
            EVAL_YAML_SEG['cvc'],
            split='test',
            name_suffix=f'{safe_name}_expB_cvc',
        )
        results[model_name]['exp_b']['cvc'] = exp_b_cvc

        # 评估 Exp B → ETIS 全量
        exp_b_etis = eval_yolo_seg(
            exp_b_weight,
            EVAL_YAML_SEG['etis'],
            split='test',
            name_suffix=f'{safe_name}_expB_etis',
        )
        results[model_name]['exp_b']['etis'] = exp_b_etis

        # 即时汇报
        ex = EXISTING.get(model_name, {})
        print(f"\n  ┌─────── {model_name} 消融速报 ───────┐")
        print(f"  │ Baseline → CVC:  {fmt(ex.get('baseline',{}).get('cvc'))}")
        print(f"  │ + TTT    → CVC:  {fmt(ex.get('ttt',{}).get('cvc'))}")
        print(f"  │ Exp A    → CVC:  {fmt(exp_a_cvc)}")
        print(f"  │ Exp B    → CVC:  {fmt(exp_b_cvc)}")
        print(f"  │")
        print(f"  │ Baseline → ETIS: {fmt(ex.get('baseline',{}).get('etis'))}")
        print(f"  │ + TTT    → ETIS: {fmt(ex.get('ttt',{}).get('etis'))}")
        print(f"  │ Exp A    → ETIS: {fmt(exp_a_etis)}")
        print(f"  │ Exp B    → ETIS: {fmt(exp_b_etis)}")
        print(f"  └──────────────────────────────────────┘")

    return results


def run_rtdetr_ablation():
    """运行 RT-DETR 模型的消融实验 (Box mAP50)."""
    model_name = 'RT-DETR-L'
    safe_name = 'rtdetr_l'
    results = {model_name: {'exp_a': {}, 'exp_b': {}}}

    print(f"\n{'#'*80}")
    print(f"  MODEL: {model_name} (Box mAP50)")
    print(f"{'#'*80}")

    # ── 实验 A: 在 CVC det 子集上微调 ──────────────
    print(f"\n{'─'*60}")
    print(f"  实验 A: {model_name} → CVC 微调 ({CVC_FINETUNE_N} 张)")
    print(f"{'─'*60}")

    exp_a_run = f'ablation_expA_{safe_name}_cvc_ft'
    exp_a_weight = finetune_rtdetr(
        RTDETR_WEIGHT,
        'configs/cvc_finetune_det.yaml',
        run_name=exp_a_run,
        lr=RTDETR_FT_LR,
    )

    # 评估 Exp A → CVC 全量 (Box)
    exp_a_cvc = eval_rtdetr_box(
        exp_a_weight,
        EVAL_YAML_DET['cvc'],
        split='test',
        name_suffix=f'{safe_name}_expA_cvc',
    )
    results[model_name]['exp_a']['cvc'] = exp_a_cvc

    # 评估 Exp A → ETIS 全量 (Box)
    exp_a_etis = eval_rtdetr_box(
        exp_a_weight,
        EVAL_YAML_DET['etis'],
        split='test',
        name_suffix=f'{safe_name}_expA_etis',
    )
    results[model_name]['exp_a']['etis'] = exp_a_etis

    # ── 实验 B: 在 Exp A 基础上, 继续微调 ETIS det 子集 ──
    print(f"\n{'─'*60}")
    print(f"  实验 B: {model_name} (Exp A) → ETIS 微调 ({ETIS_FINETUNE_N} 张)")
    print(f"{'─'*60}")

    exp_b_run = f'ablation_expB_{safe_name}_etis_ft'
    exp_b_weight = finetune_rtdetr(
        exp_a_weight,
        'configs/etis_finetune_det.yaml',
        run_name=exp_b_run,
        lr=RTDETR_FT_LR,
    )

    # 评估 Exp B → CVC 全量 (Box)
    exp_b_cvc = eval_rtdetr_box(
        exp_b_weight,
        EVAL_YAML_DET['cvc'],
        split='test',
        name_suffix=f'{safe_name}_expB_cvc',
    )
    results[model_name]['exp_b']['cvc'] = exp_b_cvc

    # 评估 Exp B → ETIS 全量 (Box)
    exp_b_etis = eval_rtdetr_box(
        exp_b_weight,
        EVAL_YAML_DET['etis'],
        split='test',
        name_suffix=f'{safe_name}_expB_etis',
    )
    results[model_name]['exp_b']['etis'] = exp_b_etis

    # 即时汇报
    ex = EXISTING.get(model_name, {})
    print(f"\n  ┌─────── {model_name} 消融速报 ───────┐")
    print(f"  │ Baseline → CVC:  {fmt(ex.get('baseline',{}).get('cvc'))}")
    print(f"  │ + TTT*   → CVC:  {fmt(ex.get('ttt',{}).get('cvc'))}  (*SAM Mask)")
    print(f"  │ Exp A    → CVC:  {fmt(exp_a_cvc)}  (Box)")
    print(f"  │ Exp B    → CVC:  {fmt(exp_b_cvc)}  (Box)")
    print(f"  │")
    print(f"  │ Baseline → ETIS: {fmt(ex.get('baseline',{}).get('etis'))}")
    print(f"  │ + TTT*   → ETIS: {fmt(ex.get('ttt',{}).get('etis'))}  (*SAM Mask)")
    print(f"  │ Exp A    → ETIS: {fmt(exp_a_etis)}  (Box)")
    print(f"  │ Exp B    → ETIS: {fmt(exp_b_etis)}  (Box)")
    print(f"  └──────────────────────────────────────────┘")

    return results


def main():
    args = parse_args()
    os.chdir(ROOT)

    print(f"[START] 消融实验 @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[ROOT]  {ROOT}")
    print(f"[CONFIG] CVC 微调样本={CVC_FINETUNE_N}, ETIS 微调样本={ETIS_FINETUNE_N}")
    print(f"[CONFIG] 微调轮数={FT_EPOCHS}, YOLO LR={YOLO_FT_LR}, RTDETR LR={RTDETR_FT_LR}")

    # Step 0: 数据准备
    if not args.skip_data_prep:
        prepare_all_finetune_data()
    else:
        print("\n[SKIP] 数据准备已跳过")

    all_results = {}

    # Step 1: YOLO 消融
    if args.models in ('all', 'yolo'):
        yolo_results = run_yolo_ablation()
        all_results.update(yolo_results)

    # Step 2: RT-DETR 消融
    if args.models in ('all', 'rtdetr'):
        rtdetr_results = run_rtdetr_ablation()
        all_results.update(rtdetr_results)

    # Step 3: 保存结果
    summary_dir = ROOT / 'results' / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / 'ablation_finetune_results.json'

    # 转换 None → str for JSON
    def serialise(obj):
        if isinstance(obj, dict):
            return {k: serialise(v) for k, v in obj.items()}
        if obj is None:
            return None
        if isinstance(obj, float):
            return round(obj, 6)
        return obj

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(serialise(all_results), f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {summary_path}")

    # Step 4: 打印消融对比表
    print_comparison_table(all_results)

    print(f"\n[DONE] 消融实验完成 @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
