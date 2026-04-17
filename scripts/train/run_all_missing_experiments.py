"""
自动补全表格中所有缺失的实验
Windows兼容版本 - 使用 workers=0 避免多进程问题
"""

import os
import json
import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def extract_mask_map50(metrics):
    """从YOLO metrics中提取 Mask mAP@0.5"""
    try:
        return metrics.seg.map50
    except:
        try:
            return metrics.results_dict.get('metrics/mAP50(M)', None)
        except:
            return None

def run_val(model_path, data_yaml, split='test', name_suffix=''):
    """运行验证/泛化测试并返回 Mask mAP@0.5"""
    from ultralytics import YOLO
    print(f"\n{'='*60}")
    print(f"[VAL] Model: {model_path}")
    print(f"[VAL] Data: {data_yaml}, Split: {split}")
    print(f"{'='*60}")
    
    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=640,
        batch=16,
        workers=0,
        plots=False,
        save_json=False,
        name=f'val_auto_{name_suffix}',
        exist_ok=True
    )
    
    mask_map50 = extract_mask_map50(metrics)
    print(f"[RESULT] Mask mAP@0.5 = {mask_map50}")
    return mask_map50

def run_train(model_name, epochs, run_name):
    """训练模型并返回最佳权重路径和Kvasir mAP@0.5"""
    from ultralytics import YOLO
    print(f"\n{'='*60}")
    print(f"[TRAIN] Model: {model_name}, Epochs: {epochs}")
    print(f"[TRAIN] Run name: {run_name}")
    print(f"{'='*60}")
    
    model = YOLO(model_name)
    results = model.train(
        data='configs/kvasir_seg.yaml',
        epochs=epochs,
        imgsz=640,
        batch=16,
        workers=0,
        name=run_name,
        exist_ok=True,
        patience=100,
        seed=0,
        deterministic=True,
        plots=True,
    )
    
    best_weight = os.path.join('runs', 'segment', run_name, 'weights', 'best.pt')
    
    # 从 results.csv 获取最佳 mask mAP50
    csv_path = os.path.join('runs', 'segment', run_name, 'results.csv')
    kvasir_map50 = None
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                best_val = 0
                for row in rows:
                    for key in row:
                        k = key.strip()
                        if 'mAP50' in k and 'mAP50-95' not in k and ('M' in k or 'mask' in k.lower()):
                            try:
                                val = float(row[key].strip())
                                if val > best_val:
                                    best_val = val
                            except:
                                pass
                if best_val > 0:
                    kvasir_map50 = best_val
    
    print(f"[TRAIN RESULT] Best Kvasir Mask mAP@0.5 = {kvasir_map50}")
    return best_weight, kvasir_map50

def fmt(v):
    if v is None:
        return "N/A"
    return f"{v*100:.1f}%"

def main():
    os.chdir(ROOT)
    
    results_log = {}
    phase1_path = ROOT / 'results' / 'summary' / 'experiment_results_phase1.json'
    phase2_path = ROOT / 'results' / 'summary' / 'experiment_results_phase2.json'
    final_path = ROOT / 'results' / 'summary' / 'experiment_results_final.json'

    # ============================================================
    # Phase 1: 使用已有权重运行缺失的 ETIS 泛化测试
    # ============================================================
    print("\n" + "="*80)
    print("PHASE 1: 使用已有权重 补测 ETIS 泛化")
    print("="*80)

    # 1. YOLOv8n-seg 5ep -> ETIS
    print("\n>>> Experiment 1: YOLOv8n-seg 5ep -> ETIS")
    v8n_5ep_etis = run_val(
        'runs/segment/demo_v8n/weights/best.pt',
        'configs/etis_larib.yaml',
        split='test',
        name_suffix='v8n_5ep_etis'
    )
    results_log['v8n_5ep_etis'] = v8n_5ep_etis

    # 2. YOLOv11s-seg 5ep -> ETIS
    print("\n>>> Experiment 2: YOLOv11s-seg 5ep -> ETIS")
    v11s_5ep_etis = run_val(
        'runs/segment/demo_v11s/weights/best.pt',
        'configs/etis_larib.yaml',
        split='test',
        name_suffix='v11s_5ep_etis'
    )
    results_log['v11s_5ep_etis'] = v11s_5ep_etis

    # 3. YOLOv11s-seg 50ep -> ETIS
    print("\n>>> Experiment 3: YOLOv11s-seg 50ep -> ETIS")
    v11s_50ep_etis = run_val(
        'runs/segment/demo_v11s_50ep/weights/best.pt',
        'configs/etis_larib.yaml',
        split='test',
        name_suffix='v11s_50ep_etis'
    )
    results_log['v11s_50ep_etis'] = v11s_50ep_etis

    # 保存Phase1结果
    with open(phase1_path, 'w') as f:
        json.dump(results_log, f, indent=2)
    print(f"\n[Phase 1 Done] Results so far: {results_log}")

    # ============================================================
    # Phase 2: 训练 YOLOv8n-seg 50ep 并全面评估
    # ============================================================
    print("\n" + "="*80)
    print("PHASE 2: 训练 YOLOv8n-seg 50ep")
    print("="*80)

    v8n_50ep_weight, v8n_50ep_kvasir = run_train('checkpoints/yolov8n-seg.pt', 50, 'v8n_50ep')
    results_log['v8n_50ep_kvasir_train'] = v8n_50ep_kvasir

    # Kvasir val 重新确认
    print("\n>>> YOLOv8n-seg 50ep -> Kvasir val (确认)")
    v8n_50ep_kvasir_val = run_val(
        v8n_50ep_weight,
        'configs/kvasir_seg.yaml',
        split='val',
        name_suffix='v8n_50ep_kvasir'
    )
    results_log['v8n_50ep_kvasir'] = v8n_50ep_kvasir_val

    # CVC泛化
    print("\n>>> YOLOv8n-seg 50ep -> CVC泛化")
    v8n_50ep_cvc = run_val(
        v8n_50ep_weight,
        'configs/cvc_clinicdb.yaml',
        split='test',
        name_suffix='v8n_50ep_cvc'
    )
    results_log['v8n_50ep_cvc'] = v8n_50ep_cvc

    # ETIS泛化
    print("\n>>> YOLOv8n-seg 50ep -> ETIS泛化")
    v8n_50ep_etis = run_val(
        v8n_50ep_weight,
        'configs/etis_larib.yaml',
        split='test',
        name_suffix='v8n_50ep_etis'
    )
    results_log['v8n_50ep_etis'] = v8n_50ep_etis

    # 保存Phase2结果
    with open(phase2_path, 'w') as f:
        json.dump(results_log, f, indent=2)
    print(f"\n[Phase 2 Done] Results so far: {results_log}")

    # ============================================================
    # Phase 3: 训练 YOLOv8n-seg 100ep 并全面评估
    # ============================================================
    print("\n" + "="*80)
    print("PHASE 3: 训练 YOLOv8n-seg 100ep")
    print("="*80)

    v8n_100ep_weight, v8n_100ep_kvasir = run_train('checkpoints/yolov8n-seg.pt', 100, 'v8n_100ep')
    results_log['v8n_100ep_kvasir_train'] = v8n_100ep_kvasir

    # Kvasir val 重新确认
    print("\n>>> YOLOv8n-seg 100ep -> Kvasir val (确认)")
    v8n_100ep_kvasir_val = run_val(
        v8n_100ep_weight,
        'configs/kvasir_seg.yaml',
        split='val',
        name_suffix='v8n_100ep_kvasir'
    )
    results_log['v8n_100ep_kvasir'] = v8n_100ep_kvasir_val

    # CVC泛化
    print("\n>>> YOLOv8n-seg 100ep -> CVC泛化")
    v8n_100ep_cvc = run_val(
        v8n_100ep_weight,
        'configs/cvc_clinicdb.yaml',
        split='test',
        name_suffix='v8n_100ep_cvc'
    )
    results_log['v8n_100ep_cvc'] = v8n_100ep_cvc

    # ETIS泛化
    print("\n>>> YOLOv8n-seg 100ep -> ETIS泛化")
    v8n_100ep_etis = run_val(
        v8n_100ep_weight,
        'configs/etis_larib.yaml',
        split='test',
        name_suffix='v8n_100ep_etis'
    )
    results_log['v8n_100ep_etis'] = v8n_100ep_etis

    # ============================================================
    # 最终汇总
    # ============================================================
    with open(final_path, 'w') as f:
        json.dump(results_log, f, indent=2)

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED - RESULTS SUMMARY")
    print("="*80)

    for key, value in results_log.items():
        if value is not None:
            print(f"  {key}: {value*100:.1f}%")
        else:
            print(f"  {key}: N/A")

    # 已知数据
    known = {
        'v8n_5ep_kvasir': 0.885,
        'v8n_5ep_cvc': 0.386,
        'v11s_5ep_kvasir': 0.911,
        'v11s_5ep_cvc': 0.426,
        'v11s_50ep_kvasir': 0.954,
        'v11s_50ep_cvc': 0.827,
        'v11s_100ep_kvasir': 0.958,
        'v11s_100ep_cvc': 0.824,
        'v11s_100ep_etis': 0.778,
    }

    print(f"\n{'='*80}")
    print("COMPLETE TABLE (Instance Segmentation - Mask mAP@0.5)")
    print(f"{'='*80}")
    print(f"\n| 模型版本 | 训练量 | Kvasir mAP50 | CVC 泛化 mAP50 | ETIS 泛化 mAP50 |")
    print(f"| :--- | :---: | :---: | :---: | :---: |")
    print(f"| YOLOv8n-seg | 5 ep | {fmt(known['v8n_5ep_kvasir'])} | {fmt(known['v8n_5ep_cvc'])} | {fmt(results_log.get('v8n_5ep_etis'))} |")
    print(f"| YOLOv8n-seg | 50 ep | {fmt(results_log.get('v8n_50ep_kvasir'))} | {fmt(results_log.get('v8n_50ep_cvc'))} | {fmt(results_log.get('v8n_50ep_etis'))} |")
    print(f"| YOLOv8n-seg | 100 ep | {fmt(results_log.get('v8n_100ep_kvasir'))} | {fmt(results_log.get('v8n_100ep_cvc'))} | {fmt(results_log.get('v8n_100ep_etis'))} |")
    print(f"| YOLOv11s-seg | 5 ep | {fmt(known['v11s_5ep_kvasir'])} | {fmt(known['v11s_5ep_cvc'])} | {fmt(results_log.get('v11s_5ep_etis'))} |")
    print(f"| YOLOv11s-seg | 50 ep | {fmt(known['v11s_50ep_kvasir'])} | {fmt(known['v11s_50ep_cvc'])} | {fmt(results_log.get('v11s_50ep_etis'))} |")
    print(f"| YOLOv11s-seg | 100 ep | {fmt(known['v11s_100ep_kvasir'])} | {fmt(known['v11s_100ep_cvc'])} | {fmt(known['v11s_100ep_etis'])} |")

    print("\n\nDONE!")

if __name__ == '__main__':
    main()
