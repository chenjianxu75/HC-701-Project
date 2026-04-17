"""
Phase 3 恢复脚本: 从中断处继续 YOLOv8n-seg 100ep 训练，然后进行所有评估
"""
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def main():
    os.chdir(ROOT)
    from ultralytics import YOLO
    
    results_log = {}
    
    # 读取Phase2已有结果
    phase2_path = ROOT / 'results' / 'summary' / 'experiment_results_phase2.json'
    final_path = ROOT / 'results' / 'summary' / 'experiment_results_final.json'
    if phase2_path.exists():
        with open(phase2_path, 'r') as f:
            results_log = json.load(f)
        print(f"Loaded Phase 1&2 results: {results_log}")
    
    # ============================================================
    # Phase 3: 恢复训练 YOLOv8n-seg 100ep
    # ============================================================
    print("\n" + "="*80)
    print("PHASE 3: 恢复训练 YOLOv8n-seg 100ep (from last.pt)")
    print("="*80)
    
    last_weight = 'runs/segment/v8n_100ep/weights/last.pt'
    best_weight = 'runs/segment/v8n_100ep/weights/best.pt'
    
    # 恢复训练
    model = YOLO(last_weight)
    model.train(
        resume=True,
        workers=0,
    )
    
    # 训练完成后获取 Kvasir val 的 mAP50
    import csv
    csv_path = 'runs/segment/v8n_100ep/results.csv'
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
    
    results_log['v8n_100ep_kvasir_train'] = kvasir_map50
    print(f"[TRAIN RESULT] Best Kvasir Mask mAP@0.5 = {kvasir_map50}")
    
    # Kvasir val 确认
    print("\n>>> YOLOv8n-seg 100ep -> Kvasir val (确认)")
    model = YOLO(best_weight)
    metrics = model.val(
        data='configs/kvasir_seg.yaml',
        split='val',
        imgsz=640,
        batch=16,
        workers=0,
        plots=False,
        name='val_auto_v8n_100ep_kvasir',
        exist_ok=True
    )
    v8n_100ep_kvasir = metrics.seg.map50
    results_log['v8n_100ep_kvasir'] = v8n_100ep_kvasir
    print(f"[RESULT] Kvasir Mask mAP@0.5 = {v8n_100ep_kvasir}")
    
    # CVC泛化
    print("\n>>> YOLOv8n-seg 100ep -> CVC泛化")
    model = YOLO(best_weight)
    metrics = model.val(
        data='configs/cvc_clinicdb.yaml',
        split='test',
        imgsz=640,
        batch=16,
        workers=0,
        plots=False,
        name='val_auto_v8n_100ep_cvc',
        exist_ok=True
    )
    v8n_100ep_cvc = metrics.seg.map50
    results_log['v8n_100ep_cvc'] = v8n_100ep_cvc
    print(f"[RESULT] CVC Mask mAP@0.5 = {v8n_100ep_cvc}")
    
    # ETIS泛化
    print("\n>>> YOLOv8n-seg 100ep -> ETIS泛化")
    model = YOLO(best_weight)
    metrics = model.val(
        data='configs/etis_larib.yaml',
        split='test',
        imgsz=640,
        batch=16,
        workers=0,
        plots=False,
        name='val_auto_v8n_100ep_etis',
        exist_ok=True
    )
    v8n_100ep_etis = metrics.seg.map50
    results_log['v8n_100ep_etis'] = v8n_100ep_etis
    print(f"[RESULT] ETIS Mask mAP@0.5 = {v8n_100ep_etis}")
    
    # ============================================================
    # 保存完整结果
    # ============================================================
    with open(final_path, 'w') as f:
        json.dump(results_log, f, indent=2)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED - FINAL RESULTS")
    print("="*80)
    
    for key, value in results_log.items():
        if value is not None:
            print(f"  {key}: {value*100:.1f}%")
        else:
            print(f"  {key}: N/A")
    
    def fmt(v):
        if v is None:
            return "N/A"
        return f"{v*100:.1f}%"
    
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
    
    print(f"\n\nDONE! Results saved to {final_path}")

if __name__ == '__main__':
    main()
