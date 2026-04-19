"""
QP-TTA Detection-Level Adaptation for RT-DETR
==============================================
Attention/query-based test-time adaptation using source prototype bank.
Only adapts the last decoder layer — architecture-specific to RT-DETR.

Updatable parameters (last decoder layer only):
  - cross_attn.sampling_offsets  (Linear 256→192)
  - cross_attn.attention_weights (Linear 256→96)
  - norm1, norm2, norm3          (LayerNorm)
  + batch-local query_delta      (ephemeral, discarded per batch)

Losses:
  L_con  = 1.0 · InfoNCE contrastive  (target queries ↔ source prototypes)
  L_attn = 0.5 · KL divergence        (attention weight alignment)
  L_ent  = 0.1 · Binary entropy        (foreground confidence sharpening)

Box evaluation uses ultralytics model.val() — identical pipeline to
eval_rtdetr.py baseline, ensuring directly comparable metrics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

RTDETR_WEIGHTS = "runs/detect/main_rtdetr_100ep_server3/weights/best.pt"
BANK_PATH = "results/artifacts/qptta/source_bank_kvasir_train.pt"

DATASETS = {
    "kvasir": {
        "img_dir": "datasets/kvasir_seg/images/val",
        "gt_dir":  "kvasir-seg/masks",
    },
    "cvc": {
        "img_dir": "datasets/cvc_clinicdb/images/test",
        "gt_dir":  "archive/TIF/Ground Truth",
    },
    "etis": {
        "img_dir": "datasets/etis_larib/images/test",
        "gt_dir":  "datasets/ETIS-Larib/masks",
    },
}

# Detection YAML configs — same as eval_rtdetr.py
EVAL_CONFIGS = {
    "kvasir": ("configs/kvasir_det.yaml",       "val"),
    "cvc":    ("configs/cvc_clinicdb_det.yaml",  "test"),
    "etis":   ("configs/etis_larib_det.yaml",    "test"),
}


# ─── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QP-TTA for RT-DETR detection.")
    p.add_argument("--weights",   default=RTDETR_WEIGHTS)
    p.add_argument("--bank",      default=BANK_PATH)
    p.add_argument("--datasets",  nargs="+", default=["cvc", "etis"],
                   choices=list(DATASETS.keys()))
    p.add_argument("--batch",     type=int, default=8)
    p.add_argument("--adapt-steps", type=int, default=3)
    p.add_argument("--lr",        type=float, default=1e-5,
                   help="LR for model params")
    p.add_argument("--query-lr",  type=float, default=5e-3,
                   help="LR for query_delta")
    p.add_argument("--k",         type=int, default=8,
                   help="Top-k positives for retrieval")
    p.add_argument("--n-neg",     type=int, default=32,
                   help="Negatives per query")
    p.add_argument("--target-conf-thres", type=float, default=0.50)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--imgsz",     type=int, default=640)
    p.add_argument("--device",    default="")
    p.add_argument("--output",    default="results/summary/qptta_rtdetr_box_results.json")
    p.add_argument("--log-dir",   default="results/logs")
    return p.parse_args()


# ─── Image loading ────────────────────────────────────────────────────

def load_images_tensor(img_dir: str | Path, imgsz: int = 640) -> torch.Tensor:
    img_dir = Path(img_dir)
    files = sorted(
        f for f in img_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".png", ".tif", ".jpeg", ".bmp"}
    )
    tensors = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = cv2.resize(img, (imgsz, imgsz))
        img = img[:, :, ::-1].copy()   # BGR → RGB
        img = img.transpose(2, 0, 1)   # HWC → CHW
        tensors.append(torch.from_numpy(img.astype(np.float32) / 255.0))
    return torch.stack(tensors)


# ─── Box evaluation (identical pipeline to eval_rtdetr.py) ────────────

def evaluate_box(model, data_yaml: str, split: str,
                 project: str, name: str) -> dict[str, float]:
    """
    Evaluate box mAP using ultralytics val() — same pipeline as
    eval_rtdetr.py baseline, ensuring directly comparable metrics.
    """
    r = model.val(
        data=data_yaml,
        split=split,
        imgsz=640,
        batch=16,
        workers=0,
        plots=True,
        save_json=True,
        project=project,
        name=name,
        exist_ok=True,
    )
    return {
        "precision":  float(r.box.mp),
        "recall":     float(r.box.mr),
        "map50":      float(r.box.map50),
        "map50_95":   float(r.box.map),
    }


# ─── QP-TTA Core ─────────────────────────────────────────────────────

class QPTTAEngine:
    """
    Query-Prototype Test-Time Adaptation engine for RT-DETR.

    Hooks the last decoder layer's cross-attention to:
    1. Capture q_in, attention_weights, context (with grad)
    2. Inject ephemeral query_delta into the last layer's input
    3. Compute contrastive + KL + entropy losses
    4. Update only allowed parameters
    """

    def __init__(self, torch_model: nn.Module, bank: dict,
                 args: argparse.Namespace):
        self.torch_model = torch_model
        self.args = args
        self.device = next(torch_model.parameters()).device

        # Load bank tensors
        self.bank_queries = bank["queries"].float().to(self.device)
        self.bank_attn_wts = bank["attn_weights"].float().to(self.device)
        self.bank_queries_norm = F.normalize(self.bank_queries, dim=-1)

        # Identify decoder components
        self.head = torch_model.model[-1]       # RTDETRDecoder
        self.decoder = self.head.decoder
        self.last_layer = self.decoder.layers[-1]
        self.cross_attn = self.last_layer.cross_attn

        # Save original forward methods
        self._orig_cross_fwd = self.cross_attn.forward
        self._orig_decoder_fwd = self.decoder.forward

        # Captured tensors (populated during patched forward)
        self.captured: dict[str, torch.Tensor] = {}

    # ── Parameter management ──

    def _get_updatable_params(self) -> list[nn.Parameter]:
        params = []
        params.extend(self.cross_attn.sampling_offsets.parameters())
        params.extend(self.cross_attn.attention_weights.parameters())
        params.extend(self.last_layer.norm1.parameters())
        params.extend(self.last_layer.norm2.parameters())
        params.extend(self.last_layer.norm3.parameters())
        return list(params)

    def _freeze_all(self):
        for p in self.torch_model.parameters():
            p.requires_grad_(False)

    def _unfreeze_updatable(self):
        for p in self._get_updatable_params():
            p.requires_grad_(True)

    # ── Patching ──

    def _patch_cross_attn(self):
        """Patch MSDeformAttn to capture internals WITH gradient flow."""
        ca = self.cross_attn
        cap = self.captured

        def hooked_fwd(query, refer_bbox, value, value_shapes, value_mask=None):
            bs, len_q = query.shape[:2]
            len_v = value.shape[1]

            cap["q_in"] = query  # keep in graph

            vp = ca.value_proj(value)
            if value_mask is not None:
                vp = vp.masked_fill(value_mask[..., None], 0.0)
            vp = vp.view(bs, len_v, ca.n_heads, ca.d_model // ca.n_heads)

            so = ca.sampling_offsets(query).view(
                bs, len_q, ca.n_heads, ca.n_levels, ca.n_points, 2)
            aw = ca.attention_weights(query).view(
                bs, len_q, ca.n_heads, ca.n_levels * ca.n_points)
            aw = F.softmax(aw, -1).view(
                bs, len_q, ca.n_heads, ca.n_levels, ca.n_points)

            cap["attention_weights"] = aw  # keep in graph for KL

            np_ = refer_bbox.shape[-1]
            if np_ == 2:
                on = torch.as_tensor(value_shapes, dtype=query.dtype,
                                     device=query.device).flip(-1)
                sl = refer_bbox[:, :, None, :, None, :] + \
                     so / on[None, None, None, :, None, :]
            elif np_ == 4:
                sl = refer_bbox[:, :, None, :, None, :2] + \
                     so / ca.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            else:
                raise ValueError(f"Invalid ref dim {np_}")

            cap["sampling_locations"] = sl  # P3: keep for TTA.md consistency

            from ultralytics.nn.modules.utils import multi_scale_deformable_attn_pytorch
            out = multi_scale_deformable_attn_pytorch(vp, value_shapes, sl, aw)
            result = ca.output_proj(out)
            cap["context"] = result  # keep in graph
            return result

        ca.forward = hooked_fwd

    def _patch_decoder(self, query_delta: torch.Tensor):
        """
        Patch the decoder's forward to inject query_delta at the last layer
        and capture dec_scores.
        """
        decoder = self.decoder
        cap = self.captured

        def patched_fwd(embed, refer_bbox, feats, shapes,
                        bbox_head, score_head, pos_mlp,
                        attn_mask=None, padding_mask=None):
            from ultralytics.nn.modules.utils import inverse_sigmoid

            output = embed
            dec_bboxes, dec_cls = [], []
            last_refined_bbox = None
            rb = refer_bbox.sigmoid()

            for i, layer in enumerate(decoder.layers):
                if i == len(decoder.layers) - 1:
                    output = output + query_delta

                output = layer(output, rb, feats, shapes,
                               padding_mask, attn_mask, pos_mlp(rb))

                bbox = bbox_head[i](output)
                refined = torch.sigmoid(bbox + inverse_sigmoid(rb))

                if decoder.training:
                    dec_cls.append(score_head[i](output))
                    if i == 0:
                        dec_bboxes.append(refined)
                    else:
                        dec_bboxes.append(torch.sigmoid(
                            bbox + inverse_sigmoid(last_refined_bbox)))
                elif i == decoder.eval_idx:
                    dec_cls.append(score_head[i](output))
                    dec_bboxes.append(refined)
                    cap["dec_scores"] = dec_cls[-1]
                    break

                last_refined_bbox = refined
                rb = refined.detach() if decoder.training else refined

            return torch.stack(dec_bboxes), torch.stack(dec_cls)

        decoder.forward = patched_fwd

    def _unpatch_all(self):
        self.cross_attn.forward = self._orig_cross_fwd
        self.decoder.forward = self._orig_decoder_fwd

    # ── Loss computation ──

    def _compute_losses(self) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute QP-TTA losses from captured decoder internals.

        The delta injection happens inside _patch_decoder (before the last
        decoder layer runs), so q_in already carries the shifted embedding.
        Do NOT add the delta a second time here.
        """
        q_in = self.captured["q_in"]               # [B, 300, 256]
        aw   = self.captured["attention_weights"]   # [B, 300, 8, 3, 4]
        dec_scores = self.captured.get("dec_scores")  # [B, 300, nc]

        if dec_scores is None:
            z = torch.tensor(0.0, device=self.device, requires_grad=True)
            return z, {"L_con": 0, "L_attn": 0, "L_ent": 0, "n_fg": 0}

        fg_probs = dec_scores.squeeze(-1).sigmoid()  # [B, 300]
        fg_mask = fg_probs >= self.args.target_conf_thres

        if not fg_mask.any():
            z = torch.tensor(0.0, device=self.device, requires_grad=True)
            return z, {"L_con": 0, "L_attn": 0, "L_ent": 0, "n_fg": 0}

        fg_idx = fg_mask.nonzero(as_tuple=False)  # [N_fg, 2]
        n_fg = fg_idx.shape[0]

        # q_in is already shifted by the decoder-level delta — use directly
        fg_q = q_in[fg_idx[:, 0], fg_idx[:, 1]]   # [N_fg, 256]
        fg_q_n = F.normalize(fg_q, dim=-1)

        # ── 1. InfoNCE contrastive loss ──
        sim_all = fg_q_n @ self.bank_queries_norm.T  # [N_fg, N_bank]
        topk = min(self.args.k, sim_all.shape[1])
        _, pos_indices = sim_all.topk(topk, dim=-1)

        N_bank = self.bank_queries_norm.shape[0]
        tau = self.args.temperature

        L_con = torch.tensor(0.0, device=self.device)
        for i in range(n_fg):
            pos_keys = self.bank_queries_norm[pos_indices[i]]  # [k, 256]

            mask_pos = torch.zeros(N_bank, dtype=torch.bool, device=self.device)
            mask_pos[pos_indices[i]] = True
            neg_pool = torch.arange(N_bank, device=self.device)[~mask_pos]
            n_neg = min(self.args.n_neg, len(neg_pool))
            neg_idx = neg_pool[torch.randperm(len(neg_pool),
                                               device=self.device)[:n_neg]]
            neg_keys = self.bank_queries_norm[neg_idx]

            pos_logit = (fg_q_n[i] @ pos_keys.T).mean() / tau
            neg_logits = (fg_q_n[i] @ neg_keys.T) / tau

            logits = torch.cat([pos_logit.unsqueeze(0), neg_logits])
            target = torch.zeros(1, dtype=torch.long, device=self.device)
            L_con = L_con + F.cross_entropy(logits.unsqueeze(0), target)

        L_con = L_con / max(n_fg, 1)

        # ── 2. Attention KL divergence ──
        fg_aw = aw[fg_idx[:, 0], fg_idx[:, 1]]   # [N_fg, 8, 3, 4]
        fg_aw_flat = fg_aw.reshape(n_fg, -1)      # [N_fg, 96]

        with torch.no_grad():
            topk_sim, topk_i = (fg_q_n @ self.bank_queries_norm.T
                                ).topk(topk, dim=-1)
            topk_w = F.softmax(topk_sim, dim=-1)

        proto_aw = self.bank_attn_wts[topk_i.reshape(-1)].reshape(
            n_fg, topk, -1)
        weighted_proto = (topk_w.unsqueeze(-1) * proto_aw).sum(dim=1)

        L_attn = F.kl_div(
            F.log_softmax(fg_aw_flat, dim=-1),
            F.softmax(weighted_proto, dim=-1),
            reduction="batchmean")

        # ── 3. Binary entropy ──
        fg_s = fg_probs[fg_mask]
        eps = 1e-7
        L_ent = -(fg_s * (fg_s + eps).log() +
                  (1 - fg_s) * (1 - fg_s + eps).log()).mean()

        # ── Total ──
        L = 1.0 * L_con + 0.5 * L_attn + 0.1 * L_ent

        return L, {
            "L_con": float(L_con), "L_attn": float(L_attn),
            "L_ent": float(L_ent), "n_fg": int(n_fg),
        }

    # ── Public API ──

    def adapt(self, images_tensor: torch.Tensor):
        """Run QP-TTA adaptation on target-domain images."""
        self._freeze_all()
        self._unfreeze_updatable()
        self._patch_cross_attn()

        model_params = self._get_updatable_params()
        optimizer = torch.optim.AdamW(model_params, lr=self.args.lr)

        N = len(images_tensor)
        bs = self.args.batch

        for step in range(self.args.adapt_steps):
            total_loss, n_batches = 0.0, 0
            step_ld = {"L_con": 0.0, "L_attn": 0.0, "L_ent": 0.0, "n_fg": 0}
            perm = torch.randperm(N)

            for start in range(0, N, bs):
                batch = images_tensor[perm[start:start + bs]].to(self.device)
                B = batch.shape[0]

                qd = torch.zeros(B, 300, 256, device=self.device,
                                 requires_grad=True)
                delta_opt = torch.optim.AdamW([qd], lr=self.args.query_lr)

                self.captured.clear()
                self._patch_decoder(qd)

                self.torch_model.eval()
                for p in model_params:
                    p.requires_grad_(True)

                _ = self.torch_model(batch)

                loss, ld = self._compute_losses()

                if loss.requires_grad and loss.item() > 0:
                    optimizer.zero_grad()
                    delta_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_params, 1.0)
                    optimizer.step()
                    delta_opt.step()

                total_loss += loss.item()
                n_batches += 1
                for k in ("L_con", "L_attn", "L_ent", "n_fg"):
                    step_ld[k] += ld[k]

                # Restore decoder forward for next batch
                self.decoder.forward = self._orig_decoder_fwd

            nb = max(n_batches, 1)
            print(f"    QP-TTA step {step + 1}/{self.args.adapt_steps}  "
                  f"loss={total_loss / nb:.4f}  "
                  f"L_con={step_ld['L_con'] / nb:.4f}  "
                  f"L_attn={step_ld['L_attn'] / nb:.4f}  "
                  f"L_ent={step_ld['L_ent'] / nb:.4f}  "
                  f"avg_fg={step_ld['n_fg'] / nb:.1f}")

        self._unpatch_all()
        self.torch_model.eval()


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.chdir(ROOT)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  QP-TTA Detection-Level Adaptation for RT-DETR")
    print("=" * 60)

    bank_path = ROOT / args.bank
    print(f"\n[QP-TTA] Loading bank from {bank_path}")
    bank = torch.load(str(bank_path), map_location="cpu", weights_only=False)
    print(f"[QP-TTA] Bank: {bank['queries'].shape[0]} prototypes")

    from ultralytics import RTDETR

    all_results = {}

    for ds in args.datasets:
        cfg = DATASETS[ds]
        eval_yaml, eval_split = EVAL_CONFIGS[ds]

        print(f"\n{'=' * 60}")
        print(f"  Target domain: {ds.upper()}")
        print(f"{'=' * 60}")

        # Fresh model per dataset
        print(f"  Loading RT-DETR from {args.weights}")
        model = RTDETR(args.weights)
        model.model.to(device)

        # Adapt
        imgs = load_images_tensor(cfg["img_dir"], args.imgsz)
        print(f"  Loaded {len(imgs)} images for adaptation")

        engine = QPTTAEngine(model.model, bank, args)
        engine.adapt(imgs)

        # ── Box evaluation via ultralytics val() ──
        # Uses identical pipeline to eval_rtdetr.py baseline
        print(f"  Evaluating with {eval_yaml} ({eval_split}) ...")
        metrics = evaluate_box(
            model, eval_yaml, eval_split,
            project=str(ROOT / "runs" / "detect"),
            name=f"qptta_rtdetr_{ds}",
        )
        print(f"  [QP-TTA] mAP50={metrics['map50'] * 100:.2f}%  "
              f"P={metrics['precision'] * 100:.2f}%  "
              f"R={metrics['recall'] * 100:.2f}%  "
              f"mAP50-95={metrics['map50_95'] * 100:.2f}%")

        all_results[ds] = {"qptta_box": metrics}

        # Log
        log_dir = ROOT / args.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f"qptta_{ds}.jsonl", "a") as f:
            f.write(json.dumps({"dataset": ds, "metrics": metrics}) + "\n")

        del model
        torch.cuda.empty_cache()

    # Save summary
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n[QP-TTA] Summary saved → {out_path}")


if __name__ == "__main__":
    main()
