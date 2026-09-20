import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path(__import__("os").environ["UACR_LEGACY_ROOT"])
EFFICIENTAD_ROOT = ROOT / "tools" / "EfficientAD"
sys.path.insert(0, str(EFFICIENTAD_ROOT))

from models import AutoEncoder, Student, Teacher  # noqa: E402


class MVTLikeTestDataset(Dataset):
    def __init__(self, root: Path, category: str, resize: int = 256):
        self.root = root
        self.category = category
        self.test_root = root / category / "test"
        self.samples = []
        for defect_dir in sorted(p for p in self.test_root.iterdir() if p.is_dir()):
            label = 0 if defect_dir.name == "good" else 1
            for path in sorted(defect_dir.iterdir()):
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.samples.append((path, defect_dir.name, label))
        self.transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, defect_type, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return {
            "image": self.transform(image),
            "path": str(path),
            "defect_type": defect_type,
            "label": label,
        }


class EfficientADInference:
    def __init__(self, ckpt_dir: Path, category: str, resize: int = 256, channel: int = 384):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.category = category
        self.resize = resize
        self.channel = channel
        self.score_in_mid_size = int(0.9 * resize)
        self.teacher = Teacher("S").to(self.device).eval()
        self.student = Student("S").to(self.device).eval()
        self.ae = AutoEncoder().to(self.device).eval()
        self.teacher.load_state_dict(torch.load(ckpt_dir / "best_teacher.pth", map_location=self.device))
        self.student.load_state_dict(torch.load(ckpt_dir / f"{category}_student.pth", map_location=self.device))
        self.ae.load_state_dict(torch.load(ckpt_dir / f"{category}_autoencoder.pth", map_location=self.device))
        q = np.load(ckpt_dir / f"{category}_quantiles.npy", allow_pickle=True).item()
        self.qa_st = torch.tensor(q["qa_st"], device=self.device)
        self.qb_st = torch.tensor(q["qb_st"], device=self.device)
        self.qa_ae = torch.tensor(q["qa_ae"], device=self.device)
        self.qb_ae = torch.tensor(q["qb_ae"], device=self.device)
        self.channel_std = torch.tensor(q["std"], device=self.device)
        self.channel_mean = torch.tensor(q["mean"], device=self.device)

    @torch.inference_mode()
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        teacher_output = self.teacher(images)
        student_output = self.student(images)
        ae_output = self.ae(images)
        y_st = student_output[:, : self.channel, :, :]
        y_stae = student_output[:, -self.channel :, :, :]
        normal_teacher_output = (teacher_output - self.channel_mean) / self.channel_std
        distance_st = torch.pow(normal_teacher_output - y_st, 2)
        distance_stae = torch.pow(ae_output - y_stae, 2)
        fmap_st = torch.mean(distance_st, dim=1, keepdim=True)
        fmap_stae = torch.mean(distance_stae, dim=1, keepdim=True)
        fmap_st = F.interpolate(fmap_st, size=(self.resize, self.resize), mode="bilinear")
        fmap_stae = F.interpolate(fmap_stae, size=(self.resize, self.resize), mode="bilinear")
        normalized_mst = (0.1 * (fmap_st - self.qa_st)) / (self.qb_st - self.qa_st)
        normalized_mae = (0.1 * (fmap_stae - self.qa_ae)) / (self.qb_ae - self.qa_ae)
        combined_map = 0.5 * normalized_mst + 0.5 * normalized_mae
        start = (self.resize - self.score_in_mid_size) // 2
        scores = torch.amax(
            combined_map[:, :, start : start + self.score_in_mid_size, start : start + self.score_in_mid_size],
            dim=(1, 2, 3),
        )
        return scores.detach().cpu()


def best_thresholds(labels: np.ndarray, scores: np.ndarray) -> dict:
    thresholds = np.unique(scores)
    rows = []
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "accuracy": float((pred == labels).mean()),
                "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
                "f1": float(f1_score(labels, pred, zero_division=0)),
            }
        )
    return {
        "best_accuracy": max(rows, key=lambda x: x["accuracy"]),
        "best_balanced": max(rows, key=lambda x: x["balanced_accuracy"]),
        "best_f1": max(rows, key=lambda x: x["f1"]),
    }


def evaluate_category(dataset_name: str, root: Path, ckpt_root: Path, category: str, out_dir: Path, batch_size: int) -> dict:
    dataset = MVTLikeTestDataset(root, category)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    infer = EfficientADInference(ckpt_root / category, category)
    labels, scores, paths, types = [], [], [], []
    start = time.perf_counter()
    for batch in loader:
        batch_scores = infer.score_batch(batch["image"])
        labels.extend([int(x) for x in batch["label"]])
        scores.extend([float(x) for x in batch_scores])
        paths.extend(batch["path"])
        types.extend(batch["defect_type"])
    elapsed = time.perf_counter() - start
    labels_np = np.asarray(labels)
    scores_np = np.asarray(scores)
    thrs = best_thresholds(labels_np, scores_np)
    threshold = thrs["best_balanced"]["threshold"]
    pred = (scores_np >= threshold).astype(int)
    correct = pred == labels_np
    rows = []
    for p, t, y, s, yp, ok in zip(paths, types, labels, scores, pred, correct):
        rows.append(
            {
                "dataset": dataset_name,
                "category": category,
                "path": p,
                "defect_type": t,
                "label": y,
                "score": s,
                "threshold_balanced": threshold,
                "pred": int(yp),
                "correct": int(ok),
                "abs_margin_to_threshold": abs(s - threshold),
                "signed_margin_score_minus_threshold": s - threshold,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / f"{category}_scores.csv", index=False)
    return {
        "dataset": dataset_name,
        "category": category,
        "n": int(len(labels_np)),
        "n_good": int((labels_np == 0).sum()),
        "n_defect": int((labels_np == 1).sum()),
        "auroc": float(roc_auc_score(labels_np, scores_np)) if len(set(labels)) > 1 else float("nan"),
        "ap": float(average_precision_score(labels_np, scores_np)) if len(set(labels)) > 1 else float("nan"),
        "best_balanced_threshold": threshold,
        "best_balanced_accuracy": float(thrs["best_balanced"]["balanced_accuracy"]),
        "accuracy_at_balanced_threshold": float(correct.mean()),
        "best_accuracy": float(thrs["best_accuracy"]["accuracy"]),
        "best_accuracy_threshold": float(thrs["best_accuracy"]["threshold"]),
        "errors": int((~correct).sum()),
        "elapsed_sec": elapsed,
        "sec_per_image": elapsed / max(1, len(labels_np)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ckpt_root = Path(args.ckpt_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    summary = []
    for category in categories:
        row = evaluate_category(args.dataset, data_root, ckpt_root, category, out_dir, args.batch_size)
        summary.append(row)
        print("DONE", json.dumps(row), flush=True)
    pd.DataFrame(summary).to_csv(out_dir / "summary_by_category.csv", index=False)
    (out_dir / "summary_by_category.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    all_errors = []
    for category in categories:
        df = pd.read_csv(out_dir / f"{category}_scores.csv")
        all_errors.append(df[df["correct"] == 0])
    if all_errors:
        pd.concat(all_errors, ignore_index=True).to_csv(out_dir / "all_errors.csv", index=False)


if __name__ == "__main__":
    main()
