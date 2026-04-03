import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset


SUMMARY_RE = re.compile(
    r"^(frame_\d+\.png) best=(frame_\d+\.png) "
    r"sym=([0-9.]+)% pat=([0-9.]+)% col=([0-9.]+)% blur=([0-9.]+)"
)


@dataclass
class CaptureFrameRecord:
    frame_name: str
    source_frame: str
    symbol_acc: float
    pattern_acc: float
    color_acc: float
    blur: float
    localized: bool
    localize_source: str


class PatternCNN(nn.Module):
    def __init__(self, num_classes: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
        x = (x - mean) / std
        x = self.features(x)
        return self.classifier(x)


def round_div(num: int, den: int) -> int:
    return (num + den // 2) // den


def parse_hex_symbols(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    return [int(item, 16) for item in text.split()]


def load_payload_csv(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["frame_name"]] = {
                "payload_symbol_count": int(row["payload_symbol_count"]),
                "payload_symbols": parse_hex_symbols(row["payload_symbols_hex"]),
                "blur_score": float(row["blur_score"]),
                "localized": row["localized"] == "1",
                "localize_source": row["localize_source"],
            }
    return rows


def load_summary(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = SUMMARY_RE.match(line.strip())
        if not m:
            continue
        rows[m.group(1)] = {
            "source_frame": m.group(2),
            "symbol_acc": float(m.group(3)),
            "pattern_acc": float(m.group(4)),
            "color_acc": float(m.group(5)),
            "blur": float(m.group(6)),
        }
    return rows


def build_payload_positions(
    grid_r: int,
    grid_c: int,
    margin: int,
    stride: int,
    sample_pad: int,
    anchor_reserved_cells: int,
    calib_row: int,
    calib_col_begin: int,
    calib_col_end: int,
    header_row: int,
    header_col_begin: int,
    header_col_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[int] = []
    ys: list[int] = []
    for r in range(grid_r):
        for c in range(grid_c):
            in_left = c < anchor_reserved_cells
            in_right = c >= grid_c - anchor_reserved_cells
            in_top = r < anchor_reserved_cells
            in_bottom = r >= grid_r - anchor_reserved_cells
            if (in_top and in_left) or (in_top and in_right) or (in_bottom and in_left) or (in_bottom and in_right):
                continue
            if r == calib_row and calib_col_begin <= c < calib_col_end:
                continue
            if r == header_row and header_col_begin <= c < header_col_end:
                continue
            xs.append(margin + c * stride - sample_pad)
            ys.append(margin + r * stride - sample_pad)
    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.int32)


def extract_patches(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, patch_size: int) -> np.ndarray:
    windows = sliding_window_view(img, (patch_size, patch_size))
    return np.ascontiguousarray(windows[ys, xs], dtype=np.uint8)


def split_source_frames(source_frames: list[str], seed: int) -> tuple[set[str], set[str], set[str]]:
    shuffled = list(source_frames)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(n * 0.7))
    n_val = max(1, int(n * 0.15))
    train = set(shuffled[:n_train])
    val = set(shuffled[n_train:n_train + n_val])
    test = set(shuffled[n_train + n_val :])
    if not test:
        test = {shuffled[-1]}
        val = set(shuffled[n_train:-1])
    if not val:
        val = {shuffled[-2]} if len(shuffled) >= 2 else set(test)
    return train, val, test


def build_split_tensors(
    records: list[CaptureFrameRecord],
    source_patterns: dict[str, np.ndarray],
    capture_dir: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    patch_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    for record in records:
        img_path = capture_dir / record.frame_name
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"failed to read image: {img_path}")
        patches = extract_patches(img, xs, ys, patch_size)
        labels = source_patterns[record.source_frame]
        if patches.shape[0] != labels.shape[0]:
            raise RuntimeError(
                f"patch/label count mismatch for {record.frame_name}: {patches.shape[0]} vs {labels.shape[0]}"
            )
        patch_chunks.append(patches)
        label_chunks.append(labels)
    if not patch_chunks:
        raise RuntimeError("no patches selected for split")
    x = np.concatenate(patch_chunks, axis=0)
    y = np.concatenate(label_chunks, axis=0)
    x = torch.from_numpy(x).unsqueeze(1).contiguous()
    y = torch.from_numpy(y.astype(np.int64, copy=False)).contiguous()
    return x, y


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--capture-deskew-dir", default="")
    parser.add_argument("--source-payload-csv", default="")
    parser.add_argument("--capture-payload-csv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="pattern_cnn_se110")
    parser.add_argument("--init-pt", default="")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-symbol-acc", type=float, default=75.0)
    parser.add_argument("--min-pattern-acc", type=float, default=75.0)
    parser.add_argument("--min-color-acc", type=float, default=95.0)
    parser.add_argument("--aspect-num", type=int, default=16)
    parser.add_argument("--aspect-den", type=int, default=9)
    parser.add_argument("--short-edge-patterns", type=int, default=110)
    parser.add_argument("--margin", type=int, default=9)
    parser.add_argument("--stride", type=int, default=9)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--sample-pad", type=int, default=2)
    parser.add_argument("--anchor-reserved-cells", type=int, default=6)
    parser.add_argument("--calib-row", type=int, default=0)
    parser.add_argument("--calib-col-begin", type=int, default=6)
    parser.add_argument("--calib-col-end", type=int, default=14)
    parser.add_argument("--header-row", type=int, default=0)
    parser.add_argument("--header-col-begin", type=int, default=14)
    parser.add_argument("--header-col-end", type=int, default=46)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary) if args.summary else artifacts_dir / "summary.txt"
    capture_dir = Path(args.capture_deskew_dir) if args.capture_deskew_dir else artifacts_dir / "capture_deskewed"
    source_payload_csv = Path(args.source_payload_csv) if args.source_payload_csv else artifacts_dir / "source_payloads.csv"
    capture_payload_csv = Path(args.capture_payload_csv) if args.capture_payload_csv else artifacts_dir / "capture_payloads.csv"

    source_payload_rows = load_payload_csv(source_payload_csv)
    capture_payload_rows = load_payload_csv(capture_payload_csv)
    summary_rows = load_summary(summary_path)

    grid_r = args.short_edge_patterns
    grid_c = round_div(args.short_edge_patterns * args.aspect_num, args.aspect_den)
    patch_size = args.tile_size + args.sample_pad * 2
    xs, ys = build_payload_positions(
        grid_r=grid_r,
        grid_c=grid_c,
        margin=args.margin,
        stride=args.stride,
        sample_pad=args.sample_pad,
        anchor_reserved_cells=args.anchor_reserved_cells,
        calib_row=args.calib_row,
        calib_col_begin=args.calib_col_begin,
        calib_col_end=args.calib_col_end,
        header_row=args.header_row,
        header_col_begin=args.header_col_begin,
        header_col_end=args.header_col_end,
    )
    payload_count = xs.shape[0]

    source_patterns: dict[str, np.ndarray] = {}
    for frame_name, row in source_payload_rows.items():
        symbols = row["payload_symbols"]
        if row["payload_symbol_count"] != payload_count or len(symbols) != payload_count:
            raise RuntimeError(
                f"source payload count mismatch for {frame_name}: "
                f"csv={row['payload_symbol_count']} parsed={len(symbols)} expected={payload_count}"
            )
        source_patterns[frame_name] = np.asarray([sym & 0x0F for sym in symbols], dtype=np.uint8)

    filtered_records: list[CaptureFrameRecord] = []
    rejected = 0
    for frame_name, summary in summary_rows.items():
        capture_row = capture_payload_rows.get(frame_name)
        if capture_row is None:
            rejected += 1
            continue
        if summary["source_frame"] not in source_patterns:
            rejected += 1
            continue
        if capture_row["payload_symbol_count"] != payload_count:
            rejected += 1
            continue
        if not capture_row["localized"]:
            rejected += 1
            continue
        if (
            summary["symbol_acc"] < args.min_symbol_acc
            or summary["pattern_acc"] < args.min_pattern_acc
            or summary["color_acc"] < args.min_color_acc
        ):
            rejected += 1
            continue
        filtered_records.append(
            CaptureFrameRecord(
                frame_name=frame_name,
                source_frame=summary["source_frame"],
                symbol_acc=summary["symbol_acc"],
                pattern_acc=summary["pattern_acc"],
                color_acc=summary["color_acc"],
                blur=summary["blur"],
                localized=capture_row["localized"],
                localize_source=capture_row["localize_source"],
            )
        )

    if not filtered_records:
        raise RuntimeError("no capture frames survived filtering")

    filtered_records.sort(key=lambda item: item.frame_name)
    kept_by_source: dict[str, int] = defaultdict(int)
    for record in filtered_records:
        kept_by_source[record.source_frame] += 1

    source_frames = sorted(kept_by_source)
    train_sources, val_sources, test_sources = split_source_frames(source_frames, args.seed)

    def take_split(source_set: set[str]) -> list[CaptureFrameRecord]:
        return [record for record in filtered_records if record.source_frame in source_set]

    train_records = take_split(train_sources)
    val_records = take_split(val_sources)
    test_records = take_split(test_sources)

    train_x, train_y = build_split_tensors(train_records, source_patterns, capture_dir, xs, ys, patch_size)
    val_x, val_y = build_split_tensors(val_records, source_patterns, capture_dir, xs, ys, patch_size)
    test_x, test_y = build_split_tensors(test_records, source_patterns, capture_dir, xs, ys, patch_size)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatternCNN().to(device)
    init_pt_path = Path(args.init_pt) if args.init_pt else None
    if init_pt_path:
        checkpoint = torch.load(init_pt_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = -1.0
    best_state = None
    best_epoch = 0

    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_samples = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = y.numel()
            running_loss += loss.item() * batch_size
            running_samples += batch_size

        train_loss = running_loss / max(running_samples, 1)
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training failed to produce a checkpoint")

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    print(f"best_epoch={best_epoch} best_val_acc={best_val:.4f} test_acc={test_acc:.4f}", flush=True)

    pt_path = output_dir / f"{args.output_prefix}.pt"
    onnx_path = output_dir / f"{args.output_prefix}.onnx"
    summary_out = output_dir / "training_summary.json"
    split_out = output_dir / "split_frames.csv"

    torch.save(
        {
            "state_dict": best_state,
            "grid_r": grid_r,
            "grid_c": grid_c,
            "patch_size": patch_size,
            "sample_pad": args.sample_pad,
            "tile_size": args.tile_size,
            "num_classes": 16,
            "best_epoch": best_epoch,
            "best_val_acc": best_val,
            "test_acc": test_acc,
        },
        pt_path,
    )

    model.eval()
    dummy = torch.zeros(1, 1, patch_size, patch_size, device=device, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    with split_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "capture_frame",
                "source_frame",
                "symbol_acc",
                "pattern_acc",
                "color_acc",
                "blur",
                "localize_source",
            ]
        )
        for split_name, records in (("train", train_records), ("val", val_records), ("test", test_records)):
            for record in records:
                writer.writerow(
                    [
                        split_name,
                        record.frame_name,
                        record.source_frame,
                        f"{record.symbol_acc:.3f}",
                        f"{record.pattern_acc:.3f}",
                        f"{record.color_acc:.3f}",
                        f"{record.blur:.3f}",
                        record.localize_source,
                    ]
                )

    summary_payload = {
        "artifacts_dir": str(artifacts_dir),
        "summary_path": str(summary_path),
        "capture_dir": str(capture_dir),
        "grid_r": grid_r,
        "grid_c": grid_c,
        "patch_size": patch_size,
        "payload_count": payload_count,
        "filters": {
            "min_symbol_acc": args.min_symbol_acc,
            "min_pattern_acc": args.min_pattern_acc,
            "min_color_acc": args.min_color_acc,
        },
        "kept_capture_frames": len(filtered_records),
        "rejected_capture_frames": rejected,
        "kept_source_frames": len(source_frames),
        "split": {
            "train_sources": sorted(train_sources),
            "val_sources": sorted(val_sources),
            "test_sources": sorted(test_sources),
            "train_capture_frames": len(train_records),
            "val_capture_frames": len(val_records),
            "test_capture_frames": len(test_records),
            "train_samples": int(train_y.numel()),
            "val_samples": int(val_y.numel()),
            "test_samples": int(test_y.numel()),
        },
        "best_epoch": best_epoch,
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "history": history,
        "init_pt_path": str(init_pt_path) if init_pt_path else "",
        "checkpoint_path": str(pt_path),
        "onnx_path": str(onnx_path),
    }
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
