import argparse
import csv
import json
import random
import re
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


class ColorCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
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
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        return self.classifier(self.features(x))


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
            }
    return rows


def load_summary(path: Path) -> list[CaptureFrameRecord]:
    rows: list[CaptureFrameRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = SUMMARY_RE.match(line.strip())
        if not m:
            continue
        rows.append(
            CaptureFrameRecord(
                frame_name=m.group(1),
                source_frame=m.group(2),
                symbol_acc=float(m.group(3)),
                pattern_acc=float(m.group(4)),
                color_acc=float(m.group(5)),
                blur=float(m.group(6)),
            )
        )
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


def extract_rgb_patches(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, patch_size: int) -> np.ndarray:
    windows = sliding_window_view(img, (patch_size, patch_size), axis=(0, 1))
    patches = windows[ys, xs]
    return np.ascontiguousarray(patches, dtype=np.uint8)


def split_capture_frames(frame_names: list[str], seed: int) -> tuple[set[str], set[str], set[str]]:
    shuffled = list(frame_names)
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
    source_colors: dict[str, np.ndarray],
    capture_dir: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    patch_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    for record in records:
        img = cv2.imread(str(capture_dir / record.frame_name), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"failed to read image: {capture_dir / record.frame_name}")
        patches = extract_rgb_patches(img, xs, ys, patch_size)
        labels = source_colors[record.source_frame]
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
    return torch.from_numpy(x).contiguous(), torch.from_numpy(y.astype(np.int64, copy=False)).contiguous()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--capture-deskew-dir", required=True)
    parser.add_argument("--source-payload-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="color_cnn")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-symbol-acc", type=float, default=85.0)
    parser.add_argument("--min-pattern-acc", type=float, default=94.0)
    parser.add_argument("--min-color-acc", type=float, default=94.0)
    parser.add_argument("--pattern-bits", type=int, default=4)
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
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload_rows = load_payload_csv(Path(args.source_payload_csv))
    summary_rows = load_summary(Path(args.summary))

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

    source_colors: dict[str, np.ndarray] = {}
    for frame_name, row in payload_rows.items():
        symbols = row["payload_symbols"]
        if row["payload_symbol_count"] != payload_count or len(symbols) != payload_count:
            raise RuntimeError(
                f"source payload count mismatch for {frame_name}: "
                f"csv={row['payload_symbol_count']} parsed={len(symbols)} expected={payload_count}"
            )
        source_colors[frame_name] = np.asarray([sym >> args.pattern_bits for sym in symbols], dtype=np.uint8)

    filtered_records = [
        row for row in summary_rows
        if row.source_frame in source_colors
        and row.symbol_acc >= args.min_symbol_acc
        and row.pattern_acc >= args.min_pattern_acc
        and row.color_acc >= args.min_color_acc
    ]
    if len(filtered_records) < 3:
        raise RuntimeError(f"not enough filtered capture frames: {len(filtered_records)}")

    filtered_records.sort(key=lambda item: item.frame_name)
    frame_names = [row.frame_name for row in filtered_records]
    train_frames, val_frames, test_frames = split_capture_frames(frame_names, args.seed)

    def take_split(frame_set: set[str]) -> list[CaptureFrameRecord]:
        return [record for record in filtered_records if record.frame_name in frame_set]

    train_records = take_split(train_frames)
    val_records = take_split(val_frames)
    test_records = take_split(test_frames)

    capture_dir = Path(args.capture_deskew_dir)
    train_x, train_y = build_split_tensors(train_records, source_colors, capture_dir, xs, ys, patch_size)
    val_x, val_y = build_split_tensors(val_records, source_colors, capture_dir, xs, ys, patch_size)
    test_x, test_y = build_split_tensors(test_records, source_colors, capture_dir, xs, ys, patch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

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
            batch = y.numel()
            running_loss += loss.item() * batch
            running_samples += batch

        train_loss = running_loss / max(running_samples, 1)
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })
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
    summary_path = output_dir / f"{args.output_prefix}_summary.json"

    torch.save(
        {
            "state_dict": best_state,
            "patch_size": patch_size,
            "num_classes": 4,
            "best_epoch": best_epoch,
            "best_val_acc": best_val,
            "test_acc": test_acc,
        },
        pt_path,
    )

    model.eval()
    dummy = torch.zeros(1, 3, patch_size, patch_size, device=device, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    summary_payload = {
        "summary": args.summary,
        "capture_deskew_dir": args.capture_deskew_dir,
        "source_payload_csv": args.source_payload_csv,
        "kept_capture_frames": len(filtered_records),
        "train_capture_frames": [r.frame_name for r in train_records],
        "val_capture_frames": [r.frame_name for r in val_records],
        "test_capture_frames": [r.frame_name for r in test_records],
        "train_samples": int(train_y.numel()),
        "val_samples": int(val_y.numel()),
        "test_samples": int(test_y.numel()),
        "best_epoch": best_epoch,
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "history": history,
        "checkpoint_path": str(pt_path),
        "onnx_path": str(onnx_path),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
