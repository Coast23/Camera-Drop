import argparse
import csv
import cv2
import numpy as np
import re
import shutil
import subprocess
from pathlib import Path


SUMMARY_RE = re.compile(
    r"^(?P<capture>frame_\d+\.png)\s+best=(?P<source>frame_\d+\.png)\s+"
    r"sym=(?P<sym>[0-9.]+)%\s+pat=(?P<pat>[0-9.]+)%\s+col=(?P<col>[0-9.]+)%\s+blur=(?P<blur>[0-9.]+)"
)


def load_frame_map(summary_path: Path):
    frame_map = {}
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        m = SUMMARY_RE.match(line.strip())
        if not m:
            continue
        frame_map[m.group("capture")] = {
            "source": m.group("source"),
            "sym": float(m.group("sym")),
            "pat": float(m.group("pat")),
            "col": float(m.group("col")),
            "blur": float(m.group("blur")),
        }
    return frame_map


def load_pattern_masks(pattern_dir: Path):
    masks = []
    for path in sorted(pattern_dir.glob("*.png")):
        try:
            idx = int(path.stem, 16)
        except ValueError:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (8, 8):
            img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_NEAREST)
        mask = (img > 0).astype(np.uint8)
        masks.append((idx, mask))
    masks.sort(key=lambda x: x[0])
    return masks


def is_anchor_reserved(r, c, grid_r, grid_c, anchor_reserved_cells):
    if r < anchor_reserved_cells and c < anchor_reserved_cells:
        return True
    if r < anchor_reserved_cells and c >= grid_c - anchor_reserved_cells:
        return True
    if r >= grid_r - anchor_reserved_cells and c < anchor_reserved_cells:
        return True
    if r >= grid_r - anchor_reserved_cells and c >= grid_c - anchor_reserved_cells:
        return True
    return False


def is_calibration_cell(r, c, calib_row, calib_col_begin, calib_col_end):
    return r == calib_row and calib_col_begin <= c < calib_col_end


def extract_gray_patch12(img, x, y, tile_size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    patch_size = tile_size + 4
    sx = max(0, min(gray.shape[1] - patch_size, x - 2))
    sy = max(0, min(gray.shape[0] - patch_size, y - 2))
    return gray[sy:sy + patch_size, sx:sx + patch_size].copy()


def decode_source_symbols(source_img, pattern_masks, margin, stride, tile_size,
                          anchor_reserved_cells, calib_row, calib_col_begin, calib_col_end):
    palette = np.array([
        [0, 255, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 255],
    ], dtype=np.float32)
    grid_r = (source_img.shape[0] - 2 * margin) // stride
    grid_c = (source_img.shape[1] - 2 * margin) // stride
    symbols = []
    max_pattern_dist = 0
    for r in range(grid_r):
        for c in range(grid_c):
            if is_anchor_reserved(r, c, grid_r, grid_c, anchor_reserved_cells):
                continue
            if is_calibration_cell(r, c, calib_row, calib_col_begin, calib_col_end):
                continue
            x = margin + c * stride
            y = margin + r * stride
            tile = source_img[y:y + tile_size, x:x + tile_size]
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            mask = (gray > 8).astype(np.uint8)
            on_pixels = tile[mask > 0]
            if on_pixels.size == 0:
                raise RuntimeError(f"empty source tile at r={r} c={c}")
            mean_bgr = on_pixels.reshape(-1, 3).mean(axis=0, dtype=np.float32)
            color_idx = int(np.argmin(np.sum((palette - mean_bgr) ** 2, axis=1)))

            best_pat = None
            best_dist = 1 << 30
            for pat_idx, pat_mask in pattern_masks:
                dist = int(np.abs(mask.astype(np.int16) - pat_mask.astype(np.int16)).sum())
                if dist < best_dist:
                    best_dist = dist
                    best_pat = pat_idx
            max_pattern_dist = max(max_pattern_dist, best_dist)
            symbols.append({
                "row": r,
                "col": c,
                "symbol": int((color_idx << 4) | best_pat),
                "pattern": int(best_pat),
                "color": int(color_idx),
            })
    return symbols, max_pattern_dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recognizer-diff", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--pattern-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-symbol-acc", type=float, default=75.0)
    parser.add_argument("--min-pattern-acc", type=float, default=75.0)
    parser.add_argument("--min-color-acc", type=float, default=95.0)
    parser.add_argument("--margin", type=int, default=9)
    parser.add_argument("--stride", type=int, default=9)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--anchor-reserved-cells", type=int, default=6)
    parser.add_argument("--calib-row", type=int, default=0)
    parser.add_argument("--calib-col-begin", type=int, default=6)
    parser.add_argument("--calib-col-end", type=int, default=14)
    args = parser.parse_args()

    recognizer_diff = Path(args.recognizer_diff)
    source_dir = Path(args.source_dir)
    capture_dir = Path(args.capture_dir)
    pattern_masks = load_pattern_masks(Path(args.pattern_dir))
    frame_map = load_frame_map(Path(args.summary))
    out_dir = Path(args.out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_patch_dir = out_dir / "patches"
    merged_patch_dir.mkdir(parents=True, exist_ok=True)
    merged_csv_path = out_dir / "labels.csv"

    capture_files = sorted(capture_dir.glob("frame_*.png"))
    if args.limit > 0:
        capture_files = capture_files[:args.limit]

    with merged_csv_path.open("w", newline="", encoding="utf-8") as merged_csv:
        writer = csv.writer(merged_csv)
        writer.writerow([
            "patch_path",
            "source_frame",
            "capture_frame",
            "row",
            "col",
            "src_symbol",
            "src_pattern",
            "src_color",
            "cap_symbol",
            "cap_pattern",
            "cap_color",
            "symbol_ok",
            "pattern_ok",
            "color_ok",
            "frame_symbol_acc",
            "frame_pattern_acc",
            "frame_color_acc",
        ])

        for idx, capture_path in enumerate(capture_files, start=1):
            mapping = frame_map.get(capture_path.name)
            if mapping is None:
                print(f"[skip] no summary mapping for {capture_path.name}")
                continue
            if mapping["sym"] < args.min_symbol_acc or mapping["pat"] < args.min_pattern_acc or mapping["col"] < args.min_color_acc:
                print(
                    f"[skip] {capture_path.name} sym={mapping['sym']:.3f} "
                    f"pat={mapping['pat']:.3f} col={mapping['col']:.3f}"
                )
                continue

            stem = capture_path.stem
            source_path = source_dir / mapping["source"]
            sample_out = out_dir / f"samples" / stem
            sample_out.mkdir(parents=True, exist_ok=True)

            cmd = [
                str(recognizer_diff),
                "--source", str(source_path),
                "--capture", str(capture_path),
                "--patterns", str(args.pattern_dir),
                "--out", str(sample_out / "viz"),
                "--dump-patches", str(sample_out / "dump"),
            ]
            subprocess.run(cmd, check=True)

            labels_path = sample_out / "dump" / "labels.csv"
            with labels_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    src_patch = sample_out / "dump" / row["patch_path"]
                    patch_name = f"{stem}_{Path(row['patch_path']).name}"
                    dst_patch = merged_patch_dir / patch_name
                    shutil.copy2(src_patch, dst_patch)
                    writer.writerow([
                        Path("patches") / patch_name,
                        mapping["source"],
                        capture_path.name,
                        row["row"],
                        row["col"],
                        row["src_symbol"],
                        row["src_pattern"],
                        row["src_color"],
                        row["cap_symbol"],
                        row["cap_pattern"],
                        row["cap_color"],
                        row["symbol_ok"],
                        row["pattern_ok"],
                        row["color_ok"],
                        mapping["sym"],
                        mapping["pat"],
                        mapping["col"],
                    ])

            print(
                f"[{idx}/{len(capture_files)}] exported {capture_path.name} "
                f"<- {mapping['source']} sym={mapping['sym']:.3f} pat={mapping['pat']:.3f}"
            )


if __name__ == "__main__":
    main()
