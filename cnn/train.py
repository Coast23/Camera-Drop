import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class PatchDataset(Dataset):
    def __init__(self, rows, root: Path):
        self.rows = rows
        self.root = root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = cv2.imread(str(self.root / row["patch_path"]), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)
        label = int(row["src_pattern"])
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


class SmallCNN(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def split_rows(rows):
    source_frames = sorted({row["source_frame"] for row in rows})
    random.Random(42).shuffle(source_frames)
    n = len(source_frames)
    n_train = max(1, int(n * 0.7))
    n_val = max(1, int(n * 0.15))
    train_frames = set(source_frames[:n_train])
    val_frames = set(source_frames[n_train:n_train + n_val])
    test_frames = set(source_frames[n_train + n_val:])
    if not test_frames:
        test_frames = set(source_frames[-1:])
        val_frames = set(source_frames[n_train:n - 1])
    train = [r for r in rows if r["source_frame"] in train_frames]
    val = [r for r in rows if r["source_frame"] in val_frames]
    test = [r for r in rows if r["source_frame"] in test_frames]
    return train, val, test


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--onnx-out", required=True)
    parser.add_argument("--pt-out", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    with (dataset_dir / "labels.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if int(r["src_pattern"]) >= 0]
    train_rows, val_rows, test_rows = split_rows(rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(PatchDataset(train_rows, dataset_dir), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(PatchDataset(val_rows, dataset_dir), batch_size=args.batch_size)
    test_loader = DataLoader(PatchDataset(test_rows, dataset_dir), batch_size=args.batch_size)

    best_val = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        val_acc = evaluate(model, val_loader, device)
        print(f"epoch={epoch} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    print(f"test_acc={test_acc:.4f}")

    pt_out = Path(args.pt_out)
    pt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, pt_out)

    model.eval()
    dummy = torch.randn(1, 1, 12, 12, device=device)
    onnx_out = Path(args.onnx_out)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_out),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
    )


if __name__ == "__main__":
    main()
