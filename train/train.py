"""
训练 YOLOv8n 检测 Camera-Drop 帧，完成后导出 ONNX。

用法：
  python scripts/train.py --data scripts/dataset/dataset.yaml

训练结果保存在 runs/detect/camera_drop/weights/best.pt
ONNX 模型：runs/detect/camera_drop/weights/best.onnx
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset.yaml 路径")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0",
                    help="GPU 索引（如 '0'），无 GPU 时填 'cpu'")
    ap.add_argument("--name", default="camera_drop")
    ap.add_argument("--patience", type=int, default=20,
                    help="Early stopping 耐心轮次")
    ap.add_argument("--workers", type=int, default=8,
                    help="Dataloader 进程数")
    ap.add_argument("--cache", default="ram",
                    help="图像缓存：ram / disk / False")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    print(f"开始训练：epochs={args.epochs}  batch={args.batch}  imgsz={args.imgsz}")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        cache=args.cache,
        exist_ok=True,
    )

    # 导出 ONNX（供 C++ OpenCV DNN 使用）
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n训练完成，最优模型：{best_pt}")
    print("导出 ONNX ...")

    model_best = YOLO(str(best_pt))
    model_best.export(
        format="onnx",
        opset=12,
        simplify=True,
        imgsz=args.imgsz,
    )
    onnx_path = best_pt.with_suffix(".onnx")
    print(f"ONNX 已导出：{onnx_path}")
    print(f"\n在 C++ 中使用：")
    print(f"  anchor_scanner.exe image.jpg {onnx_path}")


if __name__ == "__main__":
    main()
