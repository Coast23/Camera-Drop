"""
gen_combined_data.py  —  生成合并训练数据集（单模型 5 类，detect 格式）

类别：
  0  camera_drop_frame  整个 camera-drop 码包围盒
  1  qr_code            二维码整体包围盒
  2  anchor             camera-drop TL/TR/BL 三个普通锚点（白黑白黑）
  3  anchor_br          camera-drop BR 锚点（彩黑彩黑，用于方向识别）
  4  qr_finder          二维码定位图案（每个二维码 3 个）

标签格式：YOLO detect  cls cx cy w h（归一化，bbox 为透视变换后的外接矩形）

用法：
  python train/gen_data.py --template build/templates --out train/dataset --n 12000
"""

import cv2
import numpy as np
import random
import argparse
from pathlib import Path

TEMPLATE_SIZE = 1024
ANCHOR_START  = 2
ANCHOR_SIZE   = 56

# (x, y, w, h, class_id)
TEMPLATE_ANCHORS = [
    (ANCHOR_START,                          ANCHOR_START,                          ANCHOR_SIZE, ANCHOR_SIZE, 2),  # TL
    (TEMPLATE_SIZE-ANCHOR_START-ANCHOR_SIZE, ANCHOR_START,                          ANCHOR_SIZE, ANCHOR_SIZE, 2),  # TR
    (ANCHOR_START,                          TEMPLATE_SIZE-ANCHOR_START-ANCHOR_SIZE, ANCHOR_SIZE, ANCHOR_SIZE, 2),  # BL
    (TEMPLATE_SIZE-ANCHOR_START-ANCHOR_SIZE, TEMPLATE_SIZE-ANCHOR_START-ANCHOR_SIZE, ANCHOR_SIZE, ANCHOR_SIZE, 3),  # BR anchor_br
]




def warp_frame(img, max_persp=0.35):
    h, w = img.shape[:2]
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    d = min(w, h) * max_persp
    dst = src + np.random.uniform(-d, d, (4,2)).astype(np.float32)
    x_min, y_min = dst.min(axis=0)
    dst -= [x_min, y_min]
    out_w, out_h = int(dst[:,0].max())+1, int(dst[:,1].max())+1
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h))
    mask   = cv2.warpPerspective(np.full((h,w),255,np.uint8), M, (out_w, out_h))
    return warped, mask, M


def random_bg(h, w):
    t = random.randint(0, 3)
    if t == 0:
        return np.full((h,w,3), np.random.randint(30,210,3).tolist(), np.uint8)
    elif t == 1:
        bg = np.zeros((h,w,3), np.uint8)
        for c in range(3):
            v1,v2 = np.random.randint(20,200,2)
            bg[:,:,c] = np.tile(np.linspace(v1,v2,w,dtype=np.uint8),(h,1))
        return bg
    elif t == 2:
        bg = np.zeros((h,w,3), np.uint8)
        for c in range(3):
            v1,v2 = np.random.randint(20,200,2)
            bg[:,:,c] = np.linspace(v1,v2,h,dtype=np.uint8).reshape(-1,1)
        return bg
    else:
        noise = np.random.randint(40,180,(max(1,h//8),max(1,w//8),3),np.uint8)
        return cv2.GaussianBlur(cv2.resize(noise,(w,h),cv2.INTER_LINEAR),(31,31),0)


def augment(img):
    img = np.clip(img.astype(np.float32)*random.uniform(0.55,1.45)+random.randint(-50,50),0,255).astype(np.uint8)
    if random.random() < 0.45:
        img = cv2.GaussianBlur(img, (random.choice([3,5,7]),)*2, 0)
    if random.random() < 0.20:
        flat = img.ravel()
        idx  = np.random.randint(0, flat.size, int(flat.size*random.uniform(0.0005,0.005)))
        flat[idx] = np.random.choice([0,255], len(idx))
    return img


def place(bg, warped, mask, px, py):
    oh, ow = warped.shape[:2]
    roi = bg[py:py+oh, px:px+ow]
    m3 = cv2.merge([mask,mask,mask])
    roi[:] = np.where(m3>128, warped, roi)


def scale_and_place(warped, mask, bg_w, bg_h):
    out_w, out_h = warped.shape[1], warped.shape[0]
    post_sc = 1.0
    if out_w > bg_w*0.92 or out_h > bg_h*0.92:
        post_sc = min(bg_w*0.88/out_w, bg_h*0.88/out_h)
        out_w = max(40, int(out_w*post_sc))
        out_h = max(40, int(out_h*post_sc))
        warped = cv2.resize(warped, (out_w, out_h))
        mask   = cv2.resize(mask,   (out_w, out_h))
    px = random.randint(0, max(0, bg_w-out_w))
    py = random.randint(0, max(0, bg_h-out_h))
    return warped, mask, px, py, post_sc


def det_label(cls, x, y, w, h, sx, sy, M, psc, px, py, bw, bh, min_px=8):
    """YOLO detect 标签：cls cx cy w h（归一化，外接轴对齐矩形）"""
    x0, y0 = x * sx, y * sy
    x1, y1 = (x + w) * sx, (y + h) * sy
    pts = np.array([[[x0,y0]], [[x1,y0]], [[x1,y1]], [[x0,y1]]], np.float32)
    dst = cv2.perspectiveTransform(pts, M).reshape(-1, 2) * psc + [px, py]
    dst[:, 0] = np.clip(dst[:, 0], 0, bw)
    dst[:, 1] = np.clip(dst[:, 1], 0, bh)
    xA, xB = dst[:, 0].min(), dst[:, 0].max()
    yA, yB = dst[:, 1].min(), dst[:, 1].max()
    if xB - xA < min_px or yB - yA < min_px:
        return None
    cx = ((xA + xB) / 2) / bw
    cy = ((yA + yB) / 2) / bh
    rw = (xB - xA) / bw
    rh = (yB - yA) / bh
    return f"{cls} {cx:.6f} {cy:.6f} {rw:.6f} {rh:.6f}"


def gen_camera_drop(template, bg_w, bg_h):
    scale = random.uniform(0.50, 0.92)   # 最小 0.50 确保 anchor 在训练图中 ≥20px
    tw = max(80, int(template.shape[1]*scale))
    th = max(80, int(template.shape[0]*scale))
    sx, sy = tw/template.shape[1], th/template.shape[0]
    resized = cv2.resize(template, (tw, th))

    warped, mask, M = warp_frame(resized, random.uniform(0.04, 0.35))
    warped, mask, px, py, psc = scale_and_place(warped, mask, bg_w, bg_h)

    labels = []
    # class 0: 整帧
    lbl = det_label(0, 0,0,template.shape[1],template.shape[0], sx,sy,M,psc,px,py,bg_w,bg_h, min_px=40)
    if lbl: labels.append(lbl)
    # class 2: TL/TR/BL 普通锚点；class 3: BR anchor_br
    # min_px=18 确保 anchor 足够大才标注，避免训练到噪点级别的小框
    for (ax,ay,aw,ah,cls) in TEMPLATE_ANCHORS:
        lbl = det_label(cls, ax,ay,aw,ah, sx,sy,M,psc,px,py,bg_w,bg_h, min_px=18)
        if lbl: labels.append(lbl)

    return warped, mask, px, py, labels


def gen_qr(bg_w, bg_h):
    try:
        import qrcode
        from PIL import Image as PILImage
    except ImportError:
        return None

    data = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=random.randint(8,40)))
    bs   = random.randint(6, 14)   # pixel per module
    bdr  = 4                        # quiet zone modules
    qr   = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=bs, border=bdr)
    qr.add_data(data); qr.make(fit=True)
    n = qr.modules_count  # modules without quiet zone

    pil = qr.make_image(fill_color='black', back_color='white').convert('RGB')
    qr_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    qr_h, qr_w = qr_img.shape[:2]

    warped, mask, M = warp_frame(qr_img, random.uniform(0.02, 0.25))
    warped, mask, px, py, psc = scale_and_place(warped, mask, bg_w, bg_h)

    labels = []
    lbl = det_label(1, 0,0,qr_w,qr_h, 1,1,M,psc,px,py,bg_w,bg_h, min_px=30)
    if lbl: labels.append(lbl)
    finders = [
        (bdr*bs,        bdr*bs,        7*bs, 7*bs),   # TL
        ((bdr+n-7)*bs,  bdr*bs,        7*bs, 7*bs),   # TR
        (bdr*bs,        (bdr+n-7)*bs,  7*bs, 7*bs),   # BL
    ]
    for (fx,fy,fw,fh) in finders:
        lbl = det_label(4, fx,fy,fw,fh, 1,1,M,psc,px,py,bg_w,bg_h, min_px=8)
        if lbl: labels.append(lbl)

    return warped, mask, px, py, labels



def covered_ratio(lbl, ox1, oy1, ox2, oy2, bg_w, bg_h):
    parts = lbl.split()
    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    ax1, ay1 = (cx - bw/2) * bg_w, (cy - bh/2) * bg_h
    ax2, ay2 = (cx + bw/2) * bg_w, (cy + bh/2) * bg_h
    ix1, iy1 = max(ax1, ox1), max(ay1, oy1)
    ix2, iy2 = min(ax2, ox2), min(ay2, oy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    return ((ix2 - ix1) * (iy2 - iy1)) / a_area if a_area > 0 else 0.0


def filter_occluded(anchor_lbls, distractor_bboxes, bg_w, bg_h, threshold=0.4):
    result = []
    for lbl in anchor_lbls:
        ok = True
        for (ox1, oy1, ox2, oy2) in distractor_bboxes:
            if covered_ratio(lbl, ox1, oy1, ox2, oy2, bg_w, bg_h) > threshold:
                ok = False
                break
        if ok:
            result.append(lbl)
    return result


def generate_sample(templates, bg_imgs, bg_w, bg_h):
    template = random.choice(templates)   # 每个样本随机选一张模板
    if bg_imgs and random.random() < 0.5:
        bg_src = random.choice(bg_imgs)
        bh2,bw2 = bg_src.shape[:2]
        sc = max(bg_w/bw2, bg_h/bh2)
        big = cv2.resize(bg_src, (int(bw2*sc), int(bh2*sc)))
        x0 = random.randint(0, max(0,big.shape[1]-bg_w))
        y0 = random.randint(0, max(0,big.shape[0]-bg_h))
        bg = big[y0:y0+bg_h, x0:x0+bg_w].copy()
    else:
        bg = random_bg(bg_h, bg_w)

    all_labels = []
    r = random.random()

    if r < 0.60:
        # camera-drop 主样本
        w, m, px, py, lbls = gen_camera_drop(template, bg_w, bg_h)
        place(bg, w, m, px, py)

        # 将帧 label 和锚点 label 分开，以便后续做遮挡过滤
        frame_lbls  = [l for l in lbls if l.startswith('0 ')]
        anchor_lbls = [l for l in lbls if not l.startswith('0 ')]
        all_labels += frame_lbls

        distractor_bboxes = []   # [(x1,y1,x2,y2), ...]

        # 30% 加 QR 干扰
        if random.random() < 0.30:
            res = gen_qr(bg_w, bg_h)
            if res:
                w2, m2, px2, py2, lbls2 = res
                place(bg, w2, m2, px2, py2)
                all_labels += lbls2
                distractor_bboxes.append((px2, py2, px2 + w2.shape[1], py2 + w2.shape[0]))

        # 过滤被干扰物遮挡的锚点 label（遮挡面积 > 40% 则丢弃）
        all_labels += filter_occluded(anchor_lbls, distractor_bboxes, bg_w, bg_h)

    else:
        # 纯 QR 码样本
        res = gen_qr(bg_w, bg_h)
        if res: w2,m2,px2,py2,lbls2 = res; place(bg,w2,m2,px2,py2); all_labels+=lbls2

    return augment(bg), all_labels



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template",    default="test/build",
                    help="模板图像路径或包含多张模板的目录")
    ap.add_argument("--out",         default="test/scripts/combined_dataset")
    ap.add_argument("--n",           type=int,   default=12000)
    ap.add_argument("--val_ratio",   type=float, default=0.15)
    ap.add_argument("--bg_w",        type=int,   default=640)
    ap.add_argument("--bg_h",        type=int,   default=640)
    ap.add_argument("--backgrounds", default="")
    ap.add_argument("--seed",        type=int,   default=2026)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    tp = Path(args.template)
    if tp.is_dir():
        template_paths = sorted(tp.glob("encoded_*.png"))
        templates = [cv2.imread(str(p)) for p in template_paths]
        templates = [t for t in templates if t is not None]
    else:
        t = cv2.imread(str(tp))
        templates = [t] if t is not None else []

    if not templates:
        print(f"错误：未在 {args.template!r} 找到有效模板（encoded_*.png）"); return 1
    print(f"模板数量: {len(templates)}  尺寸: {templates[0].shape[1]}×{templates[0].shape[0]}")

    try:
        import qrcode; print("qrcode 库: OK")
    except ImportError:
        print("警告：未安装 qrcode，QR 类别将跳过。运行: pip install qrcode[pil]")

    bg_imgs = []
    if args.backgrounds:
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            for p in Path(args.backgrounds).glob(ext):
                im = cv2.imread(str(p))
                if im is not None: bg_imgs.append(im)
        print(f"背景图片: {len(bg_imgs)} 张")

    out_dir = Path(args.out)
    for split in ("train","val"):
        (out_dir/"images"/split).mkdir(parents=True, exist_ok=True)
        (out_dir/"labels"/split).mkdir(parents=True, exist_ok=True)

    n_val   = max(1, int(args.n * args.val_ratio))
    n_train = args.n - n_val

    for split, n in [("train", n_train), ("val", n_val)]:
        print(f"\n生成 {n} 个 {split} 样本...")
        for j in range(n):
            img, labels = generate_sample(templates, bg_imgs, args.bg_w, args.bg_h)
            name = f"{j:05d}"
            cv2.imwrite(str(out_dir/"images"/split/f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            (out_dir/"labels"/split/f"{name}.txt").write_text("\n".join(labels)+"\n" if labels else "")
            if (j+1) % 1000 == 0 or j == n-1:
                print(f"  {j+1}/{n}")

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: 5\n"
        f"names:\n"
        f"  0: camera_drop_frame\n"
        f"  1: qr_code\n"
        f"  2: anchor\n"
        f"  3: anchor_br\n"
        f"  4: qr_finder\n"
        f"task: detect\n"
    )
    print(f"\n完成！数据集: {out_dir}/  训练:{n_train}  验证:{n_val}")
    print(f"YAML: {yaml_path}")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
