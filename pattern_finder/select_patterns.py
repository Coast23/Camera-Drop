"""
select_patterns.py  —  为 camera-drop 码筛选最优 8×8 pattern 组合

思路：
  两阶段筛选：
  Stage 1（快）：Hamming 距离预筛 → 从巨大候选池中保留 top-K 个互相距离最大的候选
  Stage 2（准）：完整帧 end-to-end 多档位缩放真实评测
    - 对候选池随机采样 --trials 组（每组 n 个 pattern）
    - 每组：编码完整 1024×1024 帧 → scale 缩小→还原 → 信道模拟 → 解码 → pattern 准确率
    - 取得分最高的组合为最终结果
    - 档位 EVAL_SCALES（1.0~0.5），低分辨率档权重更高

候选池：
  1. 当前系统的 4×4→8×8 expanded patterns
  2. cimbar 已有的 16 个手工 pattern
  3. 从 Windows 字体渲染的 Unicode 符号
  4. GNU Unifont (.hex.gz)：57,000+ 字符，切成 8×8 patch

用法：
  python scripts/select_patterns.py --n 16 --save
  python scripts/select_patterns.py --n 16 --save --workers 8 --prefilter 500 --trials 200
"""

import cv2
import numpy as np
import argparse
import os
import json
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

# ── 参数 ─────────────────────────────────────────────────────────────────────
TILE_SIZE   = 8
RENDER_SIZE = 32
MIN_FILL    = 10
MAX_FILL    = 54

# Stage 2 多档位缩放评测：在 1.0 到 0.5 之间均匀取档位
# 权重越低分辨率越大（低分辨率更难区分，给更高权重）
EVAL_SCALES  = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
EVAL_WEIGHTS = [1,   2,   3,   4,   5,   6  ]   # 低分辨率档给更高权重
EVAL_SCORE_WEIGHTS = [0.74, 0.14, 0.12, 0.06, 0.03, 20.0, 44.0, 0.04, 0.08, 0.06, 0.09]
DEFAULT_WORKERS = min(16, max(1, cpu_count() - 1))

FONT_SOURCES = [
    ("C:/Windows/Fonts/webdings.ttf",   list(range(0xF020, 0xF0FF))),
    ("C:/Windows/Fonts/wingding.ttf",   list(range(0xF020, 0xF0FF))),
    ("C:/Windows/Fonts/WINGDNG2.TTF",   list(range(0xF020, 0xF0FF))),
    ("C:/Windows/Fonts/WINGDNG3.TTF",   list(range(0xF020, 0xF0FF))),
    ("C:/Windows/Fonts/seguisym.ttf",
        list(range(0x2500, 0x2600))     # Box drawing
      + list(range(0x25A0, 0x2600))     # Geometric shapes
      + list(range(0x2700, 0x27C0))     # Dingbats
      + list(range(0x2B00, 0x2BFF))),   # Misc symbols
    ("C:/Windows/Fonts/seguiemj.ttf",   list(range(0x1F300, 0x1F600))),
    ("C:/Windows/Fonts/SegoeIcons.ttf", list(range(0xE000, 0xE200))),
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. 候选 pattern 生成
# ══════════════════════════════════════════════════════════════════════════════

def gen_our_system_patterns():
    """生成当前系统的 16 个 pattern（4×4 Hamming-max 集合 → 2×2 expand → 8×8）"""
    NUM = 16
    def popcnt(x): return bin(x).count('1')
    cand = [i for i in range(1 << 16) if 6 <= popcnt(i) <= 10]
    pick = [0x00FF]
    dist = [popcnt(c ^ pick[0]) for c in cand]
    for _ in range(NUM - 1):
        best = max(range(len(cand)), key=lambda i: dist[i])
        pick.append(cand[best])
        dist = [min(dist[i], popcnt(cand[i] ^ pick[-1])) for i in range(len(cand))]

    def expand(m16):
        tile = np.zeros((8, 8), np.uint8)
        for r in range(4):
            for c in range(4):
                if (m16 >> (r * 4 + c)) & 1:
                    tile[r*2:r*2+2, c*2:c*2+2] = 1
        return tile

    return [('our_' + format(m, '04x'), expand(m)) for m in pick]


def gen_cimbar_patterns():
    """读取 cimbar 已有的 16 个 pattern（R channel: 0=前景）"""
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "cfc-master" / "app" / "src" / "cpp" / "libcimbar" / "bitmap" / "4",
        root / "libcimbar-master" / "bitmap" / "4",
        Path("D:/jiwang/1/libcimbar-master/bitmap/4"),
    ]
    base = next((p for p in candidates if p.exists()), None)
    if base is None:
        print(f"  [跳过] cimbar 目录不存在: {[str(p) for p in candidates]}")
        return []
    patterns = []
    for name in sorted(os.listdir(base)):
        if not name.endswith('.png'): continue
        img = cv2.imread(str(base / name), cv2.IMREAD_UNCHANGED)
        if img is None: continue
        ch = img[:, :, 2] if img.ndim == 3 and img.shape[2] >= 3 else img
        mask = (ch < 128).astype(np.uint8)
        patterns.append(('cimbar_' + name[:-4], mask))
    return patterns


def gen_native_8x8_patterns():
    """
    从两个来源加载真正的原生 8×8 点阵字形：
      1. font8x8（GitHub dhepper/font8x8）：ASCII/Latin/Greek/Box/Hiragana 等，公有领域
      2. Linux kbd PSF1 字体（kernel.org kbd 包）：拉丁/阿拉伯/西里尔/希伯来等，GPL

    只保留 fill 在 [MIN_FILL, MAX_FILL] 范围内的唯一字形。
    返回 list[(name, mask_8x8_uint8)]
    """
    import urllib.request, re, tarfile, io, struct

    seen     = set()
    patterns = []
    t0       = time.time()

    def add_glyph(name, row_bytes):
        """row_bytes: 8 个 uint8，每个 bit 对应一列（MSB=左）"""
        arr = np.zeros((8, 8), np.uint8)
        for r, b in enumerate(row_bytes):
            for c in range(8):
                arr[r, c] = (b >> (7 - c)) & 1
        fill = int(arr.sum())
        if fill < MIN_FILL or fill > MAX_FILL:
            return
        key = arr.tobytes()
        if key in seen:
            return
        seen.add(key)
        patterns.append((name, arr))

    # ── 来源1：font8x8（C header，公有领域）────────────────────────
    font8x8_files = {
        'f8_basic':    'https://raw.githubusercontent.com/dhepper/font8x8/master/font8x8_basic.h',
        'f8_latin':    'https://raw.githubusercontent.com/dhepper/font8x8/master/font8x8_ext_latin.h',
        'f8_greek':    'https://raw.githubusercontent.com/dhepper/font8x8/master/font8x8_greek.h',
        'f8_block':    'https://raw.githubusercontent.com/dhepper/font8x8/master/font8x8_block.h',
        'f8_box':      'https://raw.githubusercontent.com/dhepper/font8x8/master/font8x8_box.h',
        'f8_hiragana': 'https://raw.githubusercontent.com/dhepper/font8x8/master/font8x8_hiragana.h',
        'f8_sga':      'https://raw.githubusercontent.com/dhepper/font8x8/master/font8x8_sga.h',
    }
    print("  [1] 下载 font8x8 (dhepper/font8x8)...")
    for tag, url in font8x8_files.items():
        try:
            data = urllib.request.urlopen(url, timeout=15).read().decode()
            blocks = re.findall(r'\{([^}]+)\}', data)
            added = 0
            for idx, block in enumerate(blocks):
                nums = re.findall(r'0x([0-9a-fA-F]{2})', block)
                if len(nums) == 8:
                    add_glyph(f'{tag}_{idx:03d}', [int(x, 16) for x in nums])
                    added += 1
            print(f"    {tag}: {added} 字形")
        except Exception as e:
            print(f"    {tag}: 跳过 ({e})")

    # ── 来源2：kbd PSF1/PSF2 8×8 字体（kernel.org）──────────────────
    PSF1_MAGIC = b'\x36\x04'
    PSF2_MAGIC = b'\x72\xb5\x4a\x86'
    print("  [2] 下载 kbd 字体包 (kernel.org)...")
    try:
        raw_tar = urllib.request.urlopen(
            'https://mirrors.kernel.org/pub/linux/utils/kbd/kbd-2.6.4.tar.gz',
            timeout=60).read()
        tar = tarfile.open(fileobj=io.BytesIO(raw_tar), mode='r:gz')
        for member in tar.getmembers():
            if '08' not in member.name:
                continue
            if not member.name.endswith(('.psf', '.psfu')):
                continue
            raw = tar.extractfile(member).read()
            fname = member.name.split('/')[-1].replace('.psfu', '').replace('.psf', '')
            added = 0
            if raw[:2] == PSF1_MAGIC:
                charsize = raw[3]
                if charsize != 8:
                    continue
                n = 512 if (raw[2] & 1) else 256
                for i in range(n):
                    add_glyph(f'psf_{fname}_{i}', raw[4 + i*8: 4 + (i+1)*8])
                    added += 1
            elif raw[:4] == PSF2_MAGIC:
                hdrsize, _, numglyph, bpg, height, width = struct.unpack('<IIIIII', raw[8:32])
                if height != 8 or width != 8:
                    continue
                for i in range(numglyph):
                    add_glyph(f'psf_{fname}_{i}', raw[hdrsize + i*bpg: hdrsize + (i+1)*bpg])
                    added += 1
            if added:
                print(f"    {fname}: {added} 字形")
    except Exception as e:
        print(f"    kbd 字体包: 跳过 ({e})")

    elapsed = time.time() - t0
    print(f"  原生 8×8 字形共 {len(patterns)} 个（去重+fill过滤），用时 {elapsed:.1f}s")
    return patterns



    """进程池 worker：渲染单个字符，返回 (name, bytes) 或 None"""
    font_path, font_name, cp, render_size, tile_size, min_fill, max_fill = args
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(font_path, size=render_size - 4)
        char = chr(cp)
        tmp = Image.new('L', (render_size * 2, render_size * 2), 255)
        bb = ImageDraw.Draw(tmp).textbbox((render_size // 2, render_size // 2), char, font=font)
        cw, ch = bb[2] - bb[0], bb[3] - bb[1]
        if cw < 2 or ch < 2: return None
        sz = max(cw, ch) + 4
        img = Image.new('L', (sz, sz), 255)
        draw = ImageDraw.Draw(img)
        ox = (sz - cw) // 2 - (bb[0] - render_size // 2)
        oy = (sz - ch) // 2 - (bb[1] - render_size // 2)
        draw.text((ox, oy), char, fill=0, font=font)
        arr = np.array(img.resize((tile_size, tile_size), Image.LANCZOS))
        _, binary = cv2.threshold(arr, 127, 1, cv2.THRESH_BINARY_INV)
        mask = binary.astype(np.uint8)
        fill = int(mask.sum())
        if fill < min_fill or fill > max_fill: return None
        return (f'{font_name}_U{cp:04X}', mask.tobytes())
    except Exception:
        return None


def _render_one(args):
    """进程池 worker：渲染单个字符，返回 (name, bytes) 或 None"""
    font_path, font_name, cp, render_size, tile_size, min_fill, max_fill = args
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(font_path, size=render_size - 4)
        char = chr(cp)
        tmp = Image.new('L', (render_size * 2, render_size * 2), 255)
        bb = ImageDraw.Draw(tmp).textbbox((render_size // 2, render_size // 2), char, font=font)
        cw, ch = bb[2] - bb[0], bb[3] - bb[1]
        if cw < 2 or ch < 2: return None
        sz = max(cw, ch) + 4
        img = Image.new('L', (sz, sz), 255)
        draw = ImageDraw.Draw(img)
        ox = (sz - cw) // 2 - (bb[0] - render_size // 2)
        oy = (sz - ch) // 2 - (bb[1] - render_size // 2)
        draw.text((ox, oy), char, fill=0, font=font)
        arr = np.array(img.resize((tile_size, tile_size), Image.LANCZOS))
        _, binary = cv2.threshold(arr, 127, 1, cv2.THRESH_BINARY_INV)
        mask = binary.astype(np.uint8)
        fill = int(mask.sum())
        if fill < min_fill or fill > max_fill: return None
        return (f'{font_name}_U{cp:04X}', mask.tobytes())
    except Exception:
        return None


def gen_font_patterns(max_per_font=300, workers=None):
    """多进程从字体文件渲染候选 pattern"""
    if workers is None:
        workers = max(1, cpu_count() - 1)

    all_args = []
    for font_path, codepoints in FONT_SOURCES:
        if not os.path.exists(font_path):
            print(f"  [跳过] 字体不存在: {font_path}")
            continue
        font_name = Path(font_path).stem
        for cp in list(codepoints)[:max_per_font * 4]:  # 多取一些备用
            all_args.append((font_path, font_name, cp,
                             RENDER_SIZE, TILE_SIZE, MIN_FILL, MAX_FILL))

    print(f"  渲染 {len(all_args)} 个字符（{workers} 进程）...")
    with Pool(workers) as pool:
        results = pool.map(_render_one, all_args, chunksize=64)

    # 去重 + 按字体统计
    seen = set()
    patterns = []
    font_counts = {}
    for r in results:
        if r is None: continue
        name, mask_bytes = r
        if mask_bytes in seen: continue
        seen.add(mask_bytes)
        font_name = name.split('_')[0]
        if font_counts.get(font_name, 0) >= max_per_font:
            continue
        font_counts[font_name] = font_counts.get(font_name, 0) + 1
        mask = np.frombuffer(mask_bytes, np.uint8).reshape(TILE_SIZE, TILE_SIZE).copy()
        patterns.append((name, mask))

    for fname, cnt in sorted(font_counts.items()):
        print(f"    {fname}: {cnt} 个")
    return patterns


# ══════════════════════════════════════════════════════════════════════════════
# 2. 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _mask_to_bits(mask):
    """8×8 uint8 mask → uint64"""
    flat = mask.flatten().astype(np.uint64)
    result = np.uint64(0)
    for bit in flat:
        result = (result << np.uint64(1)) | bit
    return int(result)


def _hamming(a, b):
    return bin(a ^ b).count('1')


# ══════════════════════════════════════════════════════════════════════════════
# 2.5  Stage 1：Hamming 距离快速预筛（从大候选池缩减到 top-K）
# ══════════════════════════════════════════════════════════════════════════════

def hamming_prefilter(all_patterns, keep=2000):
    """
    用 Hamming 距离贪心保留 keep 个互相"最远"的 pattern。
    全部用 numpy bitwise 向量化，速度极快（21万候选约 3~5 秒）。
    """
    if len(all_patterns) <= keep:
        return all_patterns

    names = [p[0] for p in all_patterns]
    masks = [p[1] for p in all_patterns]
    N = len(masks)

    print(f"  Hamming 预筛: {N} -> {keep} 个 (向量化)...")
    t0 = time.time()

    bits = np.array([_mask_to_bits(m) for m in masks], dtype=np.uint64)

    # 强制保留所有 cimbar pattern（作为种子），再贪心填满
    cimbar_idx = [i for i, nm in enumerate(names) if nm.startswith('cimbar_')]
    selected_idx = list(cimbar_idx) if cimbar_idx else [0]
    print(f"    锁定 {len(selected_idx)} 个 cimbar pattern 作为种子")

    # 初始化 min_dist：对每个点取到已选集合的最小 Hamming 距离
    min_dist = np.full(N, 64, dtype=np.int32)
    for si in selected_idx:
        d = np.array([bin(int(b ^ bits[si])).count('1') for b in bits], dtype=np.int32)
        min_dist = np.minimum(min_dist, d)
        min_dist[si] = -1

    while len(selected_idx) < keep:
        best = int(np.argmax(min_dist))
        selected_idx.append(best)
        min_dist[best] = -1
        new_dists = np.array([bin(int(b ^ bits[best])).count('1') for b in bits], dtype=np.int32)
        min_dist = np.minimum(min_dist, new_dists)
        min_dist[best] = -1

        if len(selected_idx) % 200 == 0:
            print(f"    已选 {len(selected_idx)}/{keep}，当前 min_hamming={min_dist.max()}")

    elapsed = time.time() - t0
    result = [(names[i], masks[i]) for i in selected_idx]
    min_h = min(bin(int(_mask_to_bits(result[i][1]) ^ _mask_to_bits(result[j][1]))).count('1')
                for i in range(min(10, len(result))) for j in range(i+1, min(10, len(result))))
    print(f"  预筛完成，用时 {elapsed:.1f}s，保留 {len(result)} 个，min_hamming(前10)>={min_h}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. Stage 2：完整帧 end-to-end 评测
# ══════════════════════════════════════════════════════════════════════════════

# 与 config_acc_test.cpp 完全一致的布局常量
_GRID_SIZE = 112
_STRIDE    = 9
_MARGIN    = 8
_IMG_SIZE  = 1024

def _encode_frame(masks_list, rng_seed=0):
    """
    用给定的 N 个 pattern（masks_list，每个 8×8 uint8，1=前景）
    编码一张完整的 1024×1024 Camera-Drop 帧（纯图案层，黑底彩色前景）。

    返回 (img_bgr, raw_data_dict)
      img_bgr       : np.ndarray (1024,1024,3) uint8
      raw_data_dict : {(r,c): data_byte}  data_byte 低 P_BITS 位=pattern_idx，高位=color_idx
    """
    N = len(masks_list)
    assert (N & (N - 1)) == 0 and N >= 2
    P_BITS = int(np.log2(N))
    NUM_COLORS = 4
    C_BITS = 2

    # 颜色表（BGR，与 C++ 一致）
    COLORS = [
        (0, 255, 255),   # 0 Yellow
        (0, 255, 0),     # 1 Green
        (255, 255, 0),   # 2 Cyan
        (255, 0, 255),   # 3 Magenta
    ]

    def is_reserved(r, c):
        if r > 105 and c > 105: return True   # BR anchor
        if r == 0 and 6 <= c < 46: return True  # frame header
        return False

    img = np.zeros((_IMG_SIZE, _IMG_SIZE, 3), dtype=np.uint8)
    raw = {}
    rng = np.random.default_rng(rng_seed)
    total_vals = N * NUM_COLORS

    for r in range(_GRID_SIZE):
        for c in range(_GRID_SIZE):
            if is_reserved(r, c):
                continue
            data = int(rng.integers(0, total_vals))
            raw[(r, c)] = data
            pat_idx   = data & (N - 1)
            color_idx = data >> P_BITS
            color_bgr = COLORS[color_idx]
            mask = masks_list[pat_idx]   # 8×8, 1=前景
            sx = _MARGIN + c * _STRIDE
            sy = _MARGIN + r * _STRIDE
            ys, xs = np.where(mask == 1)
            img[sy + ys, sx + xs] = color_bgr

    return img, raw


def _channel_sim(img):
    """
    信道模拟（与 C++ 完全一致）：摩尔纹 → 模糊 → 偏色 → 高斯噪声。
    操作在副本上进行，返回新图像。
    """
    out = img.astype(np.float32)

    # 摩尔纹
    rows = np.arange(_IMG_SIZE, dtype=np.float32)
    cols = np.arange(_IMG_SIZE, dtype=np.float32)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    moire = (0.85 + 0.20 * np.sin(rr * 0.45 + cc * 0.35)).astype(np.float32)
    out *= moire[:, :, np.newaxis]
    out = np.clip(out, 0, 255)

    # 失焦模糊
    out_u8 = out.astype(np.uint8)
    out_u8 = cv2.GaussianBlur(out_u8, (5, 5), 1.2)
    out = out_u8.astype(np.float32)

    # 偏色
    out[:, :, 0] = np.clip(out[:, :, 0] * 0.8 + 50, 0, 255)   # B 衰减
    out[:, :, 1] = np.clip(out[:, :, 1] * 0.9 + 50, 0, 255)   # G 衰减
    out[:, :, 2] = np.clip(out[:, :, 2] * 1.1 + 40, 0, 255)   # R 增强

    # 高斯噪声
    noise = np.random.normal(0, 15, out.shape).astype(np.float32)
    out = np.clip(out + noise, 0, 255)

    return out.astype(np.uint8)


def _decode_frame(img_bgr, masks_list):
    """
    解码一张经过信道模拟后的帧，返回 (correct_set, total_count)。
    correct_set: {(r,c)} 解码正确的格子坐标集合（用于与 raw_data_dict 比对）
    decoded_dict: {(r,c): decoded_byte}
    """
    N = len(masks_list)
    P_BITS = int(np.log2(N))
    NUM_COLORS = 4
    COLORS_BGR = np.array([
        [0, 255, 255],   # 0 Yellow
        [0, 255, 0],     # 1 Green
        [255, 255, 0],   # 2 Cyan
        [255, 0, 255],   # 3 Magenta
    ], dtype=np.float32)

    # 预计算 Dict bits
    dict_bits = [_mask_to_bits(m) for m in masks_list]

    def is_reserved(r, c):
        if r > 105 and c > 105: return True
        if r == 0 and 6 <= c < 46: return True
        return False

    def match_pattern(tile_bits):
        best_i, best_d = 0, 65
        for i, db in enumerate(dict_bits):
            d = bin(tile_bits ^ db).count('1')
            if d < best_d:
                best_d = d; best_i = i
        return best_i

    def match_color(bgr):
        diff = COLORS_BGR - bgr.astype(np.float32)
        dists = (diff ** 2).sum(axis=1)
        return int(np.argmin(dists))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    decoded = {}

    for r in range(_GRID_SIZE):
        for c in range(_GRID_SIZE):
            if is_reserved(r, c):
                continue
            sx = _MARGIN + c * _STRIDE
            sy = _MARGIN + r * _STRIDE
            cell_gray = gray[sy:sy+8, sx:sx+8]
            cell_bgr  = img_bgr[sy:sy+8, sx:sx+8]

            _, bin_cell = cv2.threshold(cell_gray, 0, 255,
                                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            tile_bits = _mask_to_bits((bin_cell > 128).astype(np.uint8))
            best_pat = match_pattern(tile_bits)

            pat_mask = masks_list[best_pat]
            fg = cell_bgr[pat_mask == 1]
            if len(fg) > 0:
                avg_bgr = fg.mean(axis=0)
                best_col = match_color(avg_bgr)
            else:
                best_col = 0

            decoded[(r, c)] = (best_col << P_BITS) | best_pat

    return decoded


def eval_pattern_set(masks_list, scales=None, weights=None, seeds=(0, 1, 2)):
    """
    对给定的 N 个 pattern 做完整 end-to-end 多档位缩放评测。

    流程（每个 seed × 每个 scale）：
      1. encode_frame  → 完整 1024×1024 帧
      2. resize 缩小到 scale 倍 → resize 放大回 1024×1024
      3. channel_sim（摩尔纹+模糊+偏色+噪声）
      4. decode_frame  → 统计 pattern 解码准确率

    返回加权 pattern 准确率（float, 0~1），越高越好。
    """
    if scales  is None: scales  = EVAL_SCALES
    if weights is None: weights = EVAL_WEIGHTS

    total_w = 0.0
    correct_w = 0.0

    for seed in seeds:
        img, raw = _encode_frame(masks_list, rng_seed=seed)
        N = len(masks_list)
        P_BITS = int(np.log2(N))

        for scale, w in zip(scales, weights):
            scaled_px = max(2, int(_IMG_SIZE * scale + 0.5))
            small     = cv2.resize(img,   (scaled_px, scaled_px), interpolation=cv2.INTER_AREA)
            restored  = cv2.resize(small, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            simulated = _channel_sim(restored)
            decoded   = _decode_frame(simulated, masks_list)

            for (r, c), dec_byte in decoded.items():
                exp_byte = raw[(r, c)]
                # 只统计 pattern 部分是否正确（颜色识别受偏色影响，不作为 pattern 筛选标准）
                dec_pat = dec_byte  & (N - 1)
                exp_pat = exp_byte  & (N - 1)
                total_w   += w
                if dec_pat == exp_pat:
                    correct_w += w

    return correct_w / total_w if total_w > 0 else 0.0


def _eval_worker(args):
    """多进程 worker：对一组 pattern 索引做 eval，返回 (indices_tuple, score)"""
    indices, masks_bytes_list, scales, weights, seeds = args
    masks_list = [np.frombuffer(b, np.uint8).reshape(TILE_SIZE, TILE_SIZE)
                  for b in masks_bytes_list]
    score = eval_pattern_set(masks_list, scales=scales, weights=weights, seeds=seeds)
    return (indices, score)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Stage 2：C++ 批量评测 + 迭代局部搜索（ILS）
# ══════════════════════════════════════════════════════════════════════════════

import subprocess

# eval_patterns.exe 的默认路径（可被 main() 的 --eval-exe 覆盖）
def _resolve_default_eval_exe():
    root = Path(__file__).parent.parent
    for cand in (root / "tools" / "tmp_eval" / "eval_patterns.exe",
                 root / "build-user" / "eval_patterns.exe",
                 root / "build" / "eval_patterns.exe"):
        if cand.exists():
            return str(cand)
    return str(root / "tools" / "tmp_eval" / "eval_patterns.exe")


_DEFAULT_EVAL_EXE = _resolve_default_eval_exe()


def eval_cpp_batch(combos_indices, all_masks, scales, weights, seeds, exe_path,
                   batch_size=200, workers=None, on_batch_done=None,
                   better_than=None, timeout_sec=None):
    """
    将若干组 pattern 组合批量送入 eval_patterns.exe 评测（多进程并行）。

    combos_indices : list[tuple[int,...]]
    all_masks      : list[np.ndarray]
    workers        : 并行进程数（默认 CPU 核心数-1）
    on_batch_done  : 可选回调 fn(new_results: dict)，每批完成后调用
    better_than    : float 或 None。若设置，一旦发现任意得分 > better_than，
                     立即取消剩余任务并提前返回（first-improvement 模式）
    返回 dict {indices_tuple: score}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        from tqdm import tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    if workers is None:
        workers = DEFAULT_WORKERS

    # 预计算所有 mask 的 uint64 bits
    bits_cache = {idx: _mask_to_bits(all_masks[idx]) for idx in
                  {i for combo in combos_indices for i in combo}}

    # 切分成 batch_size 大小的子任务
    batches = [combos_indices[s:s + batch_size]
               for s in range(0, len(combos_indices), batch_size)]

    payload_base = {
        "scales":  scales,
        "weights": [float(w) for w in weights],
        "seeds":   list(seeds),
        "score_weights": EVAL_SCORE_WEIGHTS,
    }

    def run_one_batch(batch):
        combos_bits = [[bits_cache[i] for i in combo] for combo in batch]
        inp = json.dumps({**payload_base, "combos": combos_bits})
        run_kwargs = {
            'input': inp,
            'capture_output': True,
            'text': True,
        }
        if timeout_sec is not None and timeout_sec > 0:
            run_kwargs['timeout'] = timeout_sec
        proc = subprocess.run([exe_path], **run_kwargs)
        if proc.returncode != 0:
            raise RuntimeError(f"eval_patterns.exe 失败: {proc.stderr[:300]}")
        scores = json.loads(proc.stdout)["scores"]
        return {tuple(combo): score for combo, score in zip(batch, scores)}

    results = {}
    pbar = tqdm(total=len(combos_indices), unit="combo",
                desc="  C++评测", leave=False) if _has_tqdm else None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one_batch, b): b for b in batches}
        for fut in as_completed(futs):
            batch_res = fut.result()
            results.update(batch_res)
            if on_batch_done:
                on_batch_done(batch_res)
            if pbar:
                pbar.update(len(futs[fut]))
            # first-improvement：发现比阈值更好的，取消剩余任务提前退出
            if better_than is not None and any(v > better_than for v in batch_res.values()):
                for f in futs:
                    f.cancel()
                if pbar:
                    pbar.close()
                return results

    if pbar:
        pbar.close()
    return results


def _find_seed_indices(names, prefix, n):
    """在候选池里按名字前缀找种子，不足 n 时返回 None。"""
    idx = [i for i, nm in enumerate(names) if nm.startswith(prefix + '_')]
    return tuple(sorted(idx[:n])) if len(idx) >= n else None


def _build_diverse_random_start(masks, n, rng):
    bits = [_mask_to_bits(m) for m in masks]
    total = len(bits)
    if total < n:
        return None
    first = int(rng.integers(0, total))
    chosen = [first]
    chosen_set = {first}
    min_dist = np.full(total, 65, dtype=np.int16)
    for i in range(total):
        min_dist[i] = _hamming(bits[i], bits[first])
    min_dist[first] = -1

    while len(chosen) < n:
        order = rng.permutation(total)
        best_idx = -1
        best_score = -1
        for idx in order:
            idx = int(idx)
            if idx in chosen_set:
                continue
            score = int(min_dist[idx])
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            break
        chosen.append(best_idx)
        chosen_set.add(best_idx)
        min_dist[best_idx] = -1
        for i in range(total):
            if i in chosen_set:
                continue
            d = _hamming(bits[i], bits[best_idx])
            if d < min_dist[i]:
                min_dist[i] = d

    return tuple(sorted(chosen)) if len(chosen) == n else None


def build_start_combos(all_patterns, n, random_starts=0, rng_seed=12345):
    names = [p[0] for p in all_patterns]
    masks = [p[1] for p in all_patterns]
    starts = []
    seen = set()

    def add(label, combo):
        if combo is None or len(combo) != n:
            return
        combo = tuple(sorted(combo))
        if combo in seen:
            return
        starts.append((label, combo))
        seen.add(combo)

    add('cimbar', _find_seed_indices(names, 'cimbar', n))
    add('seedbest', _find_seed_indices(names, 'seedbest', n))
    add('our', _find_seed_indices(names, 'our', n))
    add('fallback', tuple(range(n)))

    rng = np.random.default_rng(rng_seed)
    for i in range(max(0, random_starts)):
        add(f'random{i+1}', _build_diverse_random_start(masks, n, rng))

    return starts


def _combo_to_bits_key(combo_indices, bits_list):
    return ','.join(str(bits_list[i]) for i in combo_indices)


def _combo_from_bits_value(bits_value, bits_to_idx, expected_len=None):
    if bits_value is None:
        return None
    if isinstance(bits_value, str):
        parts = [p for p in bits_value.split(',') if p != '']
    elif isinstance(bits_value, (list, tuple)):
        parts = list(bits_value)
    else:
        return None
    try:
        bits_combo = tuple(int(x) for x in parts)
    except (TypeError, ValueError):
        return None
    idx_combo = tuple(sorted(bits_to_idx[b] for b in bits_combo if b in bits_to_idx))
    if len(idx_combo) != len(bits_combo):
        return None
    if expected_len is not None and len(idx_combo) != expected_len:
        return None
    return idx_combo


def _load_checkpoint_results(checkpoint_path, bits_to_idx, expected_len=None):
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return {}, {}, 0
    with open(checkpoint_path, encoding='utf-8') as f:
        raw_ckpt = json.load(f)
    if isinstance(raw_ckpt, dict) and isinstance(raw_ckpt.get('results'), dict):
        raw_results = raw_ckpt.get('results', {})
        meta = raw_ckpt.get('__meta__') or raw_ckpt.get('meta') or {}
    else:
        raw_results = raw_ckpt if isinstance(raw_ckpt, dict) else {}
        meta = {}
    all_results = {}
    skipped = 0
    for k, v in raw_results.items():
        idx_combo = _combo_from_bits_value(k, bits_to_idx, expected_len)
        if idx_combo is None:
            skipped += 1
            continue
        all_results[idx_combo] = v
    return all_results, meta, skipped


def _save_checkpoint(checkpoint_path, all_results, bits_list, meta=None):
    if not checkpoint_path:
        return
    payload = {
        'results': {
            _combo_to_bits_key(k, bits_list): v
            for k, v in all_results.items()
        }
    }
    if meta:
        payload['__meta__'] = meta
    path = Path(checkpoint_path)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    tmp_path.replace(path)


def search_greedy_expand(all_patterns, fixed_indices,
                         target_n,
                         scales=None, weights=None, seeds=(0, 1, 2),
                         exe_path=None, workers=None,
                         checkpoint_path=None, out_dir=None,
                         eval_timeout=None):
    """
    贪心扩展：从 fixed_indices 出发，每轮从候选池中选一个新 pattern 加入，
    使当前组合得分最高，直到组合大小达到 target_n。

    适合 Stage 2A：固定 cimbar 16 个，贪心逐个添加最优的自由 pattern。

    返回 (best_patterns_list, best_score, all_results_dict)
    """
    if scales  is None: scales  = EVAL_SCALES
    if weights is None: weights = EVAL_WEIGHTS
    if exe_path is None: exe_path = _DEFAULT_EVAL_EXE
    if workers is None: workers = DEFAULT_WORKERS
    if not Path(exe_path).exists():
        raise FileNotFoundError(f"找不到 eval_patterns.exe: {exe_path}")

    names = [p[0] for p in all_patterns]
    masks = [p[1] for p in all_patterns]
    N     = len(masks)

    bits_list   = [_mask_to_bits(m) for m in masks]
    bits_to_idx = {b: i for i, b in enumerate(bits_list)}

    all_results, _, skipped = _load_checkpoint_results(checkpoint_path, bits_to_idx)
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"  断点恢复：加载 {len(all_results)} 组（跳过 {skipped} 条）")

    preview_state = {'combo': None}

    def save_ckpt():
        _save_checkpoint(checkpoint_path, all_results, bits_list)

    def save_best_preview():
        if not out_dir or not all_results:
            return
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        best_idx = max(all_results, key=all_results.get)
        combo_key = tuple(best_idx)
        if preview_state['combo'] == combo_key:
            return
        score = all_results[best_idx]
        pts = [(names[i], masks[i]) for i in best_idx]
        try:
            cols = min(len(pts), 8)
            rows = (len(pts) + cols - 1) // cols
            sc, pad = 20, 3
            canvas = np.ones((rows * (TILE_SIZE * sc + pad) + pad,
                              cols * (TILE_SIZE * sc + pad) + pad, 3), np.uint8) * 180
            for idx, (_, mask) in enumerate(pts):
                r, c = divmod(idx, cols)
                y0 = r * (TILE_SIZE * sc + pad) + pad
                x0 = c * (TILE_SIZE * sc + pad) + pad
                tile_big = cv2.resize(((1 - mask) * 255).astype(np.uint8),
                                      (TILE_SIZE * sc, TILE_SIZE * sc),
                                      interpolation=cv2.INTER_NEAREST)
                canvas[y0:y0 + TILE_SIZE * sc, x0:x0 + TILE_SIZE * sc] = cv2.cvtColor(tile_big, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(out_path / 'best_preview.png'), canvas)
            (out_path / 'best_score.txt').write_text(f"{score:.6f}\n", encoding='utf-8')

            best_dir = out_path / 'best'
            best_dir.mkdir(parents=True, exist_ok=True)
            for hex_idx, (pname, mask) in enumerate(pts):
                tile_big = cv2.resize(((1 - mask) * 255).astype(np.uint8),
                                      (TILE_SIZE * sc, TILE_SIZE * sc),
                                      interpolation=cv2.INTER_NEAREST)
                fname = best_dir / f'{hex_idx:02x}.png'
                cv2.imwrite(str(fname), tile_big)
            meta = [{"idx": hex_idx, "name": pname, "fill": int(mask.sum())}
                    for hex_idx, (pname, mask) in enumerate(pts)]
            (best_dir / 'meta.json').write_text(
                json.dumps({"score": score, "patterns": meta}, indent=2, ensure_ascii=False))
            preview_state['combo'] = combo_key
        except OSError as exc:
            print(f"    [warn] skip preview save: {exc}")


    def eval_batch(combos):
        todo = [c for c in combos if c not in all_results]
        if todo:
            def on_done(batch_res):
                all_results.update(batch_res)
                save_ckpt()
                save_best_preview()
            new = eval_cpp_batch(todo, masks, scales, weights, seeds, exe_path,
                                 workers=workers, on_batch_done=on_done,
                                 timeout_sec=eval_timeout)
            all_results.update(new)
        return {c: all_results[c] for c in combos}

    current = list(fixed_indices)
    free_slots = target_n - len(current)
    print(f"\n  贪心扩展：{len(current)} 个 fixed -> {target_n} 个（添加 {free_slots} 个自由 pattern）")

    t_total = time.time()
    for step in range(free_slots):
        used = set(current)
        candidates = [tuple(sorted(current + [i])) for i in range(N) if i not in used]
        print(f"\n  [贪心 Step {step+1}/{free_slots}] 评测 {len(candidates)} 个候选...")
        res = eval_batch(candidates)
        best_combo = max(res, key=res.get)
        best_score = res[best_combo]
        added = [i for i in best_combo if i not in used][0]
        current = list(best_combo)
        print(f"    选入: {names[added]}  得分={best_score:.4f}")
        if out_dir:
            save_patterns([(names[i], masks[i]) for i in current], out_dir)

    elapsed = time.time() - t_total
    best_combo = tuple(sorted(current))
    best_score = all_results[best_combo]
    print(f"\n  贪心完成，用时 {elapsed:.0f}s，最终得分={best_score:.4f}")

    best_patterns = [(names[i], masks[i]) for i in best_combo]
    return best_patterns, best_score, all_results


def search_best_set_ils(all_patterns, n=16,
                        scales=None, weights=None, seeds=(0, 1, 2),
                        exe_path=None, workers=None,
                        fixed_indices=None,
                        start_combo_indices=None,
                        start_patterns=None,
                        bootstrap_results=None,
                        resume_checkpoint_state=True,
                        temp_init=0.02, temp_min=1e-4, cooling=0.92,
                        checkpoint_path=None, out_dir=None,
                        eval_timeout=None):
    """
    迭代局部搜索（ILS）找最优 n-pattern 组合。

    all_patterns   : 候选池 [(name, mask), ...]
    n              : 最终组合大小（16 或 32）
    fixed_indices  : 可选，tuple[int]，这些位置锁定不参与替换（用于32-pattern模式：
                     传入 cimbar 的16个索引，ILS 只搜剩余16个可替换槽位）

    算法：
      Round 0  确定起点（cimbar 原始 → our_system → 候选池前n个）
               若 fixed_indices 不为空，起点 = fixed_indices + 候选池前(n - len(fixed))个非fixed
      Round k  对当前最优组合中【可替换槽位】穷举所有单步替换：
               若有任何一个得分 > 当前最优，接受最佳替换，进入 Round k+1
               若没有提升，停止（局部最优）

    返回 (best_patterns_list, best_score, all_results_dict)
    """
    if scales  is None: scales  = EVAL_SCALES
    if weights is None: weights = EVAL_WEIGHTS
    if exe_path is None: exe_path = _DEFAULT_EVAL_EXE
    if workers is None: workers = DEFAULT_WORKERS
    if not Path(exe_path).exists():
        raise FileNotFoundError(f"找不到 eval_patterns.exe: {exe_path}")

    names  = [p[0] for p in all_patterns]
    masks  = [p[1] for p in all_patterns]
    N      = len(masks)

    fixed_set = set(fixed_indices) if fixed_indices else set()

    # ── 加载断点 ──────────────────────────────────────────────────
    # checkpoint key = 逗号分隔的 uint64 bits（而非 index），保证跨次运行稳定
    bits_list = [_mask_to_bits(m) for m in masks]          # index -> uint64
    bits_to_idx = {b: i for i, b in enumerate(bits_list)}  # uint64 -> index

    all_results = dict(bootstrap_results or {})
    ckpt_results, ckpt_meta, skipped = _load_checkpoint_results(checkpoint_path, bits_to_idx, n)
    all_results.update(ckpt_results)
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"  断点恢复：加载 {len(ckpt_results)} 组（跳过 {skipped} 条不匹配）")

    preview_state = {'combo': None}

    import math, random as _rng
    round_idx = 0
    t_total = time.time()
    temp = temp_init
    best_combo = None
    best_score = None
    best_ever_combo = None
    best_ever_score = None
    sa_temp_scale = 0.35
    backtrack_block_rounds = 1
    blocked_combo = None
    blocked_until_round = -1

    checkpoint_state = {}

    def refresh_ckpt_state():
        checkpoint_state.clear()
        checkpoint_state.update({
            'version': 2,
            'n': n,
            'round_idx': int(round_idx),
            'temp': float(temp),
            'current_combo_bits': [bits_list[i] for i in best_combo] if best_combo else [],
            'current_score': float(best_score) if best_score is not None else None,
            'best_ever_combo_bits': [bits_list[i] for i in best_ever_combo] if best_ever_combo else [],
            'best_ever_score': float(best_ever_score) if best_ever_score is not None else None,
            'blocked_combo_bits': [bits_list[i] for i in blocked_combo] if blocked_combo else [],
            'blocked_until_round': int(blocked_until_round),
        })

    def save_ckpt():
        _save_checkpoint(checkpoint_path, all_results, bits_list, checkpoint_state or None)

    def save_best_preview():
        if not out_dir or not all_results:
            return
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        best_idx = max(all_results, key=all_results.get)
        combo_key = tuple(best_idx)
        if preview_state['combo'] == combo_key:
            return
        score = all_results[best_idx]
        pts = [(names[i], masks[i]) for i in best_idx]
        try:
            cols = min(len(pts), 8)
            rows = (len(pts) + cols - 1) // cols
            sc, pad = 20, 3
            canvas = np.ones((rows * (TILE_SIZE * sc + pad) + pad,
                              cols * (TILE_SIZE * sc + pad) + pad, 3), np.uint8) * 180
            for idx, (_, mask) in enumerate(pts):
                r, c = divmod(idx, cols)
                y0 = r * (TILE_SIZE * sc + pad) + pad
                x0 = c * (TILE_SIZE * sc + pad) + pad
                tile_big = cv2.resize(((1 - mask) * 255).astype(np.uint8),
                                      (TILE_SIZE * sc, TILE_SIZE * sc),
                                      interpolation=cv2.INTER_NEAREST)
                canvas[y0:y0 + TILE_SIZE * sc, x0:x0 + TILE_SIZE * sc] = cv2.cvtColor(tile_big, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(out_path / 'best_preview.png'), canvas)
            (out_path / 'best_score.txt').write_text(f"{score:.6f}\n", encoding='utf-8')

            best_dir = out_path / 'best'
            best_dir.mkdir(parents=True, exist_ok=True)
            for hex_idx, (pname, mask) in enumerate(pts):
                tile_big = cv2.resize(((1 - mask) * 255).astype(np.uint8),
                                      (TILE_SIZE * sc, TILE_SIZE * sc),
                                      interpolation=cv2.INTER_NEAREST)
                fname = best_dir / f'{hex_idx:02x}.png'
                cv2.imwrite(str(fname), tile_big)
            meta = [{"idx": hex_idx, "name": pname, "fill": int(mask.sum())}
                    for hex_idx, (pname, mask) in enumerate(pts)]
            (best_dir / 'meta.json').write_text(
                json.dumps({"score": score, "patterns": meta}, indent=2, ensure_ascii=False))
            preview_state['combo'] = combo_key
        except OSError as exc:
            print(f"    [warn] skip preview save: {exc}")


    def eval_batch(combos, better_than=None):
        """过滤掉已评测的，送 C++ 评测，存入 all_results（每批完成立即写 checkpoint + best_preview）。"""
        todo = [c for c in combos if c not in all_results]
        if todo:
            def on_done(batch_res):
                all_results.update(batch_res)
                save_ckpt()
                save_best_preview()
            new = eval_cpp_batch(todo, masks, scales, weights, seeds, exe_path,
                                 workers=workers, on_batch_done=on_done,
                                 better_than=better_than,
                                 timeout_sec=eval_timeout)
            all_results.update(new)
        return {c: all_results[c] for c in combos if c in all_results}

    restored_current_combo = _combo_from_bits_value(ckpt_meta.get('current_combo_bits'), bits_to_idx, n) if (ckpt_meta and resume_checkpoint_state) else None
    restored_best_combo = _combo_from_bits_value(ckpt_meta.get('best_ever_combo_bits'), bits_to_idx, n) if ckpt_meta else None
    restored_blocked_combo = _combo_from_bits_value(ckpt_meta.get('blocked_combo_bits'), bits_to_idx, n) if (ckpt_meta and resume_checkpoint_state) else None
    restored_round_idx = int(ckpt_meta.get('round_idx', 0)) if (ckpt_meta and resume_checkpoint_state) else 0
    restored_blocked_until = int(ckpt_meta.get('blocked_until_round', -1)) if (ckpt_meta and resume_checkpoint_state) else -1
    try:
        restored_temp = float(ckpt_meta.get('temp', temp_init)) if (ckpt_meta and resume_checkpoint_state) else temp_init
    except (TypeError, ValueError):
        restored_temp = temp_init
    if not np.isfinite(restored_temp):
        restored_temp = temp_init

    # ── 确定起点 / 恢复点 ────────────────────────────────────────
    if restored_current_combo is not None:
        start_combo = restored_current_combo
        print(f"\n  断点恢复: 从 checkpoint 当前组合继续（round={restored_round_idx}, T={restored_temp:.5f}）")
    elif start_combo_indices is not None:
        start_combo = tuple(sorted(start_combo_indices))
        if len(start_combo) != n:
            raise ValueError(f"start_combo_indices 长度 {len(start_combo)}/{n}，请检查")
        print(f"\n  起点: 外部传入 index 组合")
    elif start_patterns is not None:
        # 外部传入起点（例如第一阶段结果），按名字在候选池里找 index
        name_to_idx = {nm: i for i, nm in enumerate(names)}
        start_combo = tuple(sorted(name_to_idx[nm] for nm, _ in start_patterns
                                   if nm in name_to_idx))
        if len(start_combo) != n:
            raise ValueError(f"start_patterns 中有 {len(start_combo)}/{n} 个在候选池内，请检查")
        print(f"\n  起点: 外部传入 {n} 个 pattern")
    elif all_results:
        start_combo = max(all_results, key=all_results.get)
        print(f"\n  断点恢复: 从 checkpoint/缓存 已知最优组合继续")
    elif fixed_indices:
        # 32-pattern 模式：fixed 部分已确定，自由槽用候选池（非fixed）前几个填充
        free_slots = n - len(fixed_indices)
        free_pool  = [i for i in range(N) if i not in fixed_set]
        free_start = tuple(free_pool[:free_slots])
        start_combo = tuple(sorted(set(fixed_indices) | set(free_start)))
        print(f"\n  起点: 固定 {len(fixed_indices)} 个 cimbar + 自由槽 {free_slots} 个")
    else:
        start_combo = (
            _find_seed_indices(names, 'cimbar', n)
            or _find_seed_indices(names, 'seedbest', n)
            or _find_seed_indices(names, 'our', n)
            or tuple(range(n))
        )
    print(f"  起点前4: {[names[i] for i in list(start_combo)[:4]]}...")

    res = eval_batch([start_combo])
    best_combo = start_combo
    best_score = res[start_combo]

    if restored_current_combo is not None:
        round_idx = restored_round_idx
        temp = max(temp_min, restored_temp)
        best_ever_combo = restored_best_combo or max(all_results, key=all_results.get)
        best_ever_score = all_results.get(best_ever_combo, best_score)
        blocked_combo = restored_blocked_combo
        blocked_until_round = restored_blocked_until
        print(f"  恢复当前得分: {best_score:.4f}  历史最优: {best_ever_score:.4f}")
    else:
        print(f"  起点得分: {best_score:.4f}")
        best_ever_combo = restored_best_combo or best_combo
        best_ever_score = all_results.get(best_ever_combo, best_score) if all_results else best_score
        if best_ever_score < best_score:
            best_ever_combo = best_combo
            best_ever_score = best_score

    refresh_ckpt_state()
    save_ckpt()
    if out_dir and best_ever_combo is not None:
        current_patterns = [(names[i], masks[i]) for i in best_ever_combo]
        save_patterns(current_patterns, out_dir)
        save_best_preview()
        meta = [{"idx": k, "name": names[i], "fill": int(masks[i].sum())}
                for k, i in enumerate(best_ever_combo)]
        (Path(out_dir) / "meta.json").write_text(
            json.dumps({
                "best_score": best_ever_score,
                "patterns": meta,
            }, indent=2, ensure_ascii=False),
            encoding='utf-8')

    while temp > temp_min:
        round_idx += 1
        current_set = set(best_combo)

        # 生成当前组合的所有单步邻居（fixed 槽位跳过），随机打散
        neighbors = []
        combo_list = list(best_combo)
        for pos in range(n):
            if combo_list[pos] in fixed_set:
                continue
            for new_idx in range(N):
                if new_idx in current_set:
                    continue
                nb = list(combo_list)
                nb[pos] = new_idx
                neighbors.append(tuple(sorted(nb)))
        neighbors = list(dict.fromkeys(neighbors))
        if blocked_combo is not None and round_idx <= blocked_until_round:
            neighbors = [nb for nb in neighbors if nb != blocked_combo]
        _rng.shuffle(neighbors)

        print(f"\n  [Round {round_idx}]  T={temp:.5f}  cur={best_score:.4f}  "
              f"best_ever={best_ever_score:.4f}  邻居={len(neighbors)}")
        t0 = time.time()

        # 策略：每批(20个)内选最优，若批内最优满足接受条件则跳出（batch-best first-improvement）
        # 比纯 first-improvement 更稳，比 best-improvement 更快
        accepted_combo = None
        accepted_score = None
        evaluated      = 0
        evaluated_cached = 0
        evaluated_new    = 0

        def try_accept_batch(batch_res):
            """从一批结果里选最优，判断是否接受，返回 (combo, score) 或 None。"""
            if not batch_res:
                return None
            best_nb   = max(batch_res, key=batch_res.get)
            best_nb_s = batch_res[best_nb]
            delta     = best_nb_s - best_score
            if delta > 1e-6:
                return best_nb, best_nb_s
            elif delta < 0 and temp > temp_min * 10:
                effective_temp = max(temp * sa_temp_scale, temp_min)
                if _rng.random() < math.exp(delta / effective_temp):
                    return best_nb, best_nb_s
            return None

        # 已评测的先快速过一遍（免费，每次取 workers*20 个选最优）
        step = workers * 20
        cached = {nb: all_results[nb] for nb in neighbors if nb in all_results}
        if cached:
            cached_list = list(cached.items())
            for bi in range(0, len(cached_list), step):
                batch_res = dict(cached_list[bi:bi+step])
                evaluated += len(batch_res)
                evaluated_cached += len(batch_res)
                result = try_accept_batch(batch_res)
                if result:
                    accepted_combo, accepted_score = result
                    break

        # 若缓存里没找到，继续评测新的
        if accepted_combo is None:
            todo = [c for c in neighbors if c not in all_results]
            for batch_start in range(0, len(todo), step):
                batch = todo[batch_start:batch_start + step]
                if not batch:
                    break
                def on_done(batch_res):
                    all_results.update(batch_res)
                    save_ckpt()
                    save_best_preview()
                new_res = eval_cpp_batch(batch, masks, scales, weights, seeds, exe_path,
                                         batch_size=20, workers=workers,
                                         on_batch_done=on_done,
                                         timeout_sec=eval_timeout)
                all_results.update(new_res)
                evaluated += len(new_res)
                evaluated_new += len(new_res)
                result = try_accept_batch(new_res)
                if result:
                    accepted_combo, accepted_score = result
                    break

        elapsed = time.time() - t0

        if accepted_combo is not None:
            delta = accepted_score - best_score
            tag   = "[OK]" if delta > 1e-6 else "[SA]"
            removed = set(best_combo) - set(accepted_combo)
            added   = set(accepted_combo) - set(best_combo)
            print(f"    {tag} {best_score:.4f} -> {accepted_score:.4f}  "
                  f"(评测 {evaluated} 个, {elapsed:.0f}s)  "
                  f"换出{len(removed)}个 换入{len(added)}个")
            for old_i, new_i in zip(sorted(removed), sorted(added)):
                print(f"      - {names[old_i]}  +  {names[new_i]}")
            prev_combo = best_combo
            best_combo = accepted_combo
            best_score = accepted_score
            blocked_combo = prev_combo
            blocked_until_round = round_idx + backtrack_block_rounds
            if best_score > best_ever_score:
                best_ever_combo = best_combo
                best_ever_score = best_score
                if out_dir:
                    save_patterns([(names[i], masks[i]) for i in best_ever_combo], out_dir)
        else:
            print(f"    [--] 本轮无接受（评测 {evaluated} 个, {elapsed:.0f}s），降温继续")

        temp *= cooling
        refresh_ckpt_state()
        save_ckpt()

    refresh_ckpt_state()
    save_ckpt()
    elapsed_total = time.time() - t_total
    print(f"\n  SA 完成：{round_idx} 轮，总用时 {elapsed_total:.0f}s，"
          f"共评测 {len(all_results)} 组")
    print(f"  最终 best_ever={best_ever_score:.4f}  last_cur={best_score:.4f}")

    # 输出 top-5
    top5 = sorted(all_results.items(), key=lambda x: x[1], reverse=True)[:5]
    from collections import Counter
    print(f"\nTop-5：")
    for rank, (idx, score) in enumerate(top5, 1):
        src = Counter(names[i].split('_')[0] for i in idx)
        src_str = '  '.join(f"{s}x{c}" for s, c in src.most_common())
        print(f"  #{rank}  {score:.4f}  {src_str}")

    best_patterns = [(names[i], masks[i]) for i in best_ever_combo]

    if out_dir:
        save_patterns(best_patterns, out_dir)
        meta = [{"idx": k, "name": names[i], "fill": int(masks[i].sum())}
                for k, i in enumerate(best_ever_combo)]
        (Path(out_dir) / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False))

    return best_patterns, best_ever_score, all_results


def search_best_set_multistart(all_patterns, start_specs, n=16,
                               scales=None, weights=None, seeds=(0, 1, 2),
                               exe_path=None, workers=None,
                               temp_init=0.02, temp_min=1e-4, cooling=0.92,
                               checkpoint_path=None, out_dir=None,
                               eval_timeout=None):
    if not start_specs:
        raise ValueError("start_specs 不能为空")

    global_results = {}
    best_patterns = None
    best_score = -1.0
    ckpt_base = Path(checkpoint_path) if checkpoint_path else None

    for idx, (label, start_combo) in enumerate(start_specs, 1):
        run_ckpt = None
        run_out_dir = None
        if ckpt_base is not None:
            run_ckpt = ckpt_base.with_name(f"{ckpt_base.stem}_{idx:02d}_{label}{ckpt_base.suffix}")
        if out_dir:
            run_out_dir = Path(out_dir) / "_restarts" / f"{idx:02d}_{label}"
        print(f"\n[Restart {idx}/{len(start_specs)}] start={label} first4={list(start_combo)[:4]}")
        selected, score, run_results = search_best_set_ils(
            all_patterns, n=n,
            scales=scales, weights=weights, seeds=seeds,
            exe_path=exe_path, workers=workers,
            start_combo_indices=start_combo,
            bootstrap_results=global_results,
            resume_checkpoint_state=True,
            temp_init=temp_init, temp_min=temp_min, cooling=cooling,
            checkpoint_path=str(run_ckpt) if run_ckpt else None,
            out_dir=str(run_out_dir) if run_out_dir else None,
            eval_timeout=eval_timeout,
        )
        global_results.update(run_results)
        print(f"  [Restart {idx}] score={score:.4f}")
        if score > best_score:
            best_patterns = selected
            best_score = score
            if out_dir:
                save_patterns(best_patterns, out_dir)
                meta = [{"idx": i, "name": nm, "fill": int(mask.sum())}
                        for i, (nm, mask) in enumerate(best_patterns)]
                (Path(out_dir) / "meta.json").write_text(
                    json.dumps({
                        "best_score": best_score,
                        "source_restart": label,
                        "patterns": meta,
                    }, indent=2, ensure_ascii=False),
                    encoding='utf-8')

    return best_patterns, best_score, global_results





# ══════════════════════════════════════════════════════════════════════════════
# 4. 输出
# ══════════════════════════════════════════════════════════════════════════════

def print_pattern_grid(patterns):
    print("\n选出的 pattern：")
    print(f"  {'idx':>3}  {'name':40s}  {'fill':>6}  图案")
    print("  " + "-" * 72)
    for idx, (name, mask) in enumerate(patterns):
        rows = [''.join('█' if mask[r, c] else '.' for c in range(TILE_SIZE))
                for r in range(TILE_SIZE)]
        fill = int(mask.sum())
        print(f"  {idx:3d}  {name:40s}  {fill:3d}/64  {rows[0]}")
        for row in rows[1:]:
            print(f"       {'':40s}         {row}")
        print()


def print_hamming_stats(patterns):
    masks = [p[1] for p in patterns]
    bits_list = [_mask_to_bits(m) for m in masks]
    dists = [_hamming(bits_list[i], bits_list[j])
             for i in range(len(masks)) for j in range(i+1, len(masks))]
    print(f"Hamming 距离: min={min(dists)}  max={max(dists)}  avg={sum(dists)/len(dists):.1f}")
    print(f"  参考 cimbar 16个: min=18, avg=33.4")


def save_patterns(patterns, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (name, mask) in enumerate(patterns):
        tile = ((1 - mask) * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f'{idx:02x}.png'), tile)

    # 预览图
    n = len(patterns)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    sc = 20
    pad = 3
    canvas = np.ones((rows*(TILE_SIZE*sc+pad)+pad, cols*(TILE_SIZE*sc+pad)+pad, 3), np.uint8) * 180
    for idx, (name, mask) in enumerate(patterns):
        r, c = divmod(idx, cols)
        y0 = r * (TILE_SIZE*sc + pad) + pad
        x0 = c * (TILE_SIZE*sc + pad) + pad
        tile_big = cv2.resize(((1-mask)*255).astype(np.uint8),
                              (TILE_SIZE*sc, TILE_SIZE*sc), interpolation=cv2.INTER_NEAREST)
        canvas[y0:y0+TILE_SIZE*sc, x0:x0+TILE_SIZE*sc] = cv2.cvtColor(tile_big, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_dir / 'preview.png'), canvas)
    print(f"已保存 {n} 个 pattern -> {out_dir}/  预览图: preview.png")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def load_seed_patterns(seed_dir, prefix="seedbest"):
    seed_dir = Path(seed_dir)
    if not seed_dir.exists():
        return []

    patterns = []
    for png in sorted(seed_dir.glob('*.png')):
        if png.stem.lower() == 'preview':
            continue
        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (TILE_SIZE, TILE_SIZE):
            continue
        mask = (img < 128).astype(np.uint8)
        patterns.append((f"{prefix}_{png.stem}", mask))
    return patterns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",          type=int, default=16)
    ap.add_argument("--out",        default="scripts/patterns")
    ap.add_argument("--save",       action="store_true")
    ap.add_argument("--workers",    type=int, default=None,  help="进程数，默认 CPU 核心数-1")
    ap.add_argument("--max-font",   type=int, default=200,   help="每种字体最多取多少候选")
    ap.add_argument("--prefilter",  type=int, default=0,  help="Stage1 Hamming 预筛保留数量（0=不预筛）")
    ap.add_argument("--seeds",      type=str, default="0,1,2", help="每组评测使用的随机种子，逗号分隔")
    ap.add_argument("--eval-exe",   default=None,            help="eval_patterns.exe 路径，默认 build/eval_patterns.exe")
    ap.add_argument("--temp-init",  type=float, default=0.02,  help="SA 初始温度（默认0.02）")
    ap.add_argument("--temp-min",   type=float, default=1e-4,  help="SA 终止温度（默认1e-4）")
    ap.add_argument("--cooling",    type=float, default=0.92,  help="SA 降温系数（默认0.92）")
    ap.add_argument("--no-native",  action="store_true",     help="跳过原生8x8字体（font8x8+kbd PSF）")
    ap.add_argument("--no-font",    action="store_true")
    ap.add_argument("--no-cimbar",  action="store_true")
    ap.add_argument("--no-ours",    action="store_true")
    ap.add_argument("--seed-dir",   default="", help="seed pattern dir")
    ap.add_argument("--random-starts", type=int, default=2,
                    help="16-pattern 模式额外加入多少个随机多样性起点（默认 2）")
    ap.add_argument("--rng-seed", type=int, default=12345,
                    help="随机起点构造的随机种子")
    ap.add_argument("--eval-timeout", type=float, default=0.0,
                    help="单次 eval_patterns.exe 调用超时秒数，<=0 表示不设超时")
    args = ap.parse_args()

    # 提前创建输出目录（checkpoint 也写在这里）
    Path(args.out).mkdir(parents=True, exist_ok=True)

    workers = args.workers or DEFAULT_WORKERS
    seeds   = tuple(int(s) for s in args.seeds.split(','))
    exe     = args.eval_exe or _DEFAULT_EVAL_EXE
    eval_timeout = None if args.eval_timeout <= 0 else args.eval_timeout
    print(f"使用 eval_patterns.exe: {exe}")
    print(f"评测档位: {EVAL_SCALES}  权重: {EVAL_WEIGHTS}  种子: {seeds}")
    print(f"eval_timeout: {'none' if eval_timeout is None else eval_timeout}")

    print(f"score_weights: {EVAL_SCORE_WEIGHTS}")
    # ── 构建候选池 ───────────────────────────────────────────────
    all_patterns = []

    seed_patterns = load_seed_patterns(args.seed_dir) if args.seed_dir else []
    if seed_patterns:
        print(f"\n[seed] loaded {len(seed_patterns)} seed patterns from {args.seed_dir}")
        all_patterns += seed_patterns

    if not args.no_ours:
        print("\n[1] 生成当前系统 pattern...")
        p = gen_our_system_patterns()
        all_patterns += p
        print(f"    {len(p)} 个")

    if not args.no_cimbar:
        print("\n[2] 加载 cimbar pattern...")
        p = gen_cimbar_patterns()
        all_patterns += p
        print(f"    {len(p)} 个")

    if not args.no_font:
        print(f"\n[3] 从字体渲染候选（每种最多 {args.max_font} 个）...")
        p = gen_font_patterns(max_per_font=args.max_font, workers=workers)
        all_patterns += p
        print(f"    共 {len(p)} 个")

    if not args.no_native:
        print(f"\n[4] 加载原生 8x8 点阵字体（font8x8 + kbd PSF）...")
        p = gen_native_8x8_patterns()
        all_patterns += p
        print(f"    共 {len(p)} 个")

    # 去重
    seen, deduped = set(), []
    for name, mask in all_patterns:
        h = mask.tobytes()
        if h not in seen:
            seen.add(h)
            deduped.append((name, mask))
    print(f"\n候选池: {len(deduped)} 个（去重后）")

    if len(deduped) < args.n:
        print(f"[错误] 候选不足 {args.n} 个"); return 1

    # ── Stage 1：Hamming 预筛 ────────────────────────────────────
    if args.prefilter > 0 and len(deduped) > args.prefilter:
        print(f"\n[Stage 1] Hamming 距离预筛 {len(deduped)} -> {args.prefilter} 个...")
        deduped = hamming_prefilter(deduped, keep=args.prefilter)
    else:
        print(f"\n[Stage 1] 跳过预筛（候选数 {len(deduped)} ≤ {args.prefilter}）")

    # ── Stage 2：ILS 迭代局部搜索，C++ 批量评测 ──────────────────
    ckpt = Path(args.out) / "checkpoint.json"
    out_dir_rt = args.out if args.save else None

    if args.n > 16:
        # 32-pattern 三阶段：
        #   阶段A：贪心，锁定 cimbar 16个，逐个添加最优自由 pattern 直到32个
        #   阶段B：ILS，解锁全部32个，以阶段A结果为起点，替换掉不优的 cimbar
        seed_idx = [i for i, (nm, _) in enumerate(deduped) if nm.startswith('seedbest_')]
        cimbar_idx = [i for i, (nm, _) in enumerate(deduped) if nm.startswith('cimbar_')]
        if len(cimbar_idx) >= 16:
            fixed_indices = tuple(cimbar_idx[:16])
            fixed_tag = 'cimbar'
        elif len(seed_idx) >= 16:
            fixed_indices = tuple(seed_idx[:16])
            fixed_tag = 'seedbest'
        else:
            fixed_indices = tuple(range(16))
            fixed_tag = 'fallback'

        print(f"\n[Stage 2A] greedy expand: lock {len(fixed_indices)} {fixed_tag}, add {args.n - len(fixed_indices)} free patterns...")
        ckpt_a = Path(args.out) / "checkpoint_phase_a.json"
        selected_a, score_a, _ = search_greedy_expand(
            deduped,
            fixed_indices=fixed_indices,
            target_n=args.n,
            scales=EVAL_SCALES, weights=EVAL_WEIGHTS, seeds=seeds,
            exe_path=exe,
            workers=workers,
            checkpoint_path=str(ckpt_a),
            out_dir=out_dir_rt,
            eval_timeout=eval_timeout,
        )
        print(f"  阶段A得分: {score_a:.4f}")

        print(f"\n[Stage 2B] SA 全量优化：解锁全部32个，继续替换（含 cimbar）...")
        ckpt_b = Path(args.out) / "checkpoint_phase_b.json"
        selected, best_score, _ = search_best_set_ils(
            deduped, n=args.n,
            scales=EVAL_SCALES, weights=EVAL_WEIGHTS, seeds=seeds,
            exe_path=exe,
            fixed_indices=None,
            start_patterns=selected_a,
            temp_init=args.temp_init, temp_min=args.temp_min, cooling=args.cooling,
            checkpoint_path=str(ckpt_b),
            out_dir=out_dir_rt,
            eval_timeout=eval_timeout,
        )
    else:
        # 16-pattern：多起点 SA 搜索
        start_specs = build_start_combos(
            deduped, args.n,
            random_starts=args.random_starts,
            rng_seed=args.rng_seed,
        )
        print(f"\n[Stage 2] SA 搜索最优 {args.n}-pattern 组合...")
        print(f"  多起点: {[label for label, _ in start_specs]}")
        selected, best_score, _ = search_best_set_multistart(
            deduped, start_specs=start_specs, n=args.n,
            scales=EVAL_SCALES, weights=EVAL_WEIGHTS, seeds=seeds,
            exe_path=exe, workers=workers,
            temp_init=args.temp_init, temp_min=args.temp_min, cooling=args.cooling,
            checkpoint_path=str(ckpt),
            out_dir=out_dir_rt,
            eval_timeout=eval_timeout,
        )

    # ── 输出 ────────────────────────────────────────────────────
    print_pattern_grid(selected)
    print_hamming_stats(selected)
    print(f"\n最终得分: {best_score:.4f}  完成！")

    if args.save:
        save_patterns(selected, args.out)
        meta = [{"idx": i, "name": n, "fill": int(m.sum())}
                for i, (n, m) in enumerate(selected)]
        (Path(args.out) / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
