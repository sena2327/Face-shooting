# src_single/render_utils.py
import math
import random

import cv2
import numpy as np


# アルファ合成でサークル描画
def draw_alpha_circle(img, center, radius, color, alpha, thickness=-1):
    """Draw a circle with per-call alpha onto img."""
    if alpha <= 0.0 or radius <= 0:
        return
    overlay = img.copy()
    cv2.circle(overlay, center, int(radius), color, thickness)
    cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0, img)


# アルファ合成で楕円描画（影用）
def draw_alpha_ellipse(img, center, axes, angle_deg, color, alpha, thickness=-1):
    """Draw an ellipse with per-call alpha onto img.
    center: (x, y)
    axes: (axis_x, axis_y)  ※OpenCVのellipseは半径指定
    angle_deg: 回転角（度）
    """
    if alpha <= 0.0 or axes[0] <= 0 or axes[1] <= 0:
        return
    overlay = img.copy()
    cv2.ellipse(
        overlay,
        (int(center[0]), int(center[1])),
        (int(axes[0]), int(axes[1])),
        float(angle_deg),
        0,
        360,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0, img)


# アルファ合成でポリライン描画
def draw_alpha_polyline(img, pts, color, alpha, thickness=1):
    if alpha <= 0.0 or len(pts) < 2:
        return
    overlay = img.copy()
    cv2.polylines(
        overlay,
        [np.array(pts, dtype=np.int32)],
        isClosed=False,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0, img)


# アルファ合成でライン描画（ビーム影用）
def draw_alpha_line(img, p1, p2, color, alpha, thickness=1):
    if alpha <= 0.0:
        return
    overlay = img.copy()
    cv2.line(
        overlay,
        (int(p1[0]), int(p1[1])),
        (int(p2[0]), int(p2[1])),
        color,
        int(max(1, thickness)),
        cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0, img)


# --- RGBA image blending helpers (for parallax backgrounds) ---
def blend_rgba(dst_bgr, src_rgba, xoff=0, yoff=0, alpha_scale=1.0):
    """Alpha-blend an RGBA image onto a BGR canvas at integer offset.
    Supports partial overlaps. Does not tile automatically.
    """
    if src_rgba is None:
        return dst_bgr
    h, w = dst_bgr.shape[:2]
    sh, sw = src_rgba.shape[:2]
    x1 = max(0, xoff)
    y1 = max(0, yoff)
    x2 = min(w, xoff + sw)
    y2 = min(h, yoff + sh)
    if x1 >= x2 or y1 >= y2:
        return dst_bgr
    crop = src_rgba[(y1 - yoff):(y2 - yoff), (x1 - xoff):(x2 - xoff)]
    if crop.shape[2] < 4:
        # No alpha channel; treat as opaque
        dst_bgr[y1:y2, x1:x2] = crop[:, :, :3]
        return dst_bgr
    a = (crop[:, :, 3:4].astype(np.float32) / 255.0) * float(alpha_scale)
    roi = dst_bgr[y1:y2, x1:x2].astype(np.float32)
    dst_bgr[y1:y2, x1:x2] = (
        roi * (1 - a) + crop[:, :, :3].astype(np.float32) * a
    ).astype(np.uint8)
    return dst_bgr


def blend_rgba_tiled_x(dst_bgr, src_rgba, xoff, yoff=0, alpha_scale=1.0):
    """Horizontally tile an RGBA layer once so scrolling wraps.
    Draws at xoff and xoff - sw to cover wrap-around.
    """
    if src_rgba is None:
        return dst_bgr
    sh, sw = src_rgba.shape[:2]
    xoff_mod = int(xoff) % sw
    dst_bgr = blend_rgba(dst_bgr, src_rgba, xoff=xoff_mod, yoff=yoff, alpha_scale=alpha_scale)
    dst_bgr = blend_rgba(dst_bgr, src_rgba, xoff=xoff_mod - sw, yoff=yoff, alpha_scale=alpha_scale)
    return dst_bgr


# ひび割れパターン生成
def make_crack(x, y, base_r=40, branches=8, seg_min=12, seg_max=28):
    """Create a crack decal composed of multiple jagged polylines.
    Returns a dict with geometry and lifetime for later rendering.
    """
    lines = []
    cx, cy = float(x), float(y)
    for _ in range(branches):
        ang = random.uniform(0, 2 * math.pi)
        nseg = random.randint(3, 6)
        pts = [(cx, cy)]
        seg_len = random.uniform(seg_min, seg_max)
        dirx, diry = math.cos(ang), math.sin(ang)
        px, py = cx, cy
        for _ in range(nseg):
            # 進む＋ジッタでギザギザ
            jitter = random.uniform(-0.6, 0.6)
            jx = -diry * jitter * 6.0
            jy = dirx * jitter * 6.0
            px += dirx * seg_len + jx
            py += diry * seg_len + jy
            pts.append((int(px), int(py)))
            # 方向を少し曲げる
            ang += random.uniform(-0.35, 0.35)
            dirx, diry = math.cos(ang), math.sin(ang)
            seg_len *= random.uniform(0.85, 1.10)
        lines.append(pts)
    return {
        "x": cx,
        "y": cy,
        "lines": lines,
        "life": 1.2,      # 表示寿命（秒）
        "max_life": 1.2,
    }


# --- 六角グリッドATフィールド風オーバーレイ生成 ---
def build_hex_overlay(w, h, cell=42, line_th=1, color=(0, 165, 255)):
    """Return a BGR image (uint8) with a hexagonal grid drawn on black.
    cell: hex circumradius (pixel). color is BGR.
    """
    img = np.zeros((h, w, 3), np.uint8)
    # hex geometry
    R = float(cell)              # circumradius
    r = R * math.sqrt(3) / 2.0   # inradius (apothem)
    dx = 1.5 * R                 # horizontal center spacing
    dy = r                       # vertical step per row
    # Build centers and draw hexagons
    y = 0.0
    row = 0
    while y < h + R:
        x0 = R if (row % 2 == 0) else (R + 0.75 * R)
        x = x0
        while x < w + R:
            # 6 vertices around (x, y)
            pts = []
            for k in range(6):
                ang = math.radians(60 * k)
                px = int(round(x + R * math.cos(ang)))
                py = int(round(y + R * math.sin(ang)))
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                img,
                [pts],
                isClosed=True,
                color=color,
                thickness=line_th,
                lineType=cv2.LINE_AA,
            )
            x += dx
        y += dy
        row += 1
    return img