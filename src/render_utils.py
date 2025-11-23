"""
Rendering helper utilities for alpha-blended shapes and RGBA overlays.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import math

import cv2
import numpy as np

# アルファ合成でサークル描画
def draw_alpha_circle(
    img: np.ndarray,
    center: Tuple[int, int],
    radius: float,
    color: Tuple[int, int, int],
    alpha: float,
    thickness: int = -1,
) -> None:
    """Draw a circle with per-call alpha onto img."""
    if alpha <= 0.0 or radius <= 0:
        return
    overlay = img.copy()
    cv2.circle(overlay, center, int(radius), color, thickness)
    cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0, img)

# アルファ合成で楕円描画（影用）
def draw_alpha_ellipse(
    img: np.ndarray,
    center: Tuple[float, float],
    axes: Tuple[float, float],
    angle_deg: float,
    color: Tuple[int, int, int],
    alpha: float,
    thickness: int = -1,
) -> None:
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
def draw_alpha_polyline(
    img: np.ndarray,
    pts: Sequence[Tuple[float, float]],
    color: Tuple[int, int, int],
    alpha: float,
    thickness: int = 1,
) -> None:
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


# --- RGBA image blending helpers (for parallax backgrounds) ---
def blend_rgba(
    dst_bgr: np.ndarray,
    src_rgba: np.ndarray | None,
    xoff: int = 0,
    yoff: int = 0,
    alpha_scale: float = 1.0,
) -> np.ndarray:
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

def blend_rgba_tiled_x(
    dst_bgr: np.ndarray,
    src_rgba: np.ndarray | None,
    xoff: float,
    yoff: int = 0,
    alpha_scale: float = 1.0,
) -> np.ndarray:
    """Horizontally tile an RGBA layer once so scrolling wraps.
    Draws at xoff and xoff - sw to cover wrap-around.
    """
    if src_rgba is None:
        return dst_bgr
    sh, sw = src_rgba.shape[:2]
    xoff_mod = int(xoff) % sw
    dst_bgr = blend_rgba(
        dst_bgr, src_rgba, xoff=xoff_mod, yoff=yoff, alpha_scale=alpha_scale
    )
    dst_bgr = blend_rgba(
        dst_bgr, src_rgba, xoff=xoff_mod - sw, yoff=yoff, alpha_scale=alpha_scale
    )
    return dst_bgr

# --- 六角グリッドATフィールド風オーバーレイ生成 ---
def build_hex_overlay(
    w: int,
    h: int,
    cell: int = 42,
    line_th: int = 1,
    color: Tuple[int, int, int] = (0, 165, 255),
) -> np.ndarray:
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

def world_to_screen(
    wx: float,
    wy: float,
    cam_x: float,
    cam_y: float,
) -> Tuple[int, int]:
    """
    ワールド座標 (wx, wy) を画面座標 (sx, sy) に変換する。
    cam_x, cam_y は「画面の左上」がワールド上のどこかを表す。
    """
    sx = int(wx - cam_x)
    sy = int(wy - cam_y)
    return sx, sy


def screen_to_world(
    sx: float,
    sy: float,
    cam_x: float,
    cam_y: float,
) -> Tuple[float, float]:
    """
    画面座標 (sx, sy) をワールド座標 (wx, wy) に変換する。
    """
    wx = float(sx) + float(cam_x)
    wy = float(sy) + float(cam_y)
    return wx, wy


def project_to_screen(
    wx: float,
    wy: float,
    z: float,
    cam_x: float,
    cam_y: float,
    scr_w: int,
    scr_h: int,
    depth_strength: float = 0.7,
) -> Tuple[int, int, float]:
    """
    擬似3D用の投影。
    z が大きいほど奥にある＝小さく＆画面中心へ寄せる。
    返り値: (sx, sy, scale)
    """
    depth = 1.0 / (1.0 + float(z) * depth_strength)

    # カメラ変換
    sx = (wx - cam_x)
    sy = (wy - cam_y)

    # パースをかけて中心へ寄せる
    sx = sx * depth + scr_w * 0.5 * (1.0 - depth)
    sy = sy * depth + scr_h * 0.5 * (1.0 - depth)

    return int(sx), int(sy), depth


def dist_point_to_segment(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """
    距離: 点 (px,py) と セグメント (x1,y1)-(x2,y2) の最短距離を返す。
    """
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    vv = vx * vx + vy * vy
    if vv <= 1e-9:
        dx, dy = px - x1, py - y1
        return math.hypot(dx, dy)
    t = (wx * vx + wy * vy) / vv
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    projx = x1 + t * vx
    projy = y1 + t * vy
    return math.hypot(px - projx, py - projy)