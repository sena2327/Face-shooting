# src/bullet_render.py
from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np

from .render_utils import world_to_screen, draw_alpha_circle


def get_bullet_aura_params(level: int, *, remote: bool = False) -> Tuple[Tuple[int, int, int], float, float]:
    """Return (color, radius_scale, alpha) based on bullet level."""
    level = int(np.clip(level, 1, 5))
    table = {
        1: ((80, 80, 160), 1.15, 0.04),
        2: ((120, 160, 220), 1.35, 0.10),
        3: ((80, 200, 255), 1.65, 0.20),
        4: ((40, 120, 255), 1.95, 0.30),
        5: ((20, 40, 255), 2.35, 0.40),
    }
    color, radius_scale, alpha = table[level]
    if remote:
        alpha = min(0.60, alpha * 1.25)
    return color, radius_scale, alpha


def draw_player_bullet(
    img,
    bullet,
    cam_x: float,
    cam_y: float,
    scr_w: int,
    scr_h: int,
) -> None:
    """
    プレイヤー弾を青い円として描画する。
    発射地点から離れるほど小さく・暗くなる。
    """
    # ワールド座標 → 画面座標
    sx, sy = world_to_screen(bullet.x, bullet.y, cam_x, cam_y)

    # 画面外は描画しない（少しマージンを持たせる）
    if sx < -50 or sx > scr_w + 50 or sy < -50 or sy > scr_h + 50:
        return

    level = int(np.clip(getattr(bullet, "level", 1), 1, 5))

    # 進行度 t に応じて明るさを変化させる
    t = float(np.clip(getattr(bullet, "t", 0.0), 0.0, 1.0))
    max_brightness = 1.0
    min_brightness = 0.3  # 完全に真っ暗にはしない
    brightness = max_brightness * (1.0 - t) + min_brightness * t

    # ベース色は青 (BGR)
    base_color = (255, 0, 0)
    color = tuple(int(c * brightness) for c in base_color)

    base_radius = int(max(1.0, bullet.r))
    aura_color, aura_scale, aura_alpha = get_bullet_aura_params(level, remote=False)
    draw_alpha_circle(
        img,
        (int(sx), int(sy)),
        int(base_radius * aura_scale),
        aura_color,
        aura_alpha,
        thickness=-1,
    )

    cv2.circle(
        img,
        (int(sx), int(sy)),
        base_radius,
        color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    # 外側の縁取り（少し大きい半径で明るい色を描画）
    outline_radius = int(base_radius + 2)
    outline_color = (255, 255, 255)  # 白いアウトライン
    cv2.circle(
        img,
        (int(sx), int(sy)),
        outline_radius,
        outline_color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def draw_remote_player_bullet(
    img,
    bullet,
    cam_x: float,
    cam_y: float,
    scr_w: int,
    scr_h: int,
) -> None:
    """
    相手プレイヤー弾を黄色い円として描画する。
    RemotePlayerBullet.r をそのまま半径として使う。
    """
    # ワールド座標 → 画面座標
    sx, sy = world_to_screen(bullet.x, bullet.y, cam_x, cam_y)

    # 画面外は描画しない（少しマージンを持たせる）
    if sx < -50 or sx > scr_w + 50 or sy < -50 or sy > scr_h + 50:
        return

    level = int(np.clip(getattr(bullet, "level", 1), 1, 5))

    # ベース色は黄色 (BGR)
    base_color = (0, 255, 255)

    base_radius = int(max(1.0, bullet.r))
    aura_color, aura_scale, aura_alpha = get_bullet_aura_params(level, remote=True)
    draw_alpha_circle(
        img,
        (int(sx), int(sy)),
        int(base_radius * aura_scale),
        aura_color,
        aura_alpha,
        thickness=-1,
    )

    cv2.circle(
        img,
        (int(sx), int(sy)),
        base_radius,
        base_color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    # 外側の縁取り（少し大きい半径で明るい色を描画）
    outline_radius = int(base_radius + 2)
    outline_color = (255, 255, 255)  # 白いアウトライン
    cv2.circle(
        img,
        (int(sx), int(sy)),
        outline_radius,
        outline_color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )
