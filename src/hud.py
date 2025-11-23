# src/hud.py
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def draw_text(
    img,
    text: str,
    org: Tuple[int, int],
    size: float = 0.7,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_hp_bar(
    img,
    x: int,
    y: int,
    w: int,
    h: int,
    hp: float,
    hp_max: float,
    label: str = "HP",
    color_fg: Tuple[int, int, int] = (60, 220, 60),
    color_bg: Tuple[int, int, int] = (50, 50, 50),
) -> None:
    """左上に表示するシンプルなHPバー."""
    hp = max(0.0, min(hp, hp_max))
    ratio = hp / hp_max if hp_max > 0 else 0.0
    # 背景
    cv2.rectangle(img, (x, y), (x + w, y + h), color_bg, thickness=-1)
    # 残量バー
    fill_w = int(w * ratio)
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color_fg, thickness=-1)
    # 枠線
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), thickness=1)
    # テキスト
    draw_text(img, f"{label}: {int(hp)}/{int(hp_max)}", (x + 4, y - 6), size=0.5)


def draw_shield_bar(
    img,
    x: int,
    y: int,
    w: int,
    h: int,
    shield_hp: float,
    shield_hp_max: float,
    shield_on: bool,
) -> None:
    """シールド用バー。ON/OFF で色を変える."""
    shield_hp = max(0.0, min(shield_hp, shield_hp_max))
    ratio = shield_hp / shield_hp_max if shield_hp_max > 0 else 0.0
    color_bg = (40, 40, 80)
    color_fg_on = (60, 200, 255)
    color_fg_off = (80, 80, 80)
    color_fg = color_fg_on if shield_on else color_fg_off

    cv2.rectangle(img, (x, y), (x + w, y + h), color_bg, thickness=-1)
    fill_w = int(w * ratio)
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color_fg, thickness=-1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), thickness=1)
    draw_text(
        img,
        f"SHIELD: {int(shield_hp)}/{int(shield_hp_max)}",
        (x + 4, y - 6),
        size=0.5,
        color=(200, 220, 255),
    )


def draw_score(
    img,
    score: int,
    x: int,
    y: int,
) -> None:
    draw_text(img, f"SCORE: {score}", (x, y), size=0.7, color=(255, 255, 180))


def draw_item_slots(
    img,
    x: int,
    y: int,
    size: int,
    items: List[int],
    selected_index: int,
    sprites: dict[int, np.ndarray],
) -> None:
    """
    アイテムスロット UI の描画。
    items: 各スロットのアイテム種別ID（例: 0=HP,1=攻撃UP,..., -1=空）
    sprites: kind -> 画像
    """
    margin = 8
    for i, kind in enumerate(items):
        slot_x = x + i * (size + margin)
        slot_y = y

        # スロット枠
        cv2.rectangle(
            img,
            (slot_x, slot_y),
            (slot_x + size, slot_y + size),
            (200, 200, 200),
            thickness=2,
        )

        # 選択中のスロットは太枠にする
        if i == selected_index:
            cv2.rectangle(
                img,
                (slot_x - 2, slot_y - 2),
                (slot_x + size + 2, slot_y + size + 2),
                (255, 255, 0),
                thickness=2,
            )

        # アイテム画像
        if kind in sprites:
            spr = sprites[kind]
            if spr is not None and spr.size > 0:
                sh, sw = spr.shape[:2]
                scale = min(size / sw, size / sh)
                w2 = int(sw * scale)
                h2 = int(sh * scale)
                resized = cv2.resize(spr, (w2, h2), interpolation=cv2.INTER_AREA)
                ox = slot_x + (size - w2) // 2
                oy = slot_y + (size - h2) // 2
                roi = img[oy : oy + h2, ox : ox + w2]
                if resized.shape[2] == 4:
                    # RGBAの場合はアルファブレンド
                    alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
                    bg = roi.astype(np.float32)
                    fg = resized[:, :, :3].astype(np.float32)
                    img[oy : oy + h2, ox : ox + w2] = (
                        bg * (1 - alpha) + fg * alpha
                    ).astype(np.uint8)
                else:
                    img[oy : oy + h2, ox : ox + w2] = resized[:, :, :3]