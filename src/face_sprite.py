"""
Utilities for capturing and compositing opponent face sprites.
"""

from __future__ import annotations

import os
from typing import Optional
import csv

import cv2
import mediapipe as mp
import numpy as np


def _ensure_rgba(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Convert grayscale/BGR images to BGRA (opaque alpha)."""
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.ndim == 3 and img.shape[2] == 3:
        h, w = img.shape[:2]
        a = np.full((h, w, 1), 255, dtype=np.uint8)
        img = np.concatenate([img, a], axis=2)
    return img


def _load_three_point_csv(csv_path: str) -> Optional[np.ndarray]:
    """Load 3 anchor points (left eye, right eye, chin) from a CSV."""
    pts = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                coords = []
                for token in row[:2]:
                    token = token.strip()
                    if not token:
                        continue
                    try:
                        coords.append(float(token))
                    except ValueError:
                        coords = []
                        break
                if len(coords) == 2:
                    pts.append((coords[0], coords[1]))
                if len(pts) >= 3:
                    break
    except FileNotFoundError:
        print(f"[BOSS_FACE] anchor csv not found: {csv_path}")
        return None
    except Exception as exc:
        print(f"[BOSS_FACE] anchor csv read failed ({csv_path}): {exc}")
        return None

    if len(pts) < 3:
        print(f"[BOSS_FACE] anchor csv does not have 3 rows: {csv_path}")
        return None
    return np.asarray(pts[:3], dtype=np.float32)


def composite_face_with_anchors(user_face_bgr: Optional[np.ndarray],
                                user_pts: Optional[np.ndarray],
                                base_image_path: str,
                                anchor_csv_path: str) -> Optional[np.ndarray]:
    """Composite a rectangular BGR face (方式B) onto a base image using 3 anchor points."""
    if user_face_bgr is None or user_pts is None:
        return None
    base = cv2.imread(base_image_path, cv2.IMREAD_UNCHANGED)
    if base is None:
        print(f"[BOSS_FACE] base image not found: {base_image_path}")
        return None
    dst_pts = _load_three_point_csv(anchor_csv_path)
    if dst_pts is None:
        return None

    user_pts = np.asarray(user_pts, dtype=np.float32)
    if user_pts.shape != (3, 2):
        print(f"[BOSS_FACE] user_pts must be 3x2, got {user_pts.shape}")
        return None

    base_rgba = _ensure_rgba(base)
    face_rgba = _ensure_rgba(user_face_bgr)
    if base_rgba is None or face_rgba is None:
        return None

    h_base, w_base = base_rgba.shape[:2]
    try:
        M = cv2.getAffineTransform(user_pts, dst_pts)
        warped = cv2.warpAffine(
            face_rgba,
            M,
            (w_base, h_base),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
    except cv2.error as exc:
        print(f"[BOSS_FACE] affine warp failed: {exc}")
        return None

    face = warped.astype(np.float32)
    base_f = base_rgba.astype(np.float32)
    face_a = face[:, :, 3:4] / 255.0
    base_a = base_f[:, :, 3:4] / 255.0

    out_rgb = face[:, :, :3] * face_a + base_f[:, :, :3] * (1.0 - face_a)
    out_a = np.clip(face_a + base_a * (1.0 - face_a), 0.0, 1.0)

    out = np.empty_like(base_rgba, dtype=np.uint8)
    out[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    out[:, :, 3] = (out_a[..., 0] * 255.0).astype(np.uint8)
    return out

def composite_face_into_mask(face_rgba, mask_path="img/cap.png"):
    """Composite a face RGBA sprite into the helmet mask using CSV for alignment if available, else fallback to black-region fit."""
    if face_rgba is None:
        return None
    base = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if base is None:
        print(f"[P2P_FACE] mask image not found: {mask_path}")
        return face_rgba
    h_base, w_base = base.shape[:2]
    # Ensure base has 4 channels (BGRA). If it already has 4, keep as is.
    if base.ndim == 2:
        # Grayscale → BGR
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        h_base, w_base = base.shape[:2]
    if base.ndim == 3 and base.shape[2] == 3:
        a = np.full((h_base, w_base, 1), 255, dtype=np.uint8)
        base = np.concatenate([base, a], axis=2)
    # --- Try to find a 3-point CSV file for alignment ---
    mask_basename = os.path.splitext(os.path.basename(mask_path))[0]
    csv_candidates = [
        f"./{mask_basename}.csv",
        os.path.join(os.path.dirname(mask_path), mask_basename + ".csv")
    ]
    csv_path = None
    for c in csv_candidates:
        if os.path.isfile(c):
            csv_path = c
            break
    points_ok = False
    if csv_path is not None:
        try:
            import csv
            pts = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if len(row) >= 3:
                        idx, x, y = row[:3]
                        pts.append((float(x), float(y)))
                    if len(pts) >= 3:
                        break
            if len(pts) >= 3:
                # CSV: 0: left eye, 1: right eye, 2: chin
                left_eye_dst, right_eye_dst, chin_dst = pts[:3]
                fh, fw = face_rgba.shape[:2]
                left_eye_src  = (0.30 * fw, 0.40 * fh)
                right_eye_src = (0.70 * fw, 0.40 * fh)
                chin_src      = (0.50 * fw, 0.85 * fh)
                src = np.array([left_eye_src, right_eye_src, chin_src], dtype=np.float32)
                dst = np.array([left_eye_dst, right_eye_dst, chin_dst], dtype=np.float32)
                # Affine transform and warp
                M = cv2.getAffineTransform(src, dst)
                face_warped = cv2.warpAffine(
                    face_rgba, M, (w_base, h_base),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
                )
                # Alpha-composite BASE (cap/helmet) OVER the face_warped, so that the cap is on the top layer.
                # Treat face_warped as the "background" and base as the "foreground".
                # Prepare alpha channels
                face_a = face_warped[:, :, 3:4].astype(np.float32) / 255.0 if face_warped.shape[2] == 4 else np.ones((h_base, w_base, 1), dtype=np.float32)
                base_a = base[:, :, 3:4].astype(np.float32) / 255.0 if base.shape[2] == 4 else np.ones((h_base, w_base, 1), dtype=np.float32)

                face_rgb = face_warped[:, :, :3].astype(np.float32)
                base_rgb = base[:, :, :3].astype(np.float32)

                # Foreground (base) over background (face)
                # out_a = base_a + face_a * (1 - base_a)
                out_a_new = base_a + face_a * (1.0 - base_a)
                eps = 1e-6

                # out_rgb * out_a = base_rgb*base_a + face_rgb*face_a*(1-base_a)
                out_rgb_new = base_rgb * base_a + face_rgb * face_a * (1.0 - base_a)

                # Avoid division by zero
                out_rgb_final = np.zeros_like(base_rgb)
                valid = out_a_new > eps
                out_rgb_final[valid[:, :, 0]] = (out_rgb_new[valid[:, :, 0]] / out_a_new[valid[:, :, 0]])

                out = base.copy()
                out[:, :, :3] = np.clip(out_rgb_final, 0, 255).astype(np.uint8)
                out[:, :, 3:4] = np.clip(out_a_new * 255.0, 0, 255).astype(np.uint8)
                points_ok = True
                return out
        except Exception as e:
            print(f"[P2P_FACE] CSV affine failed ({csv_path}): {e}")
    # --- Fallback: original black-region logic ---
    bgr = base[:, :, :3]
    black_thresh = 10
    mask = (
        (bgr[:, :, 0] < black_thresh) &
        (bgr[:, :, 1] < black_thresh) &
        (bgr[:, :, 2] < black_thresh)
    )
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        print("[P2P_FACE] no black region found in mask; skipping composite.")
        return face_rgba
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    hole_w = max(1, x1 - x0)
    hole_h = max(1, y1 - y0)
    fh, fw = face_rgba.shape[:2]
    if fh <= 0 or fw <= 0:
        return face_rgba
    scale = min(hole_w / float(fw), hole_h / float(fh))
    scale = max(scale, 1e-3)
    new_w = max(1, int(fw * scale))
    new_h = max(1, int(fh * scale))
    face_resized = cv2.resize(face_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_face = x0 + (hole_w - new_w) // 2
    y_face = y0 + (hole_h - new_h) // 2
    out = base.copy()
    y0_clip = max(0, y_face)
    x0_clip = max(0, x_face)
    y1_clip = min(h_base, y_face + new_h)
    x1_clip = min(w_base, x_face + new_w)
    if y0_clip >= y1_clip or x0_clip >= x1_clip:
        return out
    roi_out = out[y0_clip:y1_clip, x0_clip:x1_clip]
    roi_face = face_resized[(y0_clip - y_face):(y1_clip - y_face),
                            (x0_clip - x_face):(x1_clip - x_face)]
    mask_crop = mask[y0_clip:y1_clip, x0_clip:x1_clip]
    face_rgb = roi_face[:, :, :3].astype(np.float32)
    if roi_face.shape[2] == 4:
        face_a = roi_face[:, :, 3:4].astype(np.float32) / 255.0
    else:
        face_a = np.ones((roi_face.shape[0], roi_face.shape[1], 1), dtype=np.float32)
    out_rgb = roi_out[:, :, :3].astype(np.float32)
    if roi_out.shape[2] == 4:
        out_a = roi_out[:, :, 3:4].astype(np.float32) / 255.0
    else:
        out_a = np.ones((roi_out.shape[0], roi_out.shape[1], 1), dtype=np.float32)
    m = mask_crop.astype(np.float32)[..., None]
    face_a_eff = face_a * m
    out_a_new = face_a_eff + out_a * (1.0 - face_a_eff)
    eps = 1e-6
    out_rgb_new = (face_rgb * face_a_eff + out_rgb * out_a * (1.0 - face_a_eff))
    valid = out_a_new > eps
    out_rgb_final = out_rgb.copy()
    out_rgb_final[valid[:, :, 0]] = (out_rgb_new[valid[:, :, 0]] / out_a_new[valid[:, :, 0]])
    roi_out[:, :, :3] = np.clip(out_rgb_final, 0, 255).astype(np.uint8)
    roi_out[:, :, 3:4] = np.clip(out_a_new * 255.0, 0, 255).astype(np.uint8)
    out[y0_clip:y1_clip, x0_clip:x1_clip] = roi_out
    return out

def capture_opponent_frame(cam_index=0, width=320, height=240,
                           mask_path="img/blue_cap.png"):
    """Capture a single frame from the default camera and return it as an RGBA BGRA sprite.

    1. カメラから1フレーム取得
    2. MediaPipe FaceMesh で顔の輪郭（フェイスオーバル）を検出
    3. 顔部分を切り抜いて正方形 RGBA スプライトを作成
    4. composite_face_into_mask() を使って mask_path（例: blue_cap.png）に合成

    顔検出に失敗した場合は、従来通りのセンタークロップにフォールバックする。
    Returns:
        np.ndarray of shape (H, W, 4) in BGRA, or None on failure.
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[OPP] Failed to open camera index {cam_index} for opponent capture.")
        return None

    # Try to set a reasonable resolution (may be ignored by some drivers)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Warm-up: read several frames so that exposure/white balance stabilizes
    warmup_N = 50
    ok = False
    frame = None
    for _ in range(warmup_N):
        ok, f = cap.read()
        if not ok or f is None:
            break
        frame = f
    cap.release()

    if not ok or frame is None:
        print("[OPP] Failed to capture opponent frame from camera (warm-up stage).")
        return None

    h, w = frame.shape[:2]

    # --- Face extraction using MediaPipe FaceMesh landmarks (face oval) ---
    face_crop = None
    mask_crop = None
    try:
        mp_face_mesh = mp.solutions.face_mesh
        # Commonly used FaceMesh indices for the face oval contour.
        face_oval_idx = [
            10, 338, 297, 332, 284, 251, 389, 356,
            454, 323, 361, 288, 397, 365, 379, 378,
            400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21,
            54, 103, 67, 109
        ]
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as mesh:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0]
                pts = []
                for idx in face_oval_idx:
                    if idx < len(lm.landmark):
                        pt = lm.landmark[idx]
                        x = int(pt.x * w)
                        y = int(pt.y * h)
                        pts.append([x, y])
                if len(pts) >= 3:
                    pts_np = np.array(pts, dtype=np.int32)
                    # Build full-size mask (face oval filled)
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(full_mask, [pts_np], 255)
                    # Bounding box around the oval for a tighter sprite
                    xs = pts_np[:, 0]
                    ys = pts_np[:, 1]
                    x0 = max(0, int(xs.min()))
                    y0 = max(0, int(ys.min()))
                    x1 = min(w, int(xs.max()))
                    y1 = min(h, int(ys.max()))
                    if x1 > x0 and y1 > y0:
                        face_crop = frame[y0:y1, x0:x1].copy()
                        mask_crop = full_mask[y0:y1, x0:x1].copy()
    except Exception as e:
        print(f"[OPP] FaceMesh face crop failed: {e}")
    # Fallback: if FaceMesh failed, use a simple center crop with full mask
    if face_crop is None or mask_crop is None:
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        face_crop = frame[y0:y0+side, x0:x0+side, :].copy()
        mask_crop = np.full((side, side), 255, dtype=np.uint8)

    # --- Pack face_crop and its mask into a square RGBA sprite ---
    fh, fw = face_crop.shape[:2]
    side_out = max(fh, fw)
    canvas = np.zeros((side_out, side_out, 3), dtype=np.uint8)
    mask_canvas = np.zeros((side_out, side_out), dtype=np.uint8)
    y_off = (side_out - fh) // 2
    x_off = (side_out - fw) // 2
    canvas[y_off:y_off+fh, x_off:x_off+fw, :] = face_crop
    mask_canvas[y_off:y_off+fh, x_off:x_off+fw] = mask_crop
    # Slight blur to soften edges
    mask_canvas = cv2.GaussianBlur(mask_canvas, (11, 11), 5)
    alpha = mask_canvas[..., None]
    rgba = np.concatenate([canvas, alpha], axis=2)

    # ここで blue_cap.csv に合わせて帽子を被せた画像にしてしまう
    try:
        rgba = composite_face_into_mask(rgba, mask_path=mask_path)
    except Exception as e:
        print(f"[OPP] composite into {mask_path} failed (ignored): {e}")

    return rgba
