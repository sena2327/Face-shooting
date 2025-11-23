# face_overlay_util.py
import cv2
import numpy as np
import os
import mediapipe as mp

BLACK_THRESH = 10  # blue_cap の黒い部分を穴にするしきい値

def load_base_image(path):
    """
    blue_cap.png を RGBA で読み込み、黒い部分を alpha=0 にして返す
    """
    base = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if base is None:
        print(f"[ERR] base image not found: {path}")
        return None

    h, w = base.shape[:2]

    # RGBA でない場合は α=255 を付ける
    if base.ndim == 3 or base.shape[2] == 3:
        a = np.full((h, w, 1), 255, dtype=np.uint8)
        base = np.concatenate([base, a], axis=2)

    # 黒っぽい領域を穴として alpha=0 にする
    bgr = base[:, :, :3]
    mask_black = (
        (bgr[:, :, 0] < BLACK_THRESH) &
        (bgr[:, :, 1] < BLACK_THRESH) &
        (bgr[:, :, 2] < BLACK_THRESH)
    )
    alpha = base[:, :, 3]
    alpha[mask_black] = 0
    base[:, :, 3] = alpha
    return base


def load_anchor_points(csv_path):
    """
    blue_cap 上でクリックして保存した 3 点 (両目 + 顎) を読み込む。
    pandas.to_csv(index=True) な CSV でも OK なようにしてある。
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] anchor csv not found: {csv_path}")
        return None

    try:
        data = np.genfromtxt(csv_path, delimiter=",", dtype=np.float32, skip_header=1)
    except Exception as e:
        print(f"[WARN] failed to load anchor csv '{csv_path}': {e}")
        return None

    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] >= 3:
        data = data[:, -2:]  # 最後の2列を (x,y) とみなす

    if data.shape[0] < 3 or data.shape[1] < 2:
        print(f"[WARN] anchor csv has invalid shape after cleanup: {data.shape}")
        return None

    pts = data[:3, :2].astype(np.float32)
    print(f"[INFO] loaded anchor points from {csv_path}:\n{pts}")
    return pts


def extract_face_keypoints_from_landmarks(landmarks, img_w, img_h):
    """
    MediaPipe FaceMesh の landmarks から「左目・右目・顎」の3点を返す。
    landmarks: face_mesh.process() で得た multi_face_landmarks[0]
    """
    idx_left_eye = 33
    idx_right_eye = 263
    idx_chin = 152

    def lm_to_xy(index):
        if index >= len(landmarks.landmark):
            return None
        p = landmarks.landmark[index]
        x = float(p.x * img_w)
        y = float(p.y * img_h)
        return [x, y]

    p_left = lm_to_xy(idx_left_eye)
    p_right = lm_to_xy(idx_right_eye)
    p_chin = lm_to_xy(idx_chin)

    if p_left is None or p_right is None or p_chin is None:
        return None

    pts = np.array([p_left, p_right, p_chin], dtype=np.float32)
    return pts


def composite_face_to_base(base_rgba, frame_bgr, src_pts, dst_pts):
    """
    3点対応で、カメラ画像中の顔の3点 (src_pts) を
    blue_cap 上の3点 (dst_pts) にアフィン変換で一致させる。

    戻り値: RGBA (base_rgba と同じサイズ)
    """
    if base_rgba is None or frame_bgr is None:
        return None

    h_base, w_base = base_rgba.shape[:2]

    src_pts = np.asarray(src_pts, dtype=np.float32)
    dst_pts = np.asarray(dst_pts, dtype=np.float32)

    if src_pts.shape != (3, 2) or dst_pts.shape != (3, 2):
        print(f"[WARN] invalid src/dst pts shape: src={src_pts.shape}, dst={dst_pts.shape}")
        return base_rgba.copy()

    h_src, w_src = frame_bgr.shape[:2]
    a = np.full((h_src, w_src, 1), 255, dtype=np.uint8)
    frame_rgba = np.concatenate([frame_bgr, a], axis=2)

    M = cv2.getAffineTransform(src_pts, dst_pts)
    warped = cv2.warpAffine(
        frame_rgba,
        M,
        (w_base, h_base),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    bg = warped.astype(np.float32)
    fg = base_rgba.astype(np.float32)

    alpha_fg = fg[:, :, 3:4] / 255.0
    alpha_bg = bg[:, :, 3:4] / 255.0

    out_rgb = fg[:, :, :3] * alpha_fg + bg[:, :, :3] * (1.0 - alpha_fg)
    out_a = alpha_fg + alpha_bg * (1.0 - alpha_fg)

    out = np.zeros_like(base_rgba, dtype=np.uint8)
    out[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(out_a[..., 0] * 255.0, 0, 255).astype(np.uint8)
    return out


def annotate_anchor_points(image_path, out_csv):
    """
    Utility to click 3 anchor points (left eye, right eye, chin) on an arbitrary helmet/base image.
    The points are saved as CSV with columns x,y (same order as clicked).
    Controls:
      - Left click: add point (max 3)
      - 'r': reset points
      - 's' or Enter: save when 3 points selected
      - 'q' or Esc: quit without saving
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERR] annotate_anchor_points: failed to load {image_path}")
        return False

    display = img.copy()
    points = []
    win = "Anchor Annotation (click: L-eye, R-eye, Chin)"

    def refresh_display():
        display[:] = img
        for idx, (px, py) in enumerate(points):
            cv2.circle(display, (px, py), 5, (0, 255, 0), -1, cv2.LINE_AA)
            label = ["L", "R", "C"][idx] if idx < 3 else str(idx + 1)
            cv2.putText(display, label, (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display, "Click L-eye, R-eye, Chin  |  Enter/S: Save  |  R: Reset  |  Q/Esc: Quit",
                    (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 3:
                points.append((x, y))
                refresh_display()

    refresh_display()
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    saved = False
    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(10) & 0xFF
        if key in (ord('q'), 27):
            print("[INFO] annotation cancelled")
            break
        if key == ord('r'):
            points.clear()
            refresh_display()
        if key in (ord('s'), 13):
            if len(points) == 3:
                pts = np.array(points, dtype=np.float32)
                header = "x,y"
                np.savetxt(out_csv, pts, delimiter=",", header=header, comments="")
                print(f"[INFO] saved anchor points to {out_csv}")
                saved = True
                break
            else:
                print("[WARN] need exactly 3 points before saving (L-eye, R-eye, Chin)")

    cv2.destroyWindow(win)
    return saved


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Face overlay utility (annotate anchor points).")
    parser.add_argument("--annotate", type=str, help="Path to base image for annotation (e.g., blue_cap.png)")
    parser.add_argument("--out", type=str, default="anchor_points.csv", help="Output CSV for the 3 points")
    args = parser.parse_args()

    if args.annotate:
        annotate_anchor_points(args.annotate, args.out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
