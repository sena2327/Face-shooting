#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np
import os

# === 設定ここから ======================================
CAM_INDEX = 0
BASE_IMAGE_PATH = "img/blue_cap.png"   # ヘルメット画像（PNG, α付き推奨）
ANCHOR_CSV_PATH = "blue_cap.csv"       # blue_cap 上でクリックした3点の座標
# 黒い部分を「顔をはめ込む穴」とみなすためのしきい値
BLACK_THRESH = 10
# =====================================================

# FaceMesh の Face Oval インデックス（一般的によく使われるセット）
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]


def load_base_image(path):
    """
    ヘルメット画像を RGBA で読み込む。
    黒い領域（穴）を一度だけ検出して、その矩形を返す。
    """
    base = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if base is None:
        print(f"[ERR] base image not found: {path}")
        return None, None

    h, w = base.shape[:2]

    # RGBA でない場合は α=255 を付ける
    if base.ndim == 3 or base.shape[2] == 3:
        a = np.full((h, w, 1), 255, dtype=np.uint8)
        base = np.concatenate([base, a], axis=2)

    # 黒っぽい領域を「顔の穴」とみなす
    bgr = base[:, :, :3]
    mask_black = (
        (bgr[:, :, 0] < BLACK_THRESH) &
        (bgr[:, :, 1] < BLACK_THRESH) &
        (bgr[:, :, 2] < BLACK_THRESH)
    )

    # 黒い部分は「穴」として alpha=0 にする（下の顔が見える）
    alpha = base[:, :, 3]
    alpha[mask_black] = 0
    base[:, :, 3] = alpha

    ys, xs = np.where(mask_black)
    if xs.size == 0 or ys.size == 0:
        print("[WARN] no black region in base image; face will NOT be fitted into a hole.")
        return base, None

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    hole_rect = (x0, y0, x1, y1)  # 左上(x0,y0)〜右下(x1,y1)

    print(f"[INFO] detected hole rect in base image: {hole_rect}")
    return base, hole_rect


def load_anchor_points(csv_path):
    """
    blue_cap 上でクリックして保存した 3 点 (例: 両目 + 顎) を読み込む。
    フォーマットの例:
        x0,y0
        x1,y1
        x2,y2
    または pandas.to_csv などで index 列やヘッダが付いている場合にも対応する。

    戻り値: shape=(3,2) の float32 配列 or None
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] anchor csv not found: {csv_path}")
        return None

    try:
        # まずヘッダ 1 行を飛ばして数値だけ読み込むことを試みる
        data = np.genfromtxt(csv_path, delimiter=",", dtype=np.float32, skip_header=1)
    except Exception as e:
        print(f"[WARN] failed to load anchor csv '{csv_path}': {e}")
        return None

    # 1 行だけなどの場合に対応
    if data.ndim == 1:
        data = data[None, :]

    # pandas.to_csv(index=True) の場合など、先頭列が index で残りが x,y というケースに対応
    if data.shape[1] >= 3:
        # 最後の 2 列を (x, y) とみなす
        data = data[:, -2:]

    if data.shape[0] < 3 or data.shape[1] < 2:
        print(f"[WARN] anchor csv has invalid shape after cleanup: {data.shape}")
        return None

    pts = data[:3, :2].astype(np.float32)
    print(f"[INFO] loaded anchor points from {csv_path}:\n{pts}")
    return pts


def detect_face_landmarks(frame_bgr, face_mesh):
    """
    Utility to run MediaPipe FaceMesh once and return the first face's landmarks.
    戻り値: NormalizedLandmarkList or None
    """
    if face_mesh is None:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    return res.multi_face_landmarks[0]


def extract_face_with_facemesh(frame_bgr, face_mesh, face_oval_idx, landmarks=None):
    """
    MediaPipe FaceMesh で顔の輪郭（face oval）を取って、
    その部分だけ切り抜いた BGR とマスクを返す。
    失敗時は (None, None)。
    """
    h, w = frame_bgr.shape[:2]
    lm = landmarks
    if lm is None:
        if face_mesh is None:
            return None, None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None, None
        lm = res.multi_face_landmarks[0]
    pts = []
    for idx in face_oval_idx:
        if idx < len(lm.landmark):
            pt = lm.landmark[idx]
            x = int(pt.x * w)
            y = int(pt.y * h)
            pts.append([x, y])

    if len(pts) < 3:
        return None, None

    pts_np = np.array(pts, dtype=np.int32)

    # 顔輪郭を塗りつぶしたマスク作成
    full_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(full_mask, [pts_np], 255)

    # 顔のバウンディングボックス
    xs = pts_np[:, 0]
    ys = pts_np[:, 1]
    x0 = max(0, int(xs.min()))
    y0 = max(0, int(ys.min()))
    x1 = min(w, int(xs.max()))
    y1 = min(h, int(ys.max()))
    if x1 <= x0 or y1 <= y0:
        return None, None

    face_crop = frame_bgr[y0:y1, x0:x1].copy()
    mask_crop = full_mask[y0:y1, x0:x1].copy()

    # 少し境界をぼかして自然に
    mask_crop = cv2.GaussianBlur(mask_crop, (11, 11), 5)

    return face_crop, mask_crop


def extract_face_keypoints(frame_bgr, face_mesh, landmarks=None):
    """
    MediaPipe FaceMesh から「左目・右目・顎」の3点を取得する。
    戻り値:
        shape=(3,2) の float32 配列 (左目, 右目, 顎) か、検出失敗時は None
    """
    h, w = frame_bgr.shape[:2]
    lm = landmarks
    if lm is None:
        if face_mesh is None:
            return None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]

    # 一般的に使われる FaceMesh のインデックス:
    # 左目: 33, 右目: 263, 顎(あごの先端): 152
    idx_left_eye = 33
    idx_right_eye = 263
    idx_chin = 152

    def lm_to_xy(index):
        if index >= len(lm.landmark):
            return None
        p = lm.landmark[index]
        x = float(p.x * w)
        y = float(p.y * h)
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
    3点対応で、カメラ画像中の顔の3点 (src_pts: 左目, 右目, 顎) を
    blue_cap 上の3点 (dst_pts: blue_cap.csv に保存した座標) に
    アフィン変換で一致させる。

    合成順:
      1. カメラフレーム(BGRA)をアフィン変換して blue_cap のキャンバスサイズにワープ
      2. その上から base_rgba (blue_cap.png) を αブレンドで被せる
    戻り値: RGBA
    """
    if base_rgba is None or frame_bgr is None:
        return None

    h_base, w_base = base_rgba.shape[:2]

    if src_pts is None or dst_pts is None:
        return base_rgba.copy()

    src_pts = np.asarray(src_pts, dtype=np.float32)
    dst_pts = np.asarray(dst_pts, dtype=np.float32)

    # 3点 (3x2) であることを確認
    if src_pts.shape != (3, 2) or dst_pts.shape != (3, 2):
        print(f"[WARN] invalid src/dst pts shape: src={src_pts.shape}, dst={dst_pts.shape}")
        return base_rgba.copy()

    # カメラフレームを BGRA に変換
    h_src, w_src = frame_bgr.shape[:2]
    a = np.full((h_src, w_src, 1), 255, dtype=np.uint8)
    frame_rgba = np.concatenate([frame_bgr, a], axis=2)

    # アフィン変換行列 (src → dst)
    M = cv2.getAffineTransform(src_pts, dst_pts)

    # blue_cap と同じサイズにワープ
    warped = cv2.warpAffine(
        frame_rgba,
        M,
        (w_base, h_base),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    bg = warped.astype(np.float32)      # 顔 (背景レイヤー)
    fg = base_rgba.astype(np.float32)   # blue_cap (前面レイヤー)

    alpha_fg = fg[:, :, 3:4] / 255.0  # blue_cap の α
    alpha_bg = bg[:, :, 3:4] / 255.0  # 顔（ワープ後）の α

    # 前面優先で合成
    out_rgb = fg[:, :, :3] * alpha_fg + bg[:, :, :3] * (1.0 - alpha_fg)
    out_a = alpha_fg + alpha_bg * (1.0 - alpha_fg)

    out = np.zeros_like(base_rgba, dtype=np.uint8)
    out[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(out_a[..., 0] * 255.0, 0, 255).astype(np.uint8)

    return out


def process_overlay_frame(frame_bgr, face_mesh, base_rgba, anchor_pts, face_oval_idx=FACE_OVAL_IDX):
    if frame_bgr is None:
        return None

    landmarks = detect_face_landmarks(frame_bgr, face_mesh)
    face_pts = extract_face_keypoints(frame_bgr, face_mesh, landmarks=landmarks)
    face_crop, mask_crop = extract_face_with_facemesh(frame_bgr, face_mesh, face_oval_idx, landmarks=landmarks)

    if base_rgba is not None and anchor_pts is not None and face_pts is not None:
        composite_rgba = composite_face_to_base(base_rgba, frame_bgr, face_pts, anchor_pts)
        composite_bgr = composite_rgba[:, :, :3]
    elif base_rgba is not None:
        composite_bgr = base_rgba[:, :, :3].copy()
    else:
        composite_bgr = frame_bgr.copy()

    debug_frame = frame_bgr.copy()
    if face_crop is not None and mask_crop is not None:
        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(face_crop, contours, -1, (0, 255, 0), 2)
        preview_h = max(1, frame_bgr.shape[0] // 3)
        preview_w = max(1, frame_bgr.shape[1] // 3)
        face_preview = cv2.resize(face_crop, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
        debug_frame[0:preview_h, 0:preview_w, :] = face_preview
    else:
        cv2.putText(debug_frame, "No face detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return {
        "frame_bgr": frame_bgr,
        "composite_bgr": composite_bgr,
        "debug_frame": debug_frame,
        "face_pts": face_pts,
    }


def get_overlay_outputs(cap, face_mesh, base_rgba, anchor_pts, face_oval_idx=FACE_OVAL_IDX):
    if cap is None or not cap.isOpened():
        return None
    ok, frame = cap.read()
    if not ok:
        return None
    return process_overlay_frame(frame, face_mesh, base_rgba, anchor_pts, face_oval_idx)


def get_composited_frame(cap, face_mesh, base_rgba, anchor_pts, face_oval_idx=FACE_OVAL_IDX):
    outputs = get_overlay_outputs(cap, face_mesh, base_rgba, anchor_pts, face_oval_idx)
    if outputs is None:
        return None
    return outputs["composite_bgr"]


def create_overlay_context(
    cam_index=CAM_INDEX,
    base_image_path=BASE_IMAGE_PATH,
    anchor_csv_path=ANCHOR_CSV_PATH,
):
    base_rgba, _ = load_base_image(base_image_path)
    anchor_pts = load_anchor_points(anchor_csv_path)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERR] cannot open camera index {cam_index}")
        cap.release()
        return None

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return {
        "cap": cap,
        "face_mesh": face_mesh,
        "base_rgba": base_rgba,
        "anchor_pts": anchor_pts,
        "face_oval_idx": FACE_OVAL_IDX,
    }


def release_overlay_context(ctx):
    if not ctx:
        return
    cap = ctx.get("cap")
    if cap is not None:
        cap.release()
    mesh = ctx.get("face_mesh")
    if mesh is not None:
        close_fn = getattr(mesh, "close", None)
        if callable(close_fn):
            close_fn()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Face overlay viewer / calibrator")
    parser.add_argument("--cam-index", type=int, default=CAM_INDEX, help="Camera index (default 0)")
    parser.add_argument("--base-image", type=str, default=BASE_IMAGE_PATH,
                        help="Helmet/base image path (default img/blue_cap.png)")
    parser.add_argument("--anchor-csv", type=str, default=ANCHOR_CSV_PATH,
                        help="CSV containing 3 anchor points (default blue_cap.csv)")
    args = parser.parse_args()

    ctx = create_overlay_context(args.cam_index, args.base_image, args.anchor_csv)
    if ctx is None:
        return

    cap = ctx["cap"]
    face_mesh = ctx["face_mesh"]
    base_rgba = ctx["base_rgba"]
    anchor_pts = ctx["anchor_pts"]
    face_oval_idx = ctx["face_oval_idx"]
    win_name = "Face Overlay Test (q to quit)"

    try:
        while True:
            outputs = get_overlay_outputs(cap, face_mesh, base_rgba, anchor_pts, face_oval_idx)
            if outputs is None:
                print("[WARN] camera frame read failed or processing aborted")
                break

            frame_bgr = outputs["frame_bgr"]
            debug_frame = outputs["debug_frame"]
            composite_bgr = outputs["composite_bgr"]
            h, w = frame_bgr.shape[:2]

            composite_resized = cv2.resize(composite_bgr, (w, h), interpolation=cv2.INTER_AREA)
            show = np.hstack([debug_frame, composite_resized])

            cv2.imshow(win_name, show)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
    finally:
        release_overlay_context(ctx)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
