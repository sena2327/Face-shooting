import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# ===== 使用しているランドマーク群 =====
# （前に説明したものと対応）
EYE_LANDMARKS = [
    33, 133, 159, 145, 158, 144,   # 左目・まぶた
    263, 362, 386, 374, 385, 380,  # 右目・まぶた
    469, 470, 471, 472,            # 右虹彩
    474, 475, 476, 477             # 左虹彩
]

BROW_LANDMARKS = [
    33, 133, 159, 145,
    263, 362, 386, 374,
    105, 334  # 左右の眉＋眉検知に使う目のランドマーク
]

MOUTH_LANDMARKS = [
    13, 14,   # 上下唇
    61, 291,  # 左右口角
    78, 308   # 口の横側
]

# ===== 描画色（BGR） =====
COLOR_FACE_CENTER = (255, 255, 255)  # 顔重心（白）
COLOR_EYE = (0, 255, 0)              # 目（緑）
COLOR_BROW = (255, 0, 0)             # 眉（青）
COLOR_MOUTH = (0, 0, 255)            # 口（赤）

RADIUS = 3
THICKNESS = -1  # 塗りつぶし
RADIUS_COLORED = 6


def draw_used_points(frame, face_landmarks, image_w, image_h):
    """
    face_landmarks: mediapipeの1人分のランドマーク（468点）
    """

    # --- 468点すべてをピクセル座標に変換 ---
    coords = []
    for lm in face_landmarks.landmark:
        x = int(lm.x * image_w)
        y = int(lm.y * image_h)
        coords.append((x, y))
    coords = np.array(coords)  # shape: (468, 2)

    # === 顔の全ランドマーク（白） ===
    for (x, y) in coords:
        cv2.circle(frame, (x, y), RADIUS, COLOR_FACE_CENTER, THICKNESS)

    # === 顔重心 ===
    cx, cy = coords.mean(axis=0).astype(int)
    cv2.circle(frame, (cx, cy), RADIUS + 1, COLOR_FACE_CENTER, THICKNESS)

    # === 目に使っている点 ===
    for idx in EYE_LANDMARKS:
        x, y = coords[idx]
        cv2.circle(frame, (x, y), RADIUS_COLORED, COLOR_EYE, THICKNESS)

    # === 眉（高さ・frownに使っている点） ===
    for idx in BROW_LANDMARKS:
        x, y = coords[idx]
        cv2.circle(frame, (x, y), RADIUS_COLORED, COLOR_BROW, THICKNESS)

    # === 口 ===
    for idx in MOUTH_LANDMARKS:
        x, y = coords[idx]
        cv2.circle(frame, (x, y), RADIUS_COLORED, COLOR_MOUTH, THICKNESS)

    # === 簡単な凡例 ===
    cv2.putText(frame, "Face center", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FACE_CENTER, 1)
    cv2.putText(frame, "Eyes", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_EYE, 1)
    cv2.putText(frame, "Brows", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BROW, 1)
    cv2.putText(frame, "Mouth", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MOUTH, 1)


def main():
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,   # 虹彩を使うので True 推奨
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe はRGBで処理
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            h, w = frame.shape[:2]

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                draw_used_points(frame, face_landmarks, w, h)

            cv2.imshow("Used landmarks debug view", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q で終了
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()