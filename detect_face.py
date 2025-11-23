#!/usr/bin/env python3
import argparse, time, sys
import numpy as np
import cv2
import mediapipe as mp
import csv
import socket, struct
from time import monotonic

try:
    import pyautogui   # --control で使用（任意）
except Exception:
    pyautogui = None

mp_face = mp.solutions.face_mesh

# よく使うランドマークID（MediaPipe FaceMesh）
LM_LEFT_EYE_OUTER  = 33   # 左目の外側
LM_RIGHT_EYE_OUTER = 263  # 右目の外側
LM_NOSE_TIP        = 4    # 鼻先（近い点）
LM_FOREHEAD = 10          # 額中央付近
LM_CHIN      = 152        # 顎
# 目の内外端（目の高さ基準を安定化するために使用）
LM_LEFT_EYE_INNER  = 133
LM_RIGHT_EYE_INNER = 362

# 虹彩ランドマーク（refine_landmarks=True で有効）
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# 頬（ほお）判定に使う代表ランドマーク（必要に応じて調整）
CHEEK_L_IDS = [61, 129, 205]   # 左ほお（口角・鼻横・頬外）
CHEEK_R_IDS = [291, 358, 425]  # 右ほお（対称側）

def eye_normalize_single(outer, inner, pt):
    """片目（外端-内端）基準で、回転+スケール正規化（横幅=1）。2Dピクセル座標→正規化座標。"""
    mid = 0.5 * (outer + inner)
    v = inner - outer
    angle = np.arctan2(float(v[1]), float(v[0]))
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    p = (pt - mid)
    p_rot = R @ p
    scale = max(1e-6, np.linalg.norm(v))
    return p_rot / scale  # 片目幅=1 スケール

def iris_center_radius(pts):
    """4点の虹彩ランドマークから中心と半径（平均距離）を返す。入力は同一座標系。"""
    c = np.mean(np.stack(pts, axis=0), axis=0)
    r = float(np.mean([np.linalg.norm(p - c) for p in pts]))
    return c, r

# --- EAR method for eye openness (6 landmarks) ---
def ear_for_eye(p1, p2, p3, p4, p5, p6):
    # (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    return float((l2(p2, p6) + l2(p3, p5)) / (2.0 * max(1e-6, l2(p1, p4))))

def l2(a, b):
    return float(np.linalg.norm(a-b) + 1e-9)

def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))

def safe_ratio(num, den):
    return float(num / max(1e-6, den))

def draw_text_bg(img, text, org, font, scale, color, thickness=2, bg_color=(255,255,255), pad=4):
    """
    cv2.putText に白背景をつけたユーティリティ。
    img: 描画先 (frame)
    text: 文字列
    org: (x, y) 左下座標
    color: テキスト色（黒など）
    bg_color: 背景色（デフォルト白）
    """
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(img,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + pad),
                  bg_color,
                  thickness=-1)
    cv2.putText(img, text, org, font, scale, color, thickness)

def eyes_normalize(le_out, re_out, pt):
    """
    両目外端(33,263)を基準に:
      - 目の中心へ平行移動
      - 目線（左右目外端のベクトル）に合わせて回転
      - 両目距離でスケール正規化
    を行った座標を返す（2D）。これにより yaw/roll/スケールの影響を抑えた安定なYが得られる。
    入出力はピクセル座標ベース。
    """
    mid = 0.5 * (le_out + re_out)
    v = re_out - le_out
    angle = np.arctan2(float(v[1]), float(v[0]))
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    p = (pt - mid)
    p_rot = R @ p
    scale = max(1e-6, np.linalg.norm(v))
    return p_rot / scale  # (x', y')。両目距離=1の座標系

def pitch_by_simple(fore, chin):
    v = chin - fore
    return (v[1] / (np.linalg.norm(v) + 1e-9)) - 0.5  # おおよそ -0.5..+0.5

def pitch_by_ratio(nose, le_out, re_out, le_in=None, re_in=None):
    # 目線の高さ: 目の外端と内端の平均Yがあればそれを使用、なければ外端のみ
    if le_in is not None and re_in is not None:
        eye_y = 0.25*(le_out[1] + re_out[1] + le_in[1] + re_in[1])
    else:
        eye_y = 0.5*(le_out[1] + re_out[1])
    # スケール不変にするために両目の距離で割る
    inter_oc = l2(le_out, re_out)
    if inter_oc < 1e-6:
        return 0.0
    return (nose[1] - eye_y) / inter_oc  # 正: 鼻が目線より下（うつむき）

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--flip", action="store_true", help="Mirror preview (selfie view)")
    ap.add_argument("--control", action="store_true", help="Enable OS cursor control (needs pyautogui)")
    ap.add_argument("--calib-sec", type=float, default=2.0, help="Calibration duration at startup (seconds)")
    ap.add_argument("--gain", type=float, default=1.8, help="Yaw->screen X sensitivity (larger = more movement)")
    ap.add_argument("--deadzone", type=float, default=0.02, help="Deadzone around neutral")
    ap.add_argument("--ema", type=float, default=0.25, help="EMA smoothing factor (0..1; larger = heavier)")
    ap.add_argument("--pitch-mode", choices=["simple","ratio","hybrid"], default="hybrid",
                    help="Pitch method: simple=forehead-chin vector, ratio=nose vs eye-line, hybrid=average")
    ap.add_argument("--gain-x", type=float, default=None, help="X gain (defaults to --gain)")
    ap.add_argument("--gain-y", type=float, default=None, help="Y gain (defaults to --gain)")
    ap.add_argument("--y-mode", choices=["pitch","nose","nose_stab"], default="nose_stab",
                    help="Y control: pitch=head pitch, nose=nose up/down (scaled by inter-ocular), nose_stab=eye-line normalized nose up/down (yaw/roll invariant)")
    ap.add_argument("--depth-comp", choices=["off","scale","decouple"], default="scale",
                    help="Depth compensation: off=none, scale=gain scales with distance, decouple=remove Z leakage into Y")
    ap.add_argument("--depth-gamma", type=float, default=1.0,
                    help="Exponent for --depth-comp=scale (1.0 linear)")
    ap.add_argument("--depth-beta", type=float, default=0.5,
                    help="Removal factor for Z->Y leakage with --depth-comp=decouple (0..1)")
    ap.add_argument("--invert-x", action="store_true", help="Invert X axis")
    ap.add_argument("--invert-y", action="store_true", help="Invert Y axis")
    ap.add_argument("--save-xy", type=str, default="",
                    help="Save XY trajectory CSV at exit (columns: t,x_norm,y_norm,X,Y)")
    ap.add_argument("--plot-xy", action="store_true",
                    help="Show XY trajectory plot at exit (requires matplotlib)")
    ap.add_argument("--plot-xy-save", type=str, default="",
                    help="Also save XY trajectory figure to a PNG file on exit (used with --plot-xy)")

    # --- Game I/O & detection options ---
    ap.add_argument("--udp", action="store_true",
                    help="Send tracking/features via UDP each frame")
    ap.add_argument("--udp-host", type=str, default="127.0.0.1",
                    help="UDP destination host (Unity receiver)")
    ap.add_argument("--udp-port", type=int, default=5005,
                    help="UDP destination port (Unity receiver)")
    ap.add_argument("--th-mouth", type=float, default=0.40,
                    help="Threshold for mouth_open to trigger shoot")
    ap.add_argument("--th-eye-open", type=float, default=0.26,
                    help="Eye open threshold (above=OPEN)")
    ap.add_argument("--th-eye-close", type=float, default=0.12,
                    help="Eye close threshold (below=CLOSE) for hysteresis")
    ap.add_argument("--wink-refractory-ms", type=int, default=250,
                    help="Minimum interval between wink triggers (ms)")
    ap.add_argument("--shoot-refractory-ms", type=int, default=180,
                    help="Minimum interval between shoot triggers (ms)")
    ap.add_argument("--brow-scale", type=float, default=2.0,
                    help="Gain for brow raise delta (0..1 after clamp)")
    ap.add_argument("--eye-mode", choices=["lid","iris","ear"], default="ear",
                    help="Eye openness metric: lid=eyelid ratio (vertical/width), iris=occlusion around iris, ear=Eye Aspect Ratio (6 landmarks, default)")
    ap.add_argument("--udp-extra-weapon", action="store_true",
                    help="Append extra weapon id float to UDP payload (for shooter-side decision)")
    ap.add_argument("--udp-mouth-shape", action="store_true", default=True,
                    help="Append mouth shape features (aspect, norm_w) to UDP payload (adds 2 floats) [default: ON]")
    args = ap.parse_args()

    # 補正オフセット（上下左右）と反転フラグ
    x_bias = 0.0   # 最終正規化座標に足し込む（-0.5..+0.5 目安）
    y_bias = 0.0
    invert_x = not args.invert_x  # default: left turn moves left
    invert_y = bool(args.invert_y)

    if args.control and pyautogui is None:
        print("pyautogui not found. Please `pip install pyautogui` or omit --control.")
        args.control = False

    gain_x = args.gain if args.gain_x is None else args.gain_x
    gain_y = (args.gain * 2.5) if args.gain_y is None else args.gain_y  # Y軸は既定で感度高め

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Camera open failed"); sys.exit(1)

    # 画面解像度
    if pyautogui is not None:
        SCR_W, SCR_H = pyautogui.size()
    else:
        SCR_W, SCR_H = 1920, 1080

    # --- UDP setup (optional) ---
    udp_sock = None
    udp_dest = None
    if args.udp:
        try:
            udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_dest = (args.udp_host, args.udp_port)
            print(f"UDP: sending to {udp_dest[0]}:{udp_dest[1]}")
        except Exception as e:
            print(f"UDP init failed: {e}")
            args.udp = False

    mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,      # 虹彩ランドマークを有効化
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # 起動直後に中立キャリブ
    calib_until = time.time() + args.calib_sec
    yaw0_accum, yaw0_n = 0.0, 0
    yaw0 = 0.0
    # 鼻Y（正規化）用の平滑化と基準
    nose_ema = None
    nose_stab_ema = None
    nose0_accum, nose0_n, nose0 = 0.0, 0, 0.0
    nose_stab0_accum, nose_stab0_n, nose_stab0 = 0.0, 0, 0.0
    pitch_ema = None
    pitch0_accum, pitch0_n, pitch0 = 0.0, 0, 0.0
    yaw_ema = None
    # 奥行き基準（両目間距離px と 相対Z）
    interoc0_accum, interoc0_n, interoc0 = 0.0, 0, 0.0
    relz0_accum, relz0_n, relz0 = 0.0, 0, 0.0

    # --- Feature EMAs (for stability) ---
    mouth_open_ema = None
    eyeL_open_ema  = None
    eyeR_open_ema  = None
    brow_metric_ema = None  # raw metric before baseline

    # --- Smile metric (for "smile score") ---
    smile_metric_ema = None
    smile0_accum, smile0_n, smile0 = 0.0, 0, 0.0

    # --- Baselines for features (captured during calib window) ---
    mouth0_accum, mouth0_n, mouth0 = 0.0, 0, 0.0
    eyeL0_accum, eyeL0_n, eyeL0 = 0.0, 0, 0.0
    eyeR0_accum, eyeR0_n, eyeR0 = 0.0, 0, 0.0
    brow0_accum, brow0_n, brow0 = 0.0, 0, 0.0

    # --- Cheek (puff) detection state ---
    cheekL_base, cheekR_base = 0.0, 0.0
    cheek_base_n = 0
    cheekL_ema, cheekR_ema = None, None
    CHEEK_ALPHA = 0.2        # EMA for cheeks
    CHEEK_ENTER = 0.010      # 入り閾値（要調整：カメラ/距離で変動）
    CHEEK_EXIT  = 0.006      # 抜け閾値（ヒステリシス）
    cheek_puff = 0.0         # 0..1 の強度（既定はOFF）

    # --- Discrete event states & refractory timers ---
    last_winkL_t = 0.0
    last_winkR_t = 0.0
    last_shoot_t = 0.0
    wink_left = 0.0
    wink_right = 0.0
    shoot = 0.0

    # 設定モード：起動時にオフセット/反転/ゲインを調整できる
    settings_mode = True
    setup_step = 1  # 1: Center, 2: X gain, 3: Y gain
    traj = []

    print("Startup: Please face forward (auto-centering calibration).\n"
          " r: recalibrate / q or Esc: quit")

    while True:
        ok, frame = cap.read()
        if not ok: break
        if args.flip:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 検出
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)

        yaw_raw = None   # 正向き：右向き（画面右へ動く）
        pitch_raw = None # 正向き：うつむき方向（後で上下反転して使用）
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            # 2D正規化座標 → ピクセル
            def p(id):
                return np.array([lm[id].x*w, lm[id].y*h], dtype=np.float32)
            def mean_z(ids):
                return float(np.mean([lm[i].z for i in ids]))
            # === Face centroid (pixel & normalized) ===
            coords = np.array([[pt.x * w, pt.y * h] for pt in lm], dtype=np.float32)
            face_cx_px, face_cy_px = coords.mean(axis=0)
            face_cx_norm = float(np.clip(face_cx_px / max(1.0, w), 0.0, 1.0))
            face_cy_norm = float(np.clip(face_cy_px / max(1.0, h), 0.0, 1.0))

            nose  = p(LM_NOSE_TIP)
            le_out = p(LM_LEFT_EYE_OUTER)
            re_out = p(LM_RIGHT_EYE_OUTER)

            inter_oc_px = l2(le_out, re_out)

            # 目線基準で正規化した鼻のY（yaw/roll/scaleの影響を軽減）
            nose_stab_xy = eyes_normalize(le_out, re_out, nose)  # 両目中心=0, 目線水平, 両目距離=1
            nose_stab_val = float(nose_stab_xy[1])
            if nose_stab_ema is None:
                nose_stab_ema = nose_stab_val
            else:
                a = np.clip(args.ema, 0.0, 1.0)
                nose_stab_ema = a*nose_stab_ema + (1.0-a)*nose_stab_val

            # 相対Z: MediaPipeのzを両目距離(px)で正規化（スケール不変化）
            nose_z = getattr(res.multi_face_landmarks[0].landmark[LM_NOSE_TIP], "z", 0.0)
            rel_z = float(nose_z) / max(1e-6, inter_oc_px)

            # 両目距離でスケール正規化した鼻のY値（Yは画像では下向きが正）
            inter_oc = l2(le_out, re_out)
            nose_val = float(nose[1] / max(1e-6, inter_oc))  # スケール不変の鼻高さ指標
            # EMA 平滑化（鼻）
            if nose_ema is None:
                nose_ema = nose_val
            else:
                a = np.clip(args.ema, 0.0, 1.0)
                nose_ema = a*nose_ema + (1.0-a)*nose_val

            # 鼻先から左右目尻までの距離比較 → ヨーの符号・大きさ
            dL = l2(nose, le_out)
            dR = l2(nose, re_out)
            yaw_raw = (dR - dL) / (dR + dL)   # だいたい [-1, +1] に収まる

            # 縦方向（ピッチ）: 2方式 -> simple(額-顎), ratio(鼻と目線の相対位置)
            fore = p(LM_FOREHEAD)
            chin = p(LM_CHIN)
            le_in = p(LM_LEFT_EYE_INNER)
            re_in = p(LM_RIGHT_EYE_INNER)

            pr_simple = pitch_by_simple(fore, chin)
            pr_ratio  = pitch_by_ratio(nose, le_out, re_out, le_in, re_in)

            if args.pitch_mode == "simple":
                pitch_raw = pr_simple
            elif args.pitch_mode == "ratio":
                pitch_raw = pr_ratio
            else:
                pitch_raw = 0.5*(pr_simple + pr_ratio)

            # EMA 平滑化
            if yaw_ema is None:
                yaw_ema = yaw_raw
            else:
                a = np.clip(args.ema, 0.0, 1.0)
                yaw_ema = a*yaw_ema + (1.0-a)*yaw_raw

            if pitch_ema is None:
                pitch_ema = pitch_raw
            else:
                pitch_ema = a*pitch_ema + (1.0-a)*pitch_raw

            # --- Facial features for game I/O ---
            a = np.clip(args.ema, 0.0, 1.0)

            # Mouth openness: inner lip vertical / mouth width
            mouth_h = l2(p(13), p(14))
            mouth_w = l2(p(78), p(308))
            mouth_open_raw = np.clip(safe_ratio(mouth_h, mouth_w), 0.0, 1.0)
            if mouth_open_ema is None:
                mouth_open_ema = mouth_open_raw
            else:
                mouth_open_ema = a*mouth_open_ema + (1.0-a)*mouth_open_raw

            # --- MAR (Mouth Aspect Ratio) & Smile metric ---
            # MAR 自体は mouth_open_raw (mouth_h / mouth_w) がそのまま指標になる。
            mar_raw = mouth_open_raw

            # 笑顔スコア用：口の中心と両端の相対高さを使う
            mouth_center = 0.5 * (p(13) + p(14))   # 上下唇の中点
            corner_L = p(61)
            corner_R = p(291)
            # 「口の中心よりどれだけ口角が上にあるか」を両側平均し、両目距離で正規化
            smile_raw = ((mouth_center[1] - corner_L[1]) + (mouth_center[1] - corner_R[1])) / (2.0 * max(1e-6, inter_oc_px))
            # EMA 平滑化
            if smile_metric_ema is None:
                smile_metric_ema = smile_raw
            else:
                smile_metric_ema = a*smile_metric_ema + (1.0-a)*smile_raw

            # Mouth shape features for shooter-side decision
            aspect = float(mouth_open_raw)  # mouth_h / mouth_w (0..1)
            norm_w = float(mouth_w / max(1e-6, inter_oc_px))  # width normalized by inter-ocular px

            # Eye openness metric (per eye)
            # 共通で使う目の内外端と目幅（外端-内端）
            le_outer, le_inner = p(LM_LEFT_EYE_OUTER), p(LM_LEFT_EYE_INNER)
            re_outer, re_inner = p(LM_RIGHT_EYE_OUTER), p(LM_RIGHT_EYE_INNER)
            eyeL_w = l2(le_outer, le_inner)
            eyeR_w = l2(re_outer, re_inner)
            if args.eye_mode == "iris":
                # 片目ごとに、虹彩が上下まぶたからどれだけ露出しているかを評価
                # 正規化座標系：片目の内外端で横幅=1に正規化
                # 左目

                # 左・右の上下まぶた
                L_up, L_dn = p(159), p(145)
                R_up, R_dn = p(386), p(374)

                # 虹彩4点
                L_iris_pts = [p(i) for i in LEFT_IRIS]
                R_iris_pts = [p(i) for i in RIGHT_IRIS]

                # 左目：座標正規化
                L_iris_n = [eye_normalize_single(le_outer, le_inner, q) for q in L_iris_pts]
                L_up_n   = eye_normalize_single(le_outer, le_inner, L_up)
                L_dn_n   = eye_normalize_single(le_outer, le_inner, L_dn)
                L_c, L_r = iris_center_radius(L_iris_n)

                # 右目：座標正規化
                R_iris_n = [eye_normalize_single(re_outer, re_inner, q) for q in R_iris_pts]
                R_up_n   = eye_normalize_single(re_outer, re_inner, R_up)
                R_dn_n   = eye_normalize_single(re_outer, re_inner, R_dn)
                R_c, R_r = iris_center_radius(R_iris_n)

                # 上下まぶたから虹彩中心までの垂直距離（正規化座標）
                # 画像Y軸は下向き正なので、符号に依存しないよう絶対値を取る
                L_d_up = abs(L_c[1] - L_up_n[1])
                L_d_dn = abs(L_dn_n[1] - L_c[1])
                R_d_up = abs(R_c[1] - R_up_n[1])
                R_d_dn = abs(R_dn_n[1] - R_c[1])

                # 開き度スコア：虹彩半径に対する上下の余裕比（小さすぎる→閉眼）
                # 典型値：OPENで ~1.0 以上、半目で ~0.5、閉じると <0.2
                L_score = max(0.0, min(L_d_up, L_d_dn) / max(1e-6, L_r))
                R_score = max(0.0, min(R_d_up, R_d_dn) / max(1e-6, R_r))

                # 0..1に軽くクリップ（上限は1.5程度まで許容してからクリップ）
                eyeL_open_raw = float(np.clip(L_score, 0.0, 1.0))
                eyeR_open_raw = float(np.clip(R_score, 0.0, 1.0))
            elif args.eye_mode == "ear":
                # EAR method using six landmarks per eye
                L_p1 = p(33);  L_p2 = p(159); L_p3 = p(158); L_p4 = p(133); L_p5 = p(144); L_p6 = p(145)
                R_p1 = p(263); R_p2 = p(386); R_p3 = p(385); R_p4 = p(362); R_p5 = p(380); R_p6 = p(374)
                eyeL_open_raw = np.clip(ear_for_eye(L_p1, L_p2, L_p3, L_p4, L_p5, L_p6), 0.0, 1.5)
                eyeR_open_raw = np.clip(ear_for_eye(R_p1, R_p2, R_p3, R_p4, R_p5, R_p6), 0.0, 1.5)
            else:
                # 従来法：まぶた縦/目幅
                eyeL_h = l2(p(159), p(145)); eyeL_w = l2(p(33),  p(133))
                eyeR_h = l2(p(386), p(374)); eyeR_w = l2(p(263), p(362))
                eyeL_open_raw = np.clip(safe_ratio(eyeL_h, eyeL_w), 0.0, 1.0)
                eyeR_open_raw = np.clip(safe_ratio(eyeR_h, eyeR_w), 0.0, 1.0)

            # EMA 平滑化
            if eyeL_open_ema is None:
                eyeL_open_ema = eyeL_open_raw
            else:
                eyeL_open_ema = a*eyeL_open_ema + (1.0-a)*eyeL_open_raw
            if eyeR_open_ema is None:
                eyeR_open_ema = eyeR_open_raw
            else:
                eyeR_open_ema = a*eyeR_open_ema + (1.0-a)*eyeR_open_raw

            # Brow raise metric: distance from eyelid center to eyebrow (normalized by eye width), averaged L/R
            eyeL_center_y = 0.5*(p(159)[1] + p(145)[1])
            eyeR_center_y = 0.5*(p(386)[1] + p(374)[1])
            browL_y = p(105)[1]
            browR_y = p(334)[1]
            browL_metric = safe_ratio((eyeL_center_y - browL_y), eyeL_w)  # larger when brow goes up
            browR_metric = safe_ratio((eyeR_center_y - browR_y), eyeR_w)
            brow_metric_raw = 0.5*(browL_metric + browR_metric)
            if brow_metric_ema is None:
                brow_metric_ema = brow_metric_raw
            else:
                brow_metric_ema = a*brow_metric_ema + (1.0-a)*brow_metric_raw

            # 起動時キャリブ：中立（正面/自然顔）オフセットを測る
            now = time.time()
            if now < calib_until:
                yaw0_accum += yaw_ema
                yaw0_n += 1
                yaw0 = yaw0_accum / max(1, yaw0_n)
                pitch0_accum += pitch_ema
                pitch0_n += 1
                pitch0 = pitch0_accum / max(1, pitch0_n)
                # 鼻Yの基準
                nose0_accum += nose_ema
                nose0_n += 1
                nose0 = nose0_accum / max(1, nose0_n)

                nose_stab0_accum += nose_stab_ema
                nose_stab0_n += 1
                nose_stab0 = nose_stab0_accum / max(1, nose_stab0_n)

                interoc0_accum += inter_oc_px
                interoc0_n += 1
                interoc0 = interoc0_accum / max(1, interoc0_n)

                relz0_accum += rel_z
                relz0_n += 1
                relz0 = relz0_accum / max(1, relz0_n)

                # 表情の基準
                mouth0_accum += mouth_open_ema; mouth0_n += 1
                mouth0 = mouth0_accum / max(1, mouth0_n)
                eyeL0_accum += eyeL_open_ema; eyeL0_n += 1
                eyeL0 = eyeL0_accum / max(1, eyeL0_n)
                eyeR0_accum += eyeR_open_ema; eyeR0_n += 1
                eyeR0 = eyeR0_accum / max(1, eyeR0_n)
                brow0_accum += brow_metric_ema; brow0_n += 1
                brow0 = brow0_accum / max(1, brow0_n)

                # Smile 基準（ニュートラル時の口角高さ）
                smile0_accum += smile_metric_ema
                smile0_n += 1
                smile0 = smile0_accum / max(1, smile0_n)
                
                # 頬の基準（無表情の平均 z）
                cheekL_base += mean_z(CHEEK_L_IDS)
                cheekR_base += mean_z(CHEEK_R_IDS)
                cheek_base_n += 1


            # === Cheek puff detection (z relative to baseline) ===
            if cheek_base_n > 0:
                baseL = cheekL_base / max(1, cheek_base_n)
                baseR = cheekR_base / max(1, cheek_base_n)
                zL = mean_z(CHEEK_L_IDS)
                zR = mean_z(CHEEK_R_IDS)
                # z の符号は環境依存になり得るので絶対差を採用
                dL = float(abs(zL - baseL))
                dR = float(abs(zR - baseR))
                if cheekL_ema is None:
                    cheekL_ema, cheekR_ema = dL, dR
                else:
                    cheekL_ema = (1.0-CHEEK_ALPHA)*cheekL_ema + CHEEK_ALPHA*dL
                    cheekR_ema = (1.0-CHEEK_ALPHA)*cheekR_ema + CHEEK_ALPHA*dR
                d = max(cheekL_ema, cheekR_ema)  # 片頬でもOK（両頬必須にしたいなら min に）
                # 0..1 にマッピング（ENTER 以上で 1、EXIT 以下で 0、間は線形）
                if d >= CHEEK_ENTER:
                    cheek_puff = 1.0
                elif d <= CHEEK_EXIT:
                    cheek_puff = 0.0
                else:
                    cheek_puff = (d - CHEEK_EXIT) / max(1e-6, (CHEEK_ENTER - CHEEK_EXIT))
                cheek_puff = float(np.clip(cheek_puff, 0.0, 1.0))
            else:
                cheek_puff = 0.0
            # 中立からの差分（デッドゾーン）
            yaw = yaw_ema - yaw0
            if abs(yaw) < args.deadzone:
                yaw = 0.0
            else:
                yaw = np.sign(yaw) * (abs(yaw) - args.deadzone) / (1.0 - args.deadzone)
                yaw = np.clip(yaw, -1.0, 1.0)

            pitch = pitch_ema - pitch0
            if abs(pitch) < args.deadzone:
                pitch = 0.0
            else:
                pitch = np.sign(pitch) * (abs(pitch) - args.deadzone) / (1.0 - args.deadzone)
                pitch = np.clip(pitch, -1.0, 1.0)

            # --- Y制御原料の選択 ---
            if args.y_mode == "nose":
                # 鼻の上下（基準からの差分）
                y_raw = nose_ema - nose0
            elif args.y_mode == "nose_stab":
                # 目線基準（回転/スケール補正）での鼻Y
                y_raw = nose_stab_ema - nose_stab0
            else:
                # 従来のピッチ
                y_raw = pitch_ema - pitch0

            # Y用デッドゾーンとクリップ（小さな揺れを無視）
            if abs(y_raw) < args.deadzone:
                y_val = 0.0
            else:
                y_val = np.sign(y_raw) * (abs(y_raw) - args.deadzone) / (1.0 - args.deadzone)
                y_val = np.clip(y_val, -1.0, 1.0)

            # --- 奥行き補正 ---
            eff_gain_x, eff_gain_y = gain_x, gain_y
            if args.depth_comp == "scale" and interoc0 > 1e-6:
                # 顔が近いほど inter_oc_px が大→ ゲインを下げる
                scale = (interoc0 / max(1e-6, inter_oc_px)) ** args.depth_gamma
                eff_gain_x *= scale
                eff_gain_y *= scale
            elif args.depth_comp == "decouple":
                # Z変化がYに混入する分を除去（鼻モード中心の補正）
                y_val = y_val - args.depth_beta * (rel_z - relz0)
                y_val = float(np.clip(y_val, -1.0, 1.0))

            # --- Feature post-processing & events ---
            # Normalized brow raise (0..1) from baseline
            brow_delta_up = max(0.0, brow_metric_ema - brow0)   # 眉が上がった量
            brow_raise = clamp01(args.brow_scale * brow_delta_up)

            # 眉をひそめるスコア：基準よりどれだけ下がったか（0..1）
            brow_delta_down = max(0.0, brow0 - brow_metric_ema)
            BROW_FROWN_SCALE = 4.0
            brow_frown = clamp01(BROW_FROWN_SCALE * brow_delta_down)

            # Smile score: ニュートラルとの差分（口角が上がる方向のみ 0..1 に正規化）
            smile_delta = max(0.0, smile_metric_ema - smile0)
            SMILE_SCALE = 8.0
            smile_score = clamp01(SMILE_SCALE * smile_delta)

            # MAR (Mouth Aspect Ratio) は mouth_open_ema をそのまま指標として使う
            mar = float(mouth_open_ema)

            # Eye open/close with hysteresis (per-eye adaptive thresholds for EAR)
            if args.eye_mode == "ear":
                # Baselines captured during calib window; fallback if unavailable
                baseL = eyeL0 if (eyeL0 > 1e-6) else (eyeL_open_ema if eyeL_open_ema is not None else 0.25)
                baseR = eyeR0 if (eyeR0 > 1e-6) else (eyeR_open_ema if eyeR_open_ema is not None else 0.25)
                # Ratios: tune if needed
                OPEN_RATIO  = 0.70  # >= 70% of baseline → OPEN
                CLOSE_RATIO = 0.55  # <= 55% of baseline → CLOSE
                th_open_L  = float(np.clip(baseL * OPEN_RATIO,  0.05, 0.60))
                th_close_L = float(np.clip(baseL * CLOSE_RATIO, 0.03, 0.55))
                th_open_R  = float(np.clip(baseR * OPEN_RATIO,  0.05, 0.60))
                th_close_R = float(np.clip(baseR * CLOSE_RATIO, 0.03, 0.55))
                # Apply hysteresis per eye
                left_open  = 1.0 if eyeL_open_ema >= th_open_L  else (0.0 if eyeL_open_ema <= th_close_L else None)
                right_open = 1.0 if eyeR_open_ema >= th_open_R else (0.0 if eyeR_open_ema <= th_close_R else None)
            else:
                th_open = args.th_eye_open
                th_close = args.th_eye_close
                left_open  = 1.0 if eyeL_open_ema >= th_open  else (0.0 if eyeL_open_ema <= th_close else None)
                right_open = 1.0 if eyeR_open_ema >= th_open else (0.0 if eyeR_open_ema <= th_close else None)

            if not hasattr(main, "_eyeL_state"): main._eyeL_state = 1.0
            if not hasattr(main, "_eyeR_state"): main._eyeR_state = 1.0
            if left_open is not None:  main._eyeL_state = left_open
            if right_open is not None: main._eyeR_state = right_open
            eyeL_state = main._eyeL_state
            eyeR_state = main._eyeR_state

            # Wink detection with refractory
            now_m = monotonic() * 1000.0
            wink_left = 0.0
            wink_right = 0.0
            if eyeL_state < 0.5 and eyeR_state > 0.5:
                if now_m - last_winkL_t >= args.wink_refractory_ms:
                    wink_left = 1.0
                    last_winkL_t = now_m
            if eyeR_state < 0.5 and eyeL_state > 0.5:
                if now_m - last_winkR_t >= args.wink_refractory_ms:
                    wink_right = 1.0
                    last_winkR_t = now_m

            # Shoot trigger from mouth openness with refractory
            shoot = 0.0
            if mouth_open_ema > args.th_mouth:
                if now_m - last_shoot_t >= args.shoot_refractory_ms:
                    shoot = 1.0
                    last_shoot_t = now_m

            # === Map using FACE CENTROID (no magnification) ===
            x_norm = clamp01(face_cx_norm + x_bias)
            y_norm = clamp01(face_cy_norm + y_bias)
            X = int(x_norm * (SCR_W - 1))
            Y = int(y_norm * (SCR_H - 1))

            # Log trajectory (normalized and screen coords)
            traj.append((time.time(), float(x_norm), float(y_norm), int(X), int(Y)))

            # --- UDP send (x,y,z + features) ---
            if args.udp and udp_sock is not None and udp_dest is not None:
                # Placeholder for shooter-side weapon id (always 0.0 for now)
                weapon_id = 0.0
                try:
                    # Base payload (13 floats):
                    base_vals = [
                        float(x_norm), float(y_norm), float(rel_z),
                        float(mouth_open_ema), float(eyeL_open_ema), float(eyeR_open_ema),
                        float(brow_raise), float(brow_frown), float(smile_score), float(brow_metric_ema),
                        float(wink_left), float(wink_right), float(shoot)
                    ]

                    # 常に mouth shape (aspect/norm_w/cheek_puff) を含めた16fパケットを送信し、
                    # brow_h を確実に届ける（PvP/Single の受信側は16fを優先して解釈する）
                    extra_vals = [float(aspect), float(norm_w), float(cheek_puff)]
                    pkt = struct.pack("16f", *(base_vals + extra_vals))
                    udp_sock.sendto(pkt, udp_dest)

                except Exception:
                    pass

            # カーソル移動（任意）
            if args.control and pyautogui is not None:
                try:
                    pyautogui.moveTo(X, Y, duration=0)
                except Exception:
                    pass

            # ===== 設定モード（ステップ式ウィザード） =====
            if settings_mode:
                overlay = frame.copy()
                if setup_step == 1:
                    # ① Center: Face forward. Adjust bias with arrows. Press Enter to continue.
                    cv2.putText(overlay, "[SETUP 1/3] Center: Face forward. Adjust bias with arrows. Press Enter to continue.", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2)
                    cv2.putText(overlay, f"biasX={x_bias:+.3f}  biasY={y_bias:+.3f}  invX={invert_x}  invY={invert_y}",
                                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(overlay, "R: recalibrate  X/Y: invert  q/Esc: quit", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
                elif setup_step == 2:
                    # ② Left/Right: Turn head to edges. Adjust X gain with Up/Down. Press Enter to continue.
                    cv2.putText(overlay, "[SETUP 2/3] Left/Right: Turn head to edges. Adjust X gain with Up/Down. Press Enter to continue.", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2)
                    cv2.putText(overlay, f"gainX={gain_x:.2f}  (Up:+  / Down:-)", (20, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(overlay, "R: recalibrate  X/Y: invert  q/Esc: quit", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
                else:
                    # ③ Up/Down: Move head to edges. Adjust Y gain with Up/Down. Press Enter to start.
                    cv2.putText(overlay, "[SETUP 3/3] Up/Down: Move head to edges. Adjust Y gain with Up/Down. Press Enter to start.", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2)
                    cv2.putText(overlay, f"gainY={gain_y:.2f}  (Up:+  / Down:-)", (20, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(overlay, "R: recalibrate  X/Y: invert  q/Esc: quit", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
                # Highlight current pointer position
                cv2.circle(overlay, (int(x_norm*w), int(y_norm*h)), 12, (0,255,255), -1)
                # Blend overlay
                frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

            # 可視化
            draw_text_bg(frame,
                         f"yaw={yaw_ema:+.3f} pitch={pitch_ema:+.3f} yaw0={yaw0:+.3f} pitch0={pitch0:+.3f} mode={args.pitch_mode}",
                         (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            draw_text_bg(frame,
                         f"X={X} Y={Y} (norm={x_norm:.3f},{y_norm:.3f})  gainX={gain_x:.2f} gainY={gain_y:.2f} dz={args.deadzone:.2f} yMode={args.y_mode}",
                         (20,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            draw_text_bg(frame,
                         f"depth={args.depth_comp} interOC(px)={inter_oc_px:.1f}/{interoc0:.1f} relZ={rel_z:+.4f}/{relz0:+.4f}",
                         (20,80 if args.y_mode!='nose' and args.y_mode != 'nose_stab' else 105),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            if args.y_mode == "nose":
                draw_text_bg(frame,
                             f"nose_ema={nose_ema:.4f} nose0={nose0:.4f} d={nose_ema - nose0:+.4f}",
                             (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            elif args.y_mode == "nose_stab":
                draw_text_bg(frame,
                             f"nose_stab={nose_stab_ema:.4f} base={nose_stab0:.4f} d={nose_stab_ema - nose_stab0:+.4f}",
                             (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            if not settings_mode:
                draw_text_bg(frame,
                             f"biasX={x_bias:+.3f} biasY={y_bias:+.3f} invX={invert_x} invY={invert_y}",
                             (20, 105 if args.y_mode=='pitch' else 130),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            # Feature HUD
            draw_text_bg(frame,
                         f"mouth={mouth_open_ema:.2f}  eyeL({args.eye_mode})={eyeL_open_ema:.2f}  eyeR({args.eye_mode})={eyeR_open_ema:.2f}  brow={brow_raise:.2f}",
                         (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            # (Optional) HUD for mouth shape features
            draw_text_bg(frame,
                         f"shape aspect={aspect:.2f} norm_w={norm_w:.2f}",
                         (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
            draw_text_bg(frame,
                         f"cheek_puff={cheek_puff:.2f}",
                         (20, 195 if args.eye_mode!='ear' else 215),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
            # Additional HUD: MAR / brow height / smile / frown (with white background)
            draw_text_bg(frame,
                         f"MAR={mar:.2f}  browH={brow_raise:.2f}  smile={smile_score:.2f}  frown={brow_frown:.2f}",
                         (20, 195 if args.eye_mode!='ear' else 235),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
            if args.eye_mode == "ear":
                try:
                    draw_text_bg(frame,
                                 f"EAR thr L: open>={th_open_L:.2f} close<={th_close_L:.2f}   R: open>={th_open_R:.2f} close<={th_close_R:.2f}",
                                 (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
                except Exception:
                    pass
            if wink_left > 0.5:
                cv2.putText(frame, "WINK_L", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            if wink_right > 0.5:
                cv2.putText(frame, "WINK_R", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            if shoot > 0.5:
                cv2.putText(frame, "SHOOT!", (220, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            # ガイドラインと現在位置点
            cx = int(x_norm * w)
            cy = int(y_norm * h)
            cv2.line(frame, (cx, 0), (cx, h), (255,255,0), 1)
            cv2.line(frame, (0, cy), (w, cy), (255,255,0), 1)
            cv2.circle(frame, (cx, cy), 8, (0,255,255), -1)
            # 参照点を描画
            for q in [nose, le_out, re_out]:
                cv2.circle(frame, (int(q[0]), int(q[1])), 3, (0,0,255), -1)

            if now < calib_until:
                cv2.putText(frame, f"Centering... {calib_until-now:.1f}s",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,220,255), 2)
                cv2.putText(frame, "Please face forward naturally.",
                            (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,220,255), 2)
        else:
            cv2.putText(frame, "No face", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,200,255), 2)

        cv2.putText(frame, "Enter: next/start  r: recalibrate  q/Esc: quit  X/Y: invert  --udp host:port  (features: mouth/eyes/brow/wink/shoot/cheek)", (20, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        cv2.imshow("Head Yaw Cursor", frame)
        key = cv2.waitKey(1) & 0xFF

        # --- 設定モードのキー操作 ---
        if settings_mode:
            # Enterで次へ（最後は開始）
            if key in (13, 10):
                if setup_step < 3:
                    setup_step += 1
                else:
                    settings_mode = False
            # ステップ別の操作
            if setup_step == 1:
                # バイアス調整（矢印）
                step_bias = 0.01
                if key == 81:  # 左
                    x_bias -= step_bias
                elif key == 83:  # 右
                    x_bias += step_bias
                elif key == 82:  # 上
                    y_bias -= step_bias
                elif key == 84:  # 下
                    y_bias += step_bias
                x_bias = float(np.clip(x_bias, -0.5, 0.5))
                y_bias = float(np.clip(y_bias, -0.5, 0.5))
            elif setup_step == 2:
                # X感度調整（上下キー）
                if key == 82:  # 上
                    gain_x = min(10.0, gain_x + 0.1)
                elif key == 84:  # 下
                    gain_x = max(0.1, gain_x - 0.1)
            elif setup_step == 3:
                # Y感度調整（上下キー）
                if key == 82:  # 上
                    gain_y = min(10.0, gain_y + 0.1)
                elif key == 84:  # 下
                    gain_y = max(0.1, gain_y - 0.1)
            # 反転トグル（どのステップでも可）
            if key in (ord('x'), ord('X')):
                invert_x = not invert_x
            if key in (ord('y'), ord('Y')):
                invert_y = not invert_y

        if key in (27, ord('q')):
            break
        if key == ord('r'):
            # 中立リセット
            calib_until = time.time() + args.calib_sec
            yaw0_accum, yaw0_n = 0.0, 0
            yaw0 = yaw_ema if yaw_ema is not None else 0.0
            pitch0_accum, pitch0_n = 0.0, 0
            pitch0 = pitch_ema if pitch_ema is not None else 0.0
            nose0_accum, nose0_n = 0.0, 0
            nose0 = nose_ema if nose_ema is not None else 0.0

            nose_stab0_accum, nose_stab0_n = 0.0, 0
            nose_stab0 = nose_stab_ema if nose_stab_ema is not None else 0.0

            interoc0_accum, interoc0_n = 0.0, 0
            interoc0 = inter_oc_px if 'inter_oc_px' in locals() else 0.0
            relz0_accum, relz0_n = 0.0, 0
            relz0 = rel_z if 'rel_z' in locals() else 0.0

            mouth0_accum, mouth0_n = 0.0, 0
            mouth0 = mouth_open_ema if mouth_open_ema is not None else 0.0
            eyeL0_accum, eyeL0_n = 0.0, 0
            eyeL0 = eyeL_open_ema if eyeL_open_ema is not None else 0.0
            eyeR0_accum, eyeR0_n = 0.0, 0
            eyeR0 = eyeR_open_ema if eyeR_open_ema is not None else 0.0
            brow0_accum, brow0_n = 0.0, 0
            brow0 = brow_metric_ema if brow_metric_ema is not None else 0.0
            smile0_accum, smile0_n = 0.0, 0
            smile0 = smile_metric_ema if smile_metric_ema is not None else 0.0

    # --- Save XY trajectory CSV on exit ---
    if args.save_xy and len(traj) > 0:
        try:
            with open(args.save_xy, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t","x_norm","y_norm","X","Y"])
                for (t,xn,yn,Xs,Ys) in traj:
                    w.writerow([f"{t:.6f}", f"{xn:.6f}", f"{yn:.6f}", Xs, Ys])
            print(f"Saved XY trajectory -> {args.save_xy} ({len(traj)} rows)")
        except Exception as e:
            print(f"Failed to save XY CSV: {e}")

    # --- Plot XY trajectory on exit ---
    if args.plot_xy:
        if len(traj) <= 1:
            print("No trajectory to plot (len<=1). Did tracking run long enough before quitting?")
        try:
            import matplotlib
            # Prefer a GUI backend if available; otherwise fall back to non-interactive Agg
            try:
                pass
            finally:
                if not matplotlib.get_backend().lower().startswith(("qt", "tk", "macosx", "wx")):
                    matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xs = [p[1] for p in traj]
            ys = [1.0 - p[2] for p in traj]  # math-style: upward positive
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111)
            ax.plot(xs, ys, linewidth=1)
            if len(xs) > 0:
                ax.scatter([xs[0]],[ys[0]], s=30)
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title('XY Trajectory (normalized)')
            ax.set_xlabel('x_norm'); ax.set_ylabel('y_norm (upward)')
            ax.grid(True, alpha=0.3)
            # Save if requested
            if args.plot_xy_save:
                fig.savefig(args.plot_xy_save, dpi=150, bbox_inches="tight")
                print(f"Saved plot PNG -> {args.plot_xy_save}")
            # Show only if a GUI backend is present
            try:
                plt.show()
            except Exception as e_show:
                if not args.plot_xy_save:
                    print(f"Plot window could not be shown (no GUI backend). Use --plot-xy-save FILE.png. Detail: {e_show}")
        except Exception as e:
            print(f"Plot failed (matplotlib needed): {e}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
