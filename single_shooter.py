#!/usr/bin/env python3
import socket, struct, threading, time, math, random
import os
import subprocess
import sys
from pathlib import Path
import cv2
import pygame
import numpy as np
from collections import deque
from mediapipe import solutions as mp
from src_single.net import latest, lock, udp_loop
from src_single.render_utils import (
    draw_alpha_circle,
    draw_alpha_ellipse,
    draw_alpha_polyline,
    draw_alpha_line,
    blend_rgba,
    blend_rgba_tiled_x,
    build_hex_overlay,
)
from src_single.world import (
    Z_PLAYER,
    Z_ENEMY,
    MAX_ENEMY_BULLETS,
    Beam,
    Bullet,
    EnemyBullet,
    Target,
    ItemPickup,
    Boss,
)


HOST, PORT = "0.0.0.0", 5005   # detect_face.py と合わせる
SCR_W, SCR_H = 960, 540        # ゲーム画面サイズ（自由に変更）
WORLD_W, WORLD_H = SCR_W * 4, SCR_H * 3  # ワールド全体のサイズ（画面の10倍）
CAM_STEP = 20.0                          # 矢印キー1回あたりのカメラ移動量[px]
CAM_PAN_SPEED = 1800.0                    # 片目モード時のカメラ移動速度[px/s]（基準：水平方向）

# 擬似3D用のZレイヤーは src_single.world で定義
FPS = 60
DT = 1.0 / FPS
DEPTH_STRENGTH = 0.7
BASE_DIR = Path(__file__).resolve().parent
LAUNCHER_SCRIPT = BASE_DIR / "launcher_pygame.py"
BOSS_IMAGE_PATH = Path(os.environ.get("SINGLE_BOSS_IMAGE_PATH", str(BASE_DIR / "img" / "boss.png")))
BOSS_ANCHOR_CSV = Path(os.environ.get("SINGLE_BOSS_ANCHOR_CSV", str(BASE_DIR / "boss.csv")))
DEFAULT_FACE_PIPE = BASE_DIR / "build" / "user_face_frame.npz"
FACE_FRAME_PATH = Path(os.environ.get("SINGLE_FACE_FRAME_PATH", str(DEFAULT_FACE_PIPE)))


def relaunch_home_screen() -> None:
    """Re-open the launcher so players can return to the home screen."""
    if not LAUNCHER_SCRIPT.exists():
        print(f"[launcher] Missing launcher script: {LAUNCHER_SCRIPT}")
        return
    try:
        subprocess.Popen([sys.executable, str(LAUNCHER_SCRIPT)])
    except Exception as exc:  # pragma: no cover - fallback logging
        print(f"[launcher] Failed to relaunch home screen: {exc}")

EYE_CLOSE_TH = 0.18    # 閉眼しきい値（EAR/虹彩向け）
EYE_OPEN_TH  = 0.26    # 再オープンしきい値（ヒステリシス）


EYE_CLOSE_CONSEC = 30    # 連続フレーム数（この回数以上で発動）

# --- Shield activation window parameters ---
WINDOW_N = 25        # frames
WINDOW_RATIO = 0.90  # 90% closed within the last 50 frames

# --- Per-eye EAR hysteresis ratios (aligned with detect_face.py) ---
OPEN_RATIO  = 0.70  # >= 70% of baseline -> OPEN
CLOSE_RATIO = 0.55   # <= 55% of baseline -> CLOSE
TH_OPEN_MIN,  TH_OPEN_MAX  = 0.05, 0.60
TH_CLOSE_MIN, TH_CLOSE_MAX = 0.03, 0.55

# --- Weapon decision thresholds (mouth shape) ---
# aspect = mouth_h/mouth_w (縦に開くほど↑), norm_w = mouth_w / inter_oc_px（横に広いほど↑）
# 要件: ビームは「縦に大きく開いた時だけ」。横に広い場合はスプレッドへ。
BEAM_ASPECT_MIN = 0.80     # BEAM: 縦開きの下限（少し厳しめ）
SPREAD_NORMW_MIN = 0.40    # SPREAD: 横開きの下限

# Smoothing to avoid flicker of weapon switching
WEAPON_EMA_ALPHA = 0.20
WEAPON_DWELL_SEC = 0.25

# --- 片目モード切替の連続フレーム閾値 (pvp と揃える) ---
ONE_EYE_ENTER_FRAMES = 3        # 両目→片目判定
ONE_EYE_EXIT_FRAMES  = 5        # 片目→両目判定

# --- Wave / spawn settings ---
TARGETS_PER_SPOT = 6
WAVE_KILL_REQUIREMENT = 12
SPAWN_VARIANCE = 200.0
WAVE_CLEAR_DISPLAY_SEC = 2.0
SPAWN_SPOTS = [
    (200, 200),
    (WORLD_W - 200, 200),
    (WORLD_W - 200, WORLD_H - 200),
    (200, WORLD_H - 200),
]

PHASE_WAVE  = 0
PHASE_BOSS  = 1
PHASE_CLEAR = 2
NUM_WAVES = len(SPAWN_SPOTS)


def norm_to_px(xn, yn):
    return int(xn * (SCR_W-1)), int(yn * (SCR_H-1))


def compute_edge_indicator(
    sx: float,
    sy: float,
    scr_w: int,
    scr_h: int,
    margin: int = 20,
) -> tuple[int, int] | None:
    """Return a screen-edge position for off-screen objects, or None if visible."""
    if margin <= sx <= scr_w - margin and margin <= sy <= scr_h - margin:
        return None

    cx = scr_w * 0.5
    cy = scr_h * 0.5
    dx = float(sx) - cx
    dy = float(sy) - cy
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    half_w = max(1.0, scr_w * 0.5 - margin)
    half_h = max(1.0, scr_h * 0.5 - margin)
    scale_x = half_w / abs(dx) if abs(dx) > 1e-6 else float("inf")
    scale_y = half_h / abs(dy) if abs(dy) > 1e-6 else float("inf")
    scale = min(scale_x, scale_y)

    ex = cx + dx * scale
    ey = cy + dy * scale
    ex = float(np.clip(ex, margin, scr_w - margin))
    ey = float(np.clip(ey, margin, scr_h - margin))
    return int(round(ex)), int(round(ey))


def spawn_target_for_spot(spot_idx: int) -> Target:
    """指定スポット付近にターゲットを生成するヘルパー"""
    tg = Target(WORLD_W, WORLD_H)
    idx = int(spot_idx) % len(SPAWN_SPOTS)
    sx, sy = SPAWN_SPOTS[idx]
    jitter = SPAWN_VARIANCE
    tg.x = float(np.clip(sx + random.uniform(-jitter, jitter), tg.r, WORLD_W - tg.r))
    tg.y = float(np.clip(sy + random.uniform(-jitter, jitter), tg.r, WORLD_H - tg.r))
    tg.spawn_center = (sx, sy)
    tg.spawn_spot_index = idx
    return tg

def target_visible_on_screen(tg, cam_x, cam_y, margin=32.0):
    sx, sy = world_to_screen(tg.x, tg.y, cam_x, cam_y)
    return (-margin <= sx <= SCR_W + margin) and (-margin <= sy <= SCR_H + margin)

# --- World座標系と変換: ワールド<->FPS(画面)座標 ---
def world_to_screen(wx, wy, cam_x, cam_y):
    """
    ワールド座標 (wx, wy) を画面座標 (sx, sy) に変換する。
    cam_x, cam_y は「画面の左上」がワールド上のどこにあるかを表す。
    1人称視点で「画面自体を動かす」ときは cam_x, cam_y を更新する。
    """
    sx = int(wx - cam_x)
    sy = int(wy - cam_y)
    return sx, sy

# 擬似3D用投影関数
def project_to_screen(wx, wy, z, cam_x, cam_y):
    """
    擬似3D用の投影。
    z が大きいほど奥にある＝小さく＆画面中心へ寄るように見せる。
    返り値: (sx, sy, scale)
    """
    # z に応じたスケール（z=0 → 1.0, z が大きいほど縮小）
    depth_scale = 1.0 / (1.0 + float(z) * DEPTH_STRENGTH)

    # まず通常のカメラ変換（ワールド→画面座標系）
    sx = (wx - cam_x)
    sy = (wy - cam_y)

    # スケールを掛けつつ、画面中心に向かって寄せる
    sx = sx * depth_scale + SCR_W * 0.5 * (1.0 - depth_scale)
    sy = sy * depth_scale + SCR_H * 0.5 * (1.0 - depth_scale)

    return int(sx), int(sy), depth_scale

def screen_to_world_projected(sx, sy, z, cam_x, cam_y):
    depth_scale = 1.0 / (1.0 + float(z) * DEPTH_STRENGTH)
    offset_x = SCR_W * 0.5 * (1.0 - depth_scale)
    offset_y = SCR_H * 0.5 * (1.0 - depth_scale)
    wx = cam_x + (sx - offset_x) / depth_scale
    wy = cam_y + (sy - offset_y) / depth_scale
    return wx, wy

def screen_to_world(sx, sy, cam_x, cam_y):
    """
    画面座標 (sx, sy) をワールド座標 (wx, wy) に変換する。
    """
    wx = float(sx) + float(cam_x)
    wy = float(sy) + float(cam_y)
    return wx, wy

# 距離: 点 (px,py) と セグメント (x1,y1)-(x2,y2)
def dist_point_to_segment(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    vv = vx*vx + vy*vy
    if vv <= 1e-9:
        # segment is a point
        dx, dy = px - x1, py - y1
        return math.hypot(dx, dy)
    t = (wx*vx + wy*vy) / vv
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    projx = x1 + t * vx
    projy = y1 + t * vy
    return math.hypot(px - projx, py - projy)


def capture_user_face_once(max_attempts: int = 120) -> None:
    """Capture a single face frame plus landmarks and store them for boss compositing."""
    if FACE_FRAME_PATH is None:
        return
    if FACE_FRAME_PATH.exists():
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[BOSS_FACE] camera open failed; skipping face capture.")
        return
    success = False
    try:
        with mp.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as mesh:
            frame_index = 0
            fallback_frame = None
            fallback_pts = None
            for _ in range(max_attempts):
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    continue
                frame_index += 1
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = mesh.process(frame_rgb)
                if not res.multi_face_landmarks:
                    continue
                lm = res.multi_face_landmarks[0].landmark
                pts = np.array([[float(p.x), float(p.y)] for p in lm], dtype=np.float32)

                # Keep the latest valid frame before the 50th as a fallback
                if frame_index < 50:
                    fallback_frame = frame_rgb.copy()
                    fallback_pts = pts.copy()
                    continue

                # From the 50th valid frame onward, save the first detected face and break
                try:
                    FACE_FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(FACE_FRAME_PATH, frame=frame_rgb, points=pts)
                    print(f"[BOSS_FACE] saved face frame (>=50th) to {FACE_FRAME_PATH}")
                    success = True
                except Exception as exc:
                    print(f"[BOSS_FACE] failed to save face frame: {exc}")
                break
            # Fallback: if we never reached the 50th frame but have a valid face, save that instead
            if not success and fallback_frame is not None and fallback_pts is not None:
                try:
                    FACE_FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(FACE_FRAME_PATH, frame=fallback_frame, points=fallback_pts)
                    print(f"[BOSS_FACE] saved fallback face frame (<50th) to {FACE_FRAME_PATH}")
                    success = True
                except Exception as exc:
                    print(f"[BOSS_FACE] failed to save fallback face frame: {exc}")
    except Exception as exc:
        print(f"[BOSS_FACE] face capture error: {exc}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if not success:
        print("[BOSS_FACE] face capture skipped or failed; boss will use fallback sprite.")


# --- Helper: Load 3 anchor points from boss.csv ---
def load_boss_anchors(csv_path: str) -> np.ndarray | None:
    """Load 3 anchor points (left eye, right eye, chin) from boss.csv.

    The CSV is expected to have at least 3 rows with x,y pixel coordinates.
    Extra columns are ignored. Returns a (3,2) float32 array in boss-image pixel space.
    """
    pts = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                parts = line.replace("\t", ",").split(",")
                if len(parts) < 2:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                except ValueError:
                    continue
                pts.append([x, y])
                if len(pts) >= 3:
                    break
    except OSError as exc:
        print(f"[BOSS_FACE] failed to read boss anchors from {csv_path}: {exc}")
        return None

    if len(pts) != 3:
        print(f"[BOSS_FACE] boss anchor csv must contain 3 valid points, got {len(pts)}")
        return None

    arr = np.asarray(pts, dtype=np.float32)
    if arr.shape != (3, 2):
        print(f"[BOSS_FACE] boss anchors must be 3x2, got {arr.shape}")
        return None
    return arr


# --- Helper: Build boss sprite from face npz, warping face to boss anchors and overlaying boss.png ---
def build_boss_sprite_from_face_npz(frame: np.ndarray,
                                    points: np.ndarray,
                                    boss_img_path: str,
                                    boss_anchor_csv: str) -> np.ndarray | None:
    """Create boss sprite by:
    1) Taking the captured face frame + landmarks from MediaPipe (normalized coordinates),
    2) Mapping 3 anchor landmarks (left eye, right eye, chin) to the corresponding
       3 points from boss.csv,
    3) Warping the whole face frame into boss.png's coordinate system,
    4) Overlaying boss.png as the TOP layer so its 'hole' reveals the warped face.

    Returns an RGBA sprite in boss.png resolution, or None on failure.
    """
    if frame is None or points is None:
        return None

    # Load boss image (expects RGBA; if RGB, add opaque alpha)
    boss_rgba = cv2.imread(boss_img_path, cv2.IMREAD_UNCHANGED)
    if boss_rgba is None:
        print(f"[BOSS_FACE] failed to load boss image: {boss_img_path}")
        return None
    if boss_rgba.ndim != 3:
        print(f"[BOSS_FACE] boss image must be 3-channel or 4-channel, got shape {boss_rgba.shape}")
        return None
    if boss_rgba.shape[2] == 3:
        h, w = boss_rgba.shape[:2]
        a = np.full((h, w, 1), 255, dtype=np.uint8)
        boss_rgba = np.concatenate([boss_rgba, a], axis=2)
    bh, bw = boss_rgba.shape[:2]

    # Load boss anchor points (3x2, in boss-image pixel coordinates)
    boss_pts = load_boss_anchors(boss_anchor_csv)
    if boss_pts is None:
        # Cannot align without anchors; fall back to boss only
        return boss_rgba

    # We expect MediaPipe FaceMesh normalized landmarks (N x 2, x/y in [0,1]).
    pts_np = np.asarray(points, dtype=np.float32)
    if pts_np.ndim != 2 or pts_np.shape[1] != 2 or pts_np.shape[0] <= 263:
        print(f"[BOSS_FACE] unexpected landmark shape for user face: {pts_np.shape}")
        return boss_rgba

    # Extract 3 anchor landmarks: left eye outer (33), right eye outer (263), chin (152)
    anchor_idx = np.array([33, 263, 152], dtype=np.int64)
    user_pts_norm = pts_np[anchor_idx, :]  # 3x2 in normalized coordinates

    fh, fw = frame.shape[:2]

    # Convert all normalized landmarks to pixel coordinates for face bounding box
    all_pts_pix = np.zeros_like(pts_np, dtype=np.float32)
    all_pts_pix[:, 0] = pts_np[:, 0] * float(fw)
    all_pts_pix[:, 1] = pts_np[:, 1] * float(fh)

    x_min = float(np.min(all_pts_pix[:, 0]))
    x_max = float(np.max(all_pts_pix[:, 0]))
    y_min = float(np.min(all_pts_pix[:, 1]))
    y_max = float(np.max(all_pts_pix[:, 1]))

    # Add a small margin around the face and clamp to image bounds
    margin = 0.08 * max(fw, fh)
    x0 = max(0, int(x_min - margin))
    y0 = max(0, int(y_min - margin))
    x1 = min(fw, int(x_max + margin))
    y1 = min(fh, int(y_max + margin))

    if x1 <= x0 or y1 <= y0:
        # Fallback: use full frame if bounding box is degenerate
        x0, y0, x1, y1 = 0, 0, fw, fh

    # Shift user anchor points into the cropped ROI coordinates
    user_pts_pix = np.zeros((3, 2), dtype=np.float32)
    user_pts_pix[:, 0] = user_pts_norm[:, 0] * float(fw) - float(x0)
    user_pts_pix[:, 1] = user_pts_norm[:, 1] * float(fh) - float(y0)

    # Convert frame to BGRA and crop only the face region
    if frame.shape[2] == 3:
        # RGB (from MediaPipe) -> BGRA
        frame_rgba_full = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
    elif frame.shape[2] == 4:
        # RGBA (from some pipelines) -> BGRA
        frame_rgba_full = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
    else:
        # Fallback: try to ensure 4 channels
        if frame.ndim == 2 or frame.shape[2] == 1:
            frame_rgba_full = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGRA)
        else:
            frame_rgba_full = frame.copy()
    face_roi_rgba = frame_rgba_full[y0:y1, x0:x1]

    # Compute affine transform to map user face ROI landmarks -> boss anchor points
    try:
        M = cv2.getAffineTransform(user_pts_pix.astype(np.float32),
                                   boss_pts.astype(np.float32))
    except cv2.error as exc:
        print(f"[BOSS_FACE] getAffineTransform failed: {exc}")
        return boss_rgba

    # Warp the cropped face ROI into boss.png's coordinate system (BGRA).
    # Use a fully transparent border so anything outside the original ROI
    # becomes alpha=0 instead of black or "wrapped".
    face_rgba = cv2.warpAffine(
        face_roi_rgba,
        M,
        (bw, bh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # Compose final sprite: boss.png as TOP layer, face behind showing through the hole.
    alpha_boss = boss_rgba[:, :, 3:4].astype(np.float32) / 255.0
    alpha_face = face_rgba[:, :, 3:4].astype(np.float32) / 255.0

    # Where boss has alpha > 0, use boss; where its alpha=0 (hole), show face.
    out_rgb = (
        face_rgba[:, :, :3].astype(np.float32) * (1.0 - alpha_boss) +
        boss_rgba[:, :, :3].astype(np.float32) * alpha_boss
    )

    out_a = np.clip(
        alpha_face * (1.0 - alpha_boss) + alpha_boss,
        0.0,
        1.0,
    ) * 255.0

    out = np.concatenate(
        [out_rgb.astype(np.uint8), out_a.astype(np.uint8)],
        axis=2,
    )
    return out



def main():
    # --- Initialize sound (pygame.mixer) ---
    pygame.mixer.init()
    capture_user_face_once()
    def _load_sound(path):
        try:
            snd = pygame.mixer.Sound(path)
            return snd
        except Exception as e:
            print(f"[SOUND] Failed to load {path}: {e}")
            return None

    snd_bgm = _load_sound("sound/bgm.mp3")
    snd_beam   = _load_sound("sound/beam.mp3")
    snd_shot   = _load_sound("sound/shot.mp3")
    snd_shield_gen  = _load_sound("sound/shield_generate.mp3")
    snd_shield_frag = _load_sound("sound/shield_frag.mp3")
    if snd_shield_frag is not None:
        snd_shield_frag.set_volume(1.0)
    # New: hit sound when player is damaged without shield.
    snd_hitted = _load_sound("sound/hitted.mp3")
    # Played only when a new enemy Target is spawned (not on each enemy bullet).
    snd_enemy_appearance = _load_sound("sound/enemy_appearance.mp3")

    # シールド展開SE用のチャンネルハンドル
    shield_gen_channel = None

    # 受信スレッド開始
    th = threading.Thread(target=udp_loop, daemon=True)
    th.start()

    # ワールド
    bullets = []
    current_spot_index = 0
    kills_this_wave = 0
    wave_clear_timer = 0.0
    game_phase = PHASE_WAVE
    wave_index = 0
    boss_target = None
    boss_spawned = False
    targets = [spawn_target_for_spot(current_spot_index) for _ in range(TARGETS_PER_SPOT)]
    score = 0
    last_shot_time = 0.0
    shoot_cooldown = 0.08
    shield_until = 0.0
    shield_fx_start = 0.0
    particles = []  # list of dicts: {x,y,vx,vy,life}
    enemy_bullets = []
    enemy_spawn_timer = 0.0
    enemy_spawn_period = 1.0   # 平均1秒に1発
    # --- Field item pickups (dropped from defeated enemies) ---
    items = []
    # Drop probabilities per "mob wave phase" (0–3)
    # Each list is [HP, Attack, Shotgun, Beam]
    ITEM_DROP_PROBS_WAVE = {
        0: [0.05, 0.03, 0.02, 0.01],
        1: [0.06, 0.04, 0.10, 0.015],
        2: [0.07, 0.045, 0.30, 0.20],
        3: [0.08, 0.05, 0.04, 0.40],
    }
    # --- Item system (3 slots bottom-left, stack-based, currently empty) ---
    item_slots = [[], [], []]  # each slot is a stack (list); items not yet implemented
    item_selected = 0          # start at leftmost; 0: left, 1: middle, 2: right
    last_item_move_t = 0.0
    prev_item_use = False
    item_blink_until = 0.0
    # Weapon state
    weapon_last = 0  # 0: bullet, 1: spread, 2: beam
    weapon_last_change_t = time.time()
    aspect_ema = 0.0
    normw_ema = 0.0
    beams = []
    # Latch weapon per trigger-hold to avoid mixing within a burst
    prev_shoot = False
    shot_weapon_latched = None  # None until first press; then 0/1/2 while held
    crack_sprite = None
    return_to_home = False

    boss_face_sprite = None
    boss_face_mtime = 0.0

    def try_prepare_boss_face_sprite():
        """Load and composite the boss face once user_face data is available."""
        nonlocal boss_face_sprite, boss_face_mtime
        if boss_face_sprite is not None:
            return
        npz_path = FACE_FRAME_PATH
        if npz_path is None or not npz_path.exists():
            return
        try:
            mtime = npz_path.stat().st_mtime
        except OSError:
            return
        if mtime <= boss_face_mtime:
            return
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                frame = data["frame"]
                pts = data["points"]
        except Exception as exc:
            print(f"[BOSS_FACE] user_face npz load failed: {exc}")
            return

        # Use fixed paths img/boss.png and boss.csv (as requested)
        boss_img_path = str(BASE_DIR / "img" / "boss.png")
        boss_anchor_csv = str(BASE_DIR / "boss.csv")

        # Build boss sprite by warping the captured face to boss.csv anchors
        # and overlaying boss.png as the top layer with its hole.
        sprite = build_boss_sprite_from_face_npz(
            frame,
            pts,
            boss_img_path,
            boss_anchor_csv,
        )

        if sprite is not None:
            boss_face_sprite = sprite
            boss_face_mtime = mtime
            print("[BOSS_FACE] composited sprite ready.")

    player_hp_max = 3
    player_hp = player_hp_max
    game_over_flag = False
    ATTACK_BUFF_DURATION = 8.0
    attack_buff_mul = 1.0
    attack_buff_timer = 0.0
    has_shotgun = False
    has_beam = False
    shield_hp_max = 2     # シールドHPの上限
    shield_hp = shield_hp_max
    shield_on = False
    eye_closed_count = 0
    shield_grace = 0
    def decide_weapon(aspect_raw, normw_raw):
        nonlocal aspect_ema, normw_ema, weapon_last, weapon_last_change_t
        # EMA smooth
        aspect_ema = (1-WEAPON_EMA_ALPHA)*aspect_ema + WEAPON_EMA_ALPHA*float(aspect_raw)
        normw_ema  = (1-WEAPON_EMA_ALPHA)*normw_ema  + WEAPON_EMA_ALPHA*float(normw_raw)
        # Base decision (STRICT):
        # BEAM: 縦に大きく開いた時だけ（かつ横開きではない）
        if (aspect_ema >= BEAM_ASPECT_MIN) and (normw_ema < SPREAD_NORMW_MIN):
            wid = 2  # BEAM
        elif normw_ema >= SPREAD_NORMW_MIN:
            wid = 1  # SPREAD
        else:
            wid = 0  # BULLET
        # Dwell (hold previous weapon for a minimum time)
        now = time.time()
        if now - weapon_last_change_t < WEAPON_DWELL_SEC:
            wid = weapon_last
        elif wid != weapon_last:
            weapon_last = wid
            weapon_last_change_t = now
        return wid, aspect_ema, normw_ema

    # 画面
    win = "Gaze Shooter (Python)"
    cv2.namedWindow(win)

    # --- Load parallax space backgrounds (far/mid/near) ---
    def _load_rgba(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Background not found: {path}")
            return None
        # Resize to screen if needed
        if img.shape[1] != SCR_W or img.shape[0] != SCR_H:
            img = cv2.resize(img, (SCR_W, SCR_H), interpolation=cv2.INTER_LINEAR)
        # Ensure RGBA (add opaque alpha if missing)
        if img.shape[2] == 3:
            a = np.full((SCR_H, SCR_W, 1), 255, dtype=np.uint8)
            img = np.concatenate([img, a], axis=2)
        return img

    bg_far  = _load_rgba("img/space_far.png")
    bg_mid  = _load_rgba("img/space_mid.png")
    bg_near = _load_rgba("img/space_near.png")

    # --- Load multiple alien enemy sprites (random per target) ---
    # --- Load crack overlay (single sprite) ---
    crack_img = cv2.imread("img/glass_crack_1024.png", cv2.IMREAD_UNCHANGED)
    if crack_img is None:
        print("[WARN] Glass crack image not found: img/glass_crack_1024.png")
    enemy_sprites = []
    sprite_paths = [
        "img/alien_face_enemy_purple_256.png",
        "img/alien_face_enemy_green_256.png",
        "img/alien_face_enemy_blue_256.png",
        "img/alien_face_enemy_purple_512.png",
        "img/alien_face_enemy_green_512.png",
        "img/alien_face_enemy_blue_512.png",
    ]
    for path in sprite_paths:
        img_rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_rgba is not None and img_rgba.shape[2] == 4:
            enemy_sprites.append(img_rgba)
        else:
            print(f"[WARN] Sprite not found or invalid: {path}")

    if not enemy_sprites:
        print("[WARN] No enemy sprites loaded, using placeholder circles.")

    # --- Load opponent player sprite (PVP用の相手プレイヤー顔アイコン) ---
    opponent_sprite = cv2.imread("img/opponent_player.png", cv2.IMREAD_UNCHANGED)
    if opponent_sprite is None:
        print("[WARN] Opponent sprite not found: img/opponent_player.png")
    elif opponent_sprite.shape[2] != 4:
        print("[WARN] Opponent sprite has no alpha channel (expecting RGBA): img/opponent_player.png")

    # --- Load item sprites (field pickups + HUD icons) ---
    def _load_item_rgba(path, world_size=64, slot_size=40):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Item sprite not found: {path}")
            return None, None
        # world pickup icon
        ws = cv2.resize(img, (world_size, world_size), interpolation=cv2.INTER_AREA)
        # HUD icon (smaller)
        ss = cv2.resize(img, (slot_size, slot_size), interpolation=cv2.INTER_AREA)
        return ws, ss

    item_world_sprites = [None] * 4
    item_slot_icons = [None] * 4
    item_paths = [
        "img/HP_portion.png",     # 0: heal
        "img/Attack_portion.png", # 1: attack up
        "img/shotgun.png",        # 2: spread armor
        "img/lazer.png",          # 3: beam armor
    ]
    for idx, p in enumerate(item_paths):
        ws, ss = _load_item_rgba(p)
        item_world_sprites[idx] = ws
        item_slot_icons[idx] = ss

    # --- Prebuild AT-field style overlays ---
    hex_overlay = build_hex_overlay(SCR_W, SCR_H, cell=42, line_th=1, color=(0,165,255))  # orange-ish
    tint_overlay = np.full((SCR_H, SCR_W, 3), (0, 120, 255), dtype=np.uint8)  # light orange tint (BGR)

    clock_t0 = time.time()
    accum = 0.0

    # --- XY calibration over first N frames ---
    calib_N = 100
    calib_sumx = 0.0
    calib_sumy = 0.0
    calib_count = 0
    calib_done = False
    calib_cx = 0.5
    calib_cy = 0.5
    gain_xy = 5.0  # magnify around calibrated center

    # --- World (FPS) camera: 画面左上がワールド上のどこかを表す ---
    # ここではひとまず (0,0) スタート。
    # 矢印キーで cam_x, cam_y を動かして「ワールド座標が動いているか」を確認する。
    cam_x = 0.0  # world-x of window top-left
    cam_y = 0.0  # world-y of window top-left
    # FPS（照準）の正規化座標（0..1）。両目OPENのときだけ更新し、片目モード中は維持する。
    fps_nx = 0.5
    fps_ny = 0.5

    # --- Baseline values for expression features (smile/frown/brow_h) during calibration ---
    smile0_sum = 0.0
    frown0_sum = 0.0
    browh0_sum = 0.0
    smile0 = 0.0
    frown0 = 0.0
    browh0 = 0.0


    # --- Per-eye EAR baselines (collected during calibration gate) ---
    eyeL0_sum = 0.0
    eyeR0_sum = 0.0
    eye0_count = 0
    eyeL0 = 0.0
    eyeR0 = 0.0
    th_open_L = 0.0
    th_close_L = 0.0
    th_open_R = 0.0
    th_close_R = 0.0

    # Relative expression values (current - baseline, clamped at 0)
    rel_smile = 0.0
    rel_frown = 0.0
    rel_browh = 0.0

    # Hysteresis states (1.0=open, 0.0=close)
    eyeL_state = 1.0
    eyeR_state = 1.0

    # --- Sliding window for eyes-closed ratio ---
    closed_buf = deque(maxlen=WINDOW_N)
    closed_sum = 0

    # --- One-eye mode tracking (pvp parity) ---
    one_eye_mode = False
    one_eye_consec = 0
    both_eye_consec = 0

    if snd_bgm is not None:
        snd_bgm.play()

    while True:
        t = time.time()
        # 定速ループ
        # （OpenCVのwaitKeyに合わせてやや緩めに）
        with lock:
            x = latest["x"]
            y = latest["y"]
            raw_x = x
            raw_y = y

            # --- Calibrate center over first calib_N frames, then apply 5x around that center ---
            if not calib_done:
                calib_sumx += x
                calib_sumy += y
                calib_count += 1
                if calib_count >= calib_N:
                    calib_cx = calib_sumx / float(calib_count)
                    calib_cy = calib_sumy / float(calib_count)
                    calib_done = True
            else:
                dx = x - calib_cx
                dy = y - calib_cy
                x = float(np.clip(0.5 + gain_xy * dx, 0.0, 1.0))
                y = float(np.clip(0.5 + gain_xy * dy, 0.0, 1.0))

            brow = latest["brow"]
            # Expression features from sender (for baseline capture)
            smile_val = float(latest.get("smile", 0.0))
            frown_val = float(latest.get("frown", 0.0))
            browh_val = float(latest.get("brow_h", 0.0))
            # Per-eye openness values from sender (detect_face.py)
            eyeL_val = float(latest.get("eyeL", latest.get("eyeL_open", latest.get("eyeL_open_ema", 0.0))))
            eyeR_val = float(latest.get("eyeR", latest.get("eyeR_open", latest.get("eyeR_open_ema", 0.0))))

            # During the initial calibration frames, accumulate baseline EARs and expression features
            if not calib_done:
                eyeL0_sum += eyeL_val
                eyeR0_sum += eyeR_val
                eye0_count += 1
                smile0_sum += smile_val
                frown0_sum += frown_val
                browh0_sum += browh_val
            elif (th_open_L == 0.0 and th_open_R == 0.0):
                # Finalize thresholds once after calibration completes
                eyeL0 = (eyeL0_sum / max(1, eye0_count)) if eye0_count > 0 else max(eyeL_val, 0.25)
                eyeR0 = (eyeR0_sum / max(1, eye0_count)) if eye0_count > 0 else max(eyeR_val, 0.25)
                th_open_L  = float(np.clip(eyeL0 * OPEN_RATIO,  TH_OPEN_MIN,  TH_OPEN_MAX))
                th_close_L = float(np.clip(eyeL0 * CLOSE_RATIO, TH_CLOSE_MIN, TH_CLOSE_MAX))
                th_open_R  = float(np.clip(eyeR0 * OPEN_RATIO,  TH_OPEN_MIN,  TH_OPEN_MAX))
                th_close_R = float(np.clip(eyeR0 * CLOSE_RATIO, TH_CLOSE_MIN, TH_CLOSE_MAX))
                # Finalize baseline expression values (average over calibration frames)
                denom = max(1, eye0_count)
                smile0 = (smile0_sum / denom)
                frown0 = (frown0_sum / denom)
                browh0 = (browh0_sum / denom)
            # Update relative expression values (non-negative)
            if calib_done:
                rel_smile = max(0.0, smile_val - smile0)
                rel_frown = max(0.0, frown_val - frown0)
                rel_browh = max(0.0, browh_val - browh0)

        try_prepare_boss_face_sprite()

        # === Setup screen: hold the game until calibration is done ===
        if not calib_done:
            img = np.zeros((SCR_H, SCR_W, 3), np.uint8)
            msg1 = "SETUP / CALIBRATION"
            msg2 = "Please face forward for 100 frames"
            msg3 = f"Progress: {calib_count}/{calib_N}"
            cv2.putText(img, msg1, (int(SCR_W*0.5 - 220), int(SCR_H*0.45)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.putText(img, msg2, (int(SCR_W*0.5 - 300), int(SCR_H*0.45+40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,220,255), 2)
            cv2.putText(img, msg3, (int(SCR_W*0.5 - 180), int(SCR_H*0.45+80)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,255,200), 2)
            cv2.putText(img, "Q = Quit", (12, SCR_H-14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            cv2.imshow(win, img)
            k = cv2.waitKey(int(1000/FPS)) & 0xFF
            if k in (ord('q'), ord('Q'), 27):
                break
            # Skip game logic until calibration completes
            continue

        # シールド：片目閉眼連続で起動、目を開けても2フレ維持、HP>0 でのみ動作
        now_t = time.time()

        # シールド展開音は3秒で必ず止める
        if shield_gen_channel is not None and snd_shield_gen is not None:
            if (now_t - shield_fx_start) > 3.0:
                shield_gen_channel.stop()
                shield_gen_channel = None

        # --- Per-eye EAR hysteresis decision (uses baselines captured during setup) ---
        def _apply_hyst(val, prev, th_open, th_close, th_open_fb=0.26, th_close_fb=0.12):
            to = th_open if th_open > 0.0 else th_open_fb
            tc = th_close if th_close > 0.0 else th_close_fb
            if val >= to:
                return 1.0
            elif val <= tc:
                return 0.0
            else:
                return prev

        eyeL_state = _apply_hyst(eyeL_val, eyeL_state, th_open_L, th_close_L)
        eyeR_state = _apply_hyst(eyeR_val, eyeR_state, th_open_R, th_close_R)
        # 両眼閉じているときのみ「閉眼フレーム」とみなす
        eyes_closed_now = (eyeL_state < 0.5) and (eyeR_state < 0.5)

        # Trigger and wink flags from latest UDP packet
        shoot_ev = float(latest.get("shoot", 0.0)) > 0.5
        winkL = float(latest.get("winkL", 0.0)) > 0.5
        winkR = float(latest.get("winkR", 0.0)) > 0.5

        # 目状態の基本判定
        eye_open_now = (eyeL_state >= 0.5) and (eyeR_state >= 0.5)
        one_eye_now  = (not eyes_closed_now) and (not eye_open_now)

        # --- 片目モード切替（連続フレーム条件をPvPに合わせる） ---
        one_eye_consec = one_eye_consec + 1 if one_eye_now else 0
        both_eye_consec = both_eye_consec + 1 if eye_open_now else 0

        if not one_eye_mode:
            if one_eye_consec >= ONE_EYE_ENTER_FRAMES:
                one_eye_mode = True
                both_eye_consec = 0
        else:
            if both_eye_consec >= ONE_EYE_EXIT_FRAMES:
                one_eye_mode = False
                one_eye_consec = 0

        # 両目OPENかつ、片目モードでない（= FPS操作に完全に戻った）ときのみ照準更新
        if eye_open_now and (not one_eye_mode):
            fps_nx = x
            fps_ny = y

        # --- Item selection & use ---
        # Selection: move right when brow_h relative value exceeds threshold, with 1s cooldown.
        # When currently at the rightmost slot (2), the next move wraps to leftmost (0).
        if calib_done and (rel_browh > 0.18) and ((now_t - last_item_move_t) >= 1.0):
            item_selected = (item_selected + 1) % 3
            last_item_move_t = now_t

        # Use: when frown relative value is high and eyes are open (rising edge only)
        item_use_ev = calib_done and (rel_frown > 0.50) and eye_open_now
        if item_use_ev and not prev_item_use:
            # Use (pop) top of stack from currently selected slot if any.
            slot_stack = item_slots[item_selected]
            if slot_stack:
                kind = slot_stack.pop()
                # 0: HP potion  -> recover 1 HP (up to max)
                if kind == 0:
                    player_hp = min(player_hp_max, player_hp + 1)
                # 1: Attack potion -> temporary attack power up (2x for 10 seconds)
                elif kind == 1:
                    attack_buff_mul = 2.0
                    attack_buff_timer = 10.0
                # 2: Shotgun unlock -> enable spread weapon
                elif kind == 2:
                    has_shotgun = True
                # 3: Beam unlock -> enable beam weapon
                elif kind == 3:
                    has_beam = True
            # Blink effect when using item (even if stack is empty)
            item_blink_until = now_t + 2.0
        prev_item_use = item_use_ev

        # --- Update sliding window of closed/open ---
        if len(closed_buf) == WINDOW_N:
            oldest = closed_buf[0]
            closed_sum -= 1 if oldest else 0
        closed_buf.append(eyes_closed_now)
        closed_sum += 1 if eyes_closed_now else 0
        closed_ratio_ok = (len(closed_buf) == WINDOW_N) and ((closed_sum / WINDOW_N) >= WINDOW_RATIO)

        # --- シールドのロジック（50フレーム中 90% 以上が閉眼で起動）---
        if (not shield_on) and closed_ratio_ok:
            shield_on = True
            shield_hp = shield_hp_max
            shield_fx_start = now_t
            if snd_shield_gen is not None:
                shield_gen_channel = snd_shield_gen.play()

        # 維持条件：HP>0 の間だけ維持（開眼では解除しない）
        if shield_on and (shield_hp <= 0):
            shield_on = False
            if shield_gen_channel is not None and snd_shield_gen is not None:
                shield_gen_channel.stop()
                shield_gen_channel = None

        # Real-time weapon decision from mouth shape (aspect/norm_w)
        aspect_raw = float(latest.get("aspect", 0.0))
        normw_raw  = float(latest.get("norm_w", 0.0))
        weapon_id_runtime, aspect_s, normw_s = decide_weapon(aspect_raw, normw_raw)
        # Latch the weapon at the moment the trigger is pressed; keep it while held
        if shoot_ev and not prev_shoot:
            shot_weapon_latched = int(weapon_id_runtime)
        elif not shoot_ev:
            shot_weapon_latched = None

        wid_for_shot = int(weapon_id_runtime) if (shot_weapon_latched is None) else int(shot_weapon_latched)

        if wid_for_shot == 2 and not has_beam:
            wid_for_shot = 0
        if wid_for_shot == 1 and not has_shotgun:
            wid_for_shot = 0

        # 攻撃アップの残り時間を更新
        if attack_buff_timer > 0.0:
            attack_buff_timer -= dt
            if attack_buff_timer <= 0.0:
                attack_buff_timer = 0.0
                attack_buff_mul = 1.0

        # 発射（クールダウン）：自分の弾を生成して照準へ飛ばす（ヒットスキャン→実弾へ変更）
        if shoot_ev and (time.time() - last_shot_time) >= shoot_cooldown:
            bx, by = norm_to_px(fps_nx, fps_ny)  # target point (crosshair in pixels)
            base_dmg = 1.0 + 2.0 * brow
            dmg = base_dmg * attack_buff_mul

            margin_x = int(SCR_W * 0.20)  # 左右から20%内側へ（画面基準）
            margin_y = 80                 # 下端から80px 上（画面基準）

            # --- Convert muzzle positions to WORLD coordinates ---
            # screen → world = add cam_x, cam_y
            muzzles = [
                (float(cam_x + margin_x),             float(cam_y + (SCR_H - margin_y))),  # 左砲口（world座標）
                (float(cam_x + (SCR_W - margin_x)),   float(cam_y + (SCR_H - margin_y))),  # 右砲口（world座標）
            ]

            speed = 2160.0  # [px/s]
            wid = int(wid_for_shot)
            # Play appropriate firing sound once per shot (not per muzzle)
            if wid == 2:
                if snd_beam is not None:
                    snd_beam.play()
            else:
                if snd_shot is not None:
                    snd_shot.play()

            for muzzle_x, muzzle_y in muzzles:
                # Convert FPS crosshair (screen) → world
                target_wx, target_wy = screen_to_world_projected(bx, by, Z_ENEMY, cam_x, cam_y)

                dx = target_wx - muzzle_x
                dy = target_wy - muzzle_y
                n  = math.hypot(dx, dy)
                if n < 1e-6:
                    dx, dy = 1.0, 0.0
                    n = 1.0
                ux, uy = dx / n, dy / n

                if wid == 0:  # BULLET (existing behavior)
                    dist_total = n
                    bullets.append(Bullet(muzzle_x, muzzle_y, ux * speed, uy * speed, dmg,
                                          dist_total=dist_total, r0=6.0, r_min=0.8,
                                          aim_x=target_wx, aim_y=target_wy))
                elif wid == 1:  # SPREAD: angle offsets ±12°, ±6°, 0°
                    for deg in (-12, -6, 0, 6, 12):
                        rad = math.radians(deg)
                        c, s = math.cos(rad), math.sin(rad)
                        ux2, uy2 = (ux*c - uy*s), (ux*s + uy*c)
                        dist_total = n
                        bullets.append(Bullet(muzzle_x, muzzle_y, ux2 * (speed*0.95), uy2 * (speed*0.95), dmg*0.45,
                                              dist_total=dist_total, r0=5.0, r_min=0.8,
                                              aim_x=target_wx, aim_y=target_wy))
                else:  # BEAM: short-lived visual + continuous damage along the beam path
                    lines = [(muzzle_x, muzzle_y, target_wx, target_wy)]
                    beams.append(Beam(lines, ttl=0.12, dmg=dmg*3.0, aim=(target_wx, target_wy), radius=42.0, dps=dmg*6.0))

            last_shot_time = time.time()

        # 物理更新
        dt = DT
        if wave_clear_timer > 0.0:
            wave_clear_timer = max(0.0, wave_clear_timer - dt)
            if game_phase == PHASE_WAVE and wave_clear_timer == 0.0:
                kills_this_wave = 0
        if game_phase == PHASE_WAVE and wave_clear_timer == 0.0 and len(targets) < TARGETS_PER_SPOT:
            spawned = 0
            while len(targets) < TARGETS_PER_SPOT:
                targets.append(spawn_target_for_spot(current_spot_index))
                spawned += 1
            if spawned and snd_enemy_appearance is not None:
                snd_enemy_appearance.play()
        if game_phase == PHASE_BOSS and not boss_spawned and wave_clear_timer <= 0.0:
            try_prepare_boss_face_sprite()
            targets.clear()
            boss_target = Boss(WORLD_W, WORLD_H)
            boss_target.spawn_center = (boss_target.x, boss_target.y)
            boss_target.spawn_spot_index = -1
            if boss_face_sprite is not None:
                boss_target.sprite_override = boss_face_sprite
            targets.append(boss_target)
            boss_spawned = True
            if snd_enemy_appearance is not None:
                snd_enemy_appearance.play()
        # 片目モード中は顔の動きでカメラ(cam_x, cam_y)をパンする
        if one_eye_mode:
            # 中心(0.5,0.5)からのズレを速度ベクトルとして解釈
            dx_norm = float(x) - 0.5
            dy_norm = float(y) - 0.5
            # 水平方向: CAM_PAN_SPEED, 垂直方向: その2倍
            cam_x += dx_norm * CAM_PAN_SPEED * dt
            cam_y += dy_norm * (CAM_PAN_SPEED * 2.0) * dt
        for b in bullets: b.step(dt)
        bullets = [b for b in bullets if 0 <= b.x < WORLD_W and 0 <= b.y < WORLD_H and b.ttl > 0]
        for b in list(beams):
            b.step(dt)
        beams = [b for b in beams if b.ttl > 0]

        # --- Beam continuous hit detection (targets & enemy bullets) ---
        if beams:
            # Apply damage per frame proportional to dt (DPS)
            for b in beams:
                # Hit targets
                for tg in targets:
                    hit = False
                    for (x1,y1,x2,y2) in b.lines:
                        d = dist_point_to_segment(tg.x, tg.y, x1, y1, x2, y2)
                        if d <= (tg.r + b.radius):
                            hit = True
                            break
                    if hit:
                        tg.hp -= (b.dps * dt)
                        # small sparkle effect on hit
                        for _ in range(4):
                            ang = random.uniform(0, 2*math.pi)
                            spd = random.uniform(220, 520)
                            life = random.uniform(0.04, 0.10)
                            particles.append({
                                "x": float(tg.x), "y": float(tg.y),
                                "vx": math.cos(ang)*spd,
                                "vy": math.sin(ang)*spd,
                                "life": life
                            })
                # Hit enemy bullets (parry by beam)
                kept_eb = []
                for eb in enemy_bullets:
                    destroyed = False
                    for (x1,y1,x2,y2) in b.lines:
                        d = dist_point_to_segment(eb.x, eb.y, x1, y1, x2, y2)
                        if d <= (max(8.0, eb.r) + b.radius*0.5):
                            destroyed = True
                            break
                    if destroyed:
                        # effect
                        for _ in range(10):
                            ang = random.uniform(0, 2*math.pi)
                            spd = random.uniform(200, 480)
                            life = random.uniform(0.05, 0.12)
                            particles.append({
                                "x": float(eb.x), "y": float(eb.y),
                                "vx": math.cos(ang)*spd,
                                "vy": math.sin(ang)*spd,
                                "life": life
                            })
                        score += 15
                    else:
                        kept_eb.append(eb)
                enemy_bullets = kept_eb

        for tg in targets:
            tg.step(dt)


        # 敵弾スポーン（だんだん近づいてくる）※ボスフェーズでは停止
        if game_phase != PHASE_BOSS:
            enemy_spawn_timer += dt
            if enemy_spawn_timer >= enemy_spawn_period and len(targets) > 0:
                enemy_spawn_timer = 0.0
                visible_targets = [tg for tg in targets if target_visible_on_screen(tg, cam_x, cam_y)]
                if visible_targets and len(enemy_bullets) < MAX_ENEMY_BULLETS:
                    tg = random.choice(visible_targets)
                    enemy_bullets.append(EnemyBullet(tg.x, tg.y))
                    enemy_spawn_period = random.uniform(0.8, 1.4)

        # プレイヤーのワールド中心座標（カメラ位置＋画面中心）
        player_world_x = cam_x + SCR_W * 0.5
        player_world_y = cam_y + SCR_H * 0.5

        if game_phase == PHASE_BOSS and boss_target is not None:
            boss_visible = target_visible_on_screen(boss_target, cam_x, cam_y)
            boss_target.step_ai(dt, (player_world_x, player_world_y), enemy_bullets, now_t, boss_visible)

        # 敵弾更新＆プレイヤー当たり判定（シールドではなく“打ち返し必須”）
        kept = []
        for eb in enemy_bullets:
            eb.step(dt, player_world_x, player_world_y, SCR_W, SCR_H)
            # ワールド外・寿命切れは捨てる
            if eb.ttl <= 0 or eb.x < -100 or eb.x > WORLD_W + 100 or eb.y < -100 or eb.y > WORLD_H + 100:
                continue
            # 半径が閾値に達したら着弾ダメージ（シールド優先で肩代わり）
            if eb.r >= eb.impact_r:
                hit_sx, hit_sy = world_to_screen(eb.x, eb.y, cam_x, cam_y)
                visible_hit = (-20 <= hit_sx <= SCR_W + 20) and (-20 <= hit_sy <= SCR_H + 20)
                if not visible_hit:
                    continue
                if crack_img is not None:
                    crack_sprite = {
                        "img": crack_img,
                        "x": float(eb.x),
                        "y": float(eb.y),
                        "life": 1.5,
                        "max": 1.5,
                    }
                if snd_hitted is not None:
                    snd_hitted.play()
                if shield_on and shield_hp > 0:
                    # シールド中はシールドHPを1だけ削り、必ずシールド被弾音を再生
                    shield_hp -= 1
                    if snd_shield_frag is not None:
                        snd_shield_frag.play()
                    # シールド被弾エフェクト（シアン寄り）
                    for i in range(18):
                        ang = random.uniform(0, 2*math.pi)
                        spd = random.uniform(200, 400)
                        life = random.uniform(0.08, 0.18)
                        particles.append({
                            "x": float(eb.x), "y": float(eb.y),
                            "vx": math.cos(ang)*spd,
                            "vy": math.sin(ang)*spd,
                            "life": life
                        })
                    # HP 0 ならシールド解除
                    if shield_hp <= 0:
                        shield_on = False
                        if shield_gen_channel is not None and snd_shield_gen is not None:
                            shield_gen_channel.stop()
                            shield_gen_channel = None
                else:
                    # シールドなしならプレイヤーHPにダメージ
                    player_hp -= 1
                    # --- Insert game over check here ---
                    if player_hp <= 0:
                        game_phase = PHASE_CLEAR
                        game_over_flag = True
                        break

                # 着弾エフェクト（共通・爆ぜる）
                for i in range(28):
                    ang = random.uniform(0, 2*math.pi)
                    spd = random.uniform(220, 460)
                    life = random.uniform(0.10, 0.22)
                    particles.append({
                        "x": float(eb.x), "y": float(eb.y),
                        "vx": math.cos(ang)*spd,
                        "vy": math.sin(ang)*spd,
                        "life": life
                    })

                # この弾は着弾で消滅（以降は kept に入れない）
                continue

            # ここまで到達した弾は未着弾なので維持
            kept.append(eb)
        enemy_bullets = kept
        # If game over, break out of main loop
        if game_over_flag:
            break


        # パーティクル更新
        new_particles = []
        for p in particles:
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["life"] -= dt
            if p["life"] > 0:
                new_particles.append(p)
        particles = new_particles

        if crack_sprite is not None:
            crack_sprite["life"] -= dt
            if crack_sprite["life"] <= 0.0:
                crack_sprite = None

        # 当たり判定（円 vs 点/小円）
        hit_idx = []
        for i, tg in enumerate(targets):
            for b in bullets:
                # 照準付近のみ当たり判定：
                # 1) 弾が照準直前（残距離 <= hit_window_px）で、
                # 2) ターゲットが照準中心から aim_radius_px 以内
                rem = max(0.0, b.dist_total - b.dist_travel)
                if rem <= b.hit_window_px:
                    if (tg.x - b.ax)**2 + (tg.y - b.ay)**2 <= (tg.r + b.aim_radius_px)**2:
                        tg.hp -= b.dmg
                        b.ttl = 0.0
            # プレイヤー弾 vs 敵弾（パリィ）
            # (handled after this block for all targets)
            if tg.hp <= 0:
                score += int(100 * tg.hp_max)
                hit_idx.append(i)
        # プレイヤー弾 vs 敵弾（パリィ）
        kept_enemy = []
        for eb in enemy_bullets:
            removed = False
            for b in bullets:
                # 弾同士の衝突（半径を考慮）
                if (b.x - eb.x)**2 + (b.y - eb.y)**2 <= (max(6.0, eb.r + 4.0))**2:
                    # 衝突エフェクト
                    for i in range(12):
                        ang = random.uniform(0, 2*math.pi)
                        spd = random.uniform(240, 520)
                        life = random.uniform(0.06, 0.16)
                        particles.append({
                            "x": float(eb.x), "y": float(eb.y),
                            "vx": math.cos(ang)*spd,
                            "vy": math.sin(ang)*spd,
                            "life": life
                        })
                    score += 30
                    b.ttl = 0.0
                    removed = True
                    break
            if not removed:
                kept_enemy.append(eb)
        enemy_bullets = kept_enemy
        # 倒したら補充
        clear_for_boss = False
        for i in sorted(hit_idx, reverse=True):
            tg = targets[i]
            # Try to spawn a field item at the defeated enemy position, using per-item drop probabilities
            if item_world_sprites:
                # Select the table based on current mob wave phase (wave_index % 4)
                probs = ITEM_DROP_PROBS_WAVE.get(wave_index % 4, ITEM_DROP_PROBS_WAVE[0])
                for k, p in enumerate(probs):
                    if random.random() < p:
                        spr = item_world_sprites[k]
                        if spr is not None:
                            items.append(ItemPickup(tg.x, tg.y, k, spr))
                        break  # Only spawn one item per enemy
            targets.pop(i)
            spawned = False
            if game_phase == PHASE_WAVE:
                kills_this_wave += 1
                if wave_clear_timer <= 0.0:
                    if kills_this_wave >= WAVE_KILL_REQUIREMENT:
                        kills_this_wave = 0
                        wave_clear_timer = WAVE_CLEAR_DISPLAY_SEC
                        wave_index += 1
                        if wave_index >= NUM_WAVES:
                            game_phase = PHASE_BOSS
                            boss_spawned = False
                            clear_for_boss = True
                        else:
                            current_spot_index = (current_spot_index + 1) % len(SPAWN_SPOTS)
                    else:
                        targets.append(spawn_target_for_spot(current_spot_index))
                        spawned = True
                if spawned and snd_enemy_appearance is not None:
                    snd_enemy_appearance.play()
            else:
                if getattr(tg, "is_boss", False):
                    boss_target = None
                    game_phase = PHASE_CLEAR
        if clear_for_boss:
            targets.clear()

        if game_phase == PHASE_WAVE and wave_clear_timer > 0.0 and targets:
            targets.clear()

        # パララックス背景（奥→中→手前）: カメラ移動量で変化させる
        img = np.zeros((SCR_H, SCR_W, 3), np.uint8)
        # カメラ位置でオフセットを決定（横スクロール: cam_x, 縦: cam_y）
        off_far  = int((-cam_x * 0.20) % SCR_W)   # 奥は少しだけ動く
        off_mid  = int((-cam_x * 0.50) % SCR_W)   # 中景は中程度に動く
        off_near = int((-cam_x * 1.00) % SCR_W)   # 手前は自機とほぼ同じ速度で動く
        # Y方向にも少しだけパララックス（奥ほど小さく）
        y0_far  = int(SCR_H * 0.05 - cam_y * 0.05)
        y0_mid  = int(SCR_H * 0.10 - cam_y * 0.10)
        y0_near = int(SCR_H * 0.15 - cam_y * 0.15)
        img = blend_rgba_tiled_x(img, bg_far,  off_far,  yoff=y0_far,  alpha_scale=0.95)
        img = blend_rgba_tiled_x(img, bg_mid,  off_mid,  yoff=y0_mid,  alpha_scale=0.95)
        img = blend_rgba_tiled_x(img, bg_near, off_near, yoff=y0_near, alpha_scale=0.95)

        # current crosshair position in FPS(画面)座標:
        # 両目OPENのときに更新された正規化座標(fps_nx,fps_ny)から決まる
        fps_x, fps_y = norm_to_px(fps_nx, fps_ny)

        # --- Field items: update, pick-up (when cursor overlaps), and draw ---
        pickup_radius = 42.0
        for it in items:
            it.step(dt)
        kept_items = []
        for it in items:
            # world座標上の位置
            ix_w, iy_w = it.get_draw_pos()
            # 画面座標へ変換
            ix_s, iy_s = world_to_screen(ix_w, iy_w, cam_x, cam_y)
            # pick-up if cursor is near and there is a free item slot (判定は画面座標で)
            dx = fps_x - ix_s
            dy = fps_y - iy_s
            dist2 = dx * dx + dy * dy
            picked = False
            if dist2 <= pickup_radius * pickup_radius:
                free_slot = None
                for sidx in range(3):
                    if len(item_slots[sidx]) == 0:
                        free_slot = sidx
                        break
                if free_slot is not None:
                    item_slots[free_slot].append(it.kind)
                    picked = True
            if not picked:
                kept_items.append(it)
            else:
                # small sparkle on pick-up（パーティクルはワールド座標に持っておく）
                for _ in range(10):
                    ang = random.uniform(0, 2 * math.pi)
                    spd = random.uniform(180, 420)
                    life = random.uniform(0.06, 0.14)
                    particles.append({
                        "x": float(ix_w), "y": float(iy_w),
                        "vx": math.cos(ang) * spd,
                        "vy": math.sin(ang) * spd,
                        "life": life
                    })
        items = kept_items

        # draw items (after deciding which ones remain)
        for it in items:
            ix_w, iy_w = it.get_draw_pos()
            spr = it.sprite
            if spr is None:
                continue
            ix_s, iy_s = world_to_screen(ix_w, iy_w, cam_x, cam_y)
            sh, sw = spr.shape[:2]
            sx = int(ix_s) - sw // 2
            sy = int(iy_s) - sh // 2
            img = blend_rgba(img, spr, xoff=sx, yoff=sy, alpha_scale=1.0)

        # ターゲット
        for tg in targets:
            # Ensure boss always uses the latest composited face sprite (even if prepared later)
            if getattr(tg, "is_boss", False) and boss_face_sprite is not None:
                tg.sprite_override = boss_face_sprite
            # ワールド中心座標 → 擬似3D投影（画面座標）。位置だけ奥行きで動かし、サイズは固定。
            tx_s, ty_s, _ = project_to_screen(tg.x, tg.y, tg.z, cam_x, cam_y)
            indicator_pos = compute_edge_indicator(tx_s, ty_s, SCR_W, SCR_H, margin=28)
            if indicator_pos is not None:
                cv2.circle(img, indicator_pos, 6, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
            # 描画用の半径（z によるスケーリングは行わない）
            base_r = max(4.0, float(tg.r))

            # --- shadow (ground-contact ellipse) ---
            # 画面下側ほど影が大きく/濃くなる簡易パース（深度は画面上のYで算出）
            depth = np.clip(ty_s / float(SCR_H), 0.0, 1.0)
            # 影中心はターゲット中心の少し下＆わずかに右下へ（光源=左上想定）
            offx = int(-8 + 10 * depth)
            offy = int(6 + 18 * depth)
            shadow_cx = int(tx_s) + offx
            shadow_cy = int(ty_s + base_r) + offy
            # 影サイズは横長楕円。半径指定（OpenCVはaxes=半径）
            ax = int(max(1.0, base_r * (1.4 + 0.7 * depth)))   # 横方向
            ay = int(max(1.0, base_r * (0.45 + 0.35 * depth))) # 縦方向（扁平）
            alpha = float(np.clip(0.14 + 0.20 * depth, 0.08, 0.35))
            draw_alpha_ellipse(img, (shadow_cx, shadow_cy), (ax, ay), 0.0, (0, 0, 0), alpha, thickness=-1)

            # HPに応じた色（従来ロジックそのまま）
            col = (60, 220, 255) if tg.hp >= tg.hp_max*0.67 else (0, 170, 255) if tg.hp > tg.hp_max*0.34 else (0, 100, 255)

            # Draw alien sprite with alpha (override for boss or random per target)
            sprite_rgba = getattr(tg, "sprite_override", None)
            if sprite_rgba is None and enemy_sprites:
                if not hasattr(tg, "sprite"):
                    tg.sprite = random.choice(enemy_sprites)
                sprite_rgba = tg.sprite
            if sprite_rgba is not None:
                sprite = sprite_rgba
                if sprite.ndim == 3 and sprite.shape[2] == 3:
                    h, w = sprite.shape[:2]
                    a = np.full((h, w, 1), 255, dtype=np.uint8)
                    sprite = np.concatenate([sprite, a], axis=2)
                ex, ey = int(tx_s), int(ty_s)
                if getattr(tg, "sprite_override", None) is not None:
                    ref_h, ref_w = sprite.shape[:2]
                    scale = max(0.25, (base_r * 2.6) / float(max(ref_h, ref_w)))
                    new_w = max(16, int(ref_w * scale))
                    new_h = max(16, int(ref_h * scale))
                else:
                    sprite_size = max(8, int(base_r * 2.2))
                    new_w = sprite_size
                    new_h = sprite_size
                if sprite.shape[0] != new_h or sprite.shape[1] != new_w:
                    sprite = cv2.resize(sprite, (new_w, new_h), interpolation=cv2.INTER_AREA)
                sh, sw = sprite.shape[:2]
                sx, sy = ex - sw//2, ey - sh//2
                if sx < SCR_W and sy < SCR_H and sx + sw > 0 and sy + sh > 0:
                    x1 = max(sx, 0)
                    y1 = max(sy, 0)
                    x2 = min(sx+sw, SCR_W)
                    y2 = min(sy+sh, SCR_H)
                    if x2 > x1 and y2 > y1:
                        sprite_x1 = x1 - sx
                        sprite_y1 = y1 - sy
                        sprite_x2 = sprite_x1 + (x2 - x1)
                        sprite_y2 = sprite_y1 + (y2 - y1)
                        roi = img[y1:y2, x1:x2]
                        spr = sprite[sprite_y1:sprite_y2, sprite_x1:sprite_x2]
                        alpha_spr = spr[:, :, 3:4] / 255.0
                        roi[:] = (roi * (1 - alpha_spr) + spr[:, :, :3] * alpha_spr).astype(np.uint8)
            else:
                cv2.circle(img, (int(tx_s), int(ty_s)), int(base_r), col, 2)

            # HPバー（見かけのサイズに合わせて少し縮小）
            w = int(base_r * 2)
            x0 = int(tx_s - base_r)
            y0 = int(ty_s - base_r - 10)
            hpw = int(w * max(0.0, tg.hp / tg.hp_max))
            cv2.rectangle(img, (x0, y0), (x0+w, y0+6), (40,40,40), -1)
            cv2.rectangle(img, (x0, y0), (x0+hpw, y0+6), (0,220,0), -1)

        # 弾（自機）
        for b in bullets:
            bx_s, by_s = world_to_screen(b.x, b.y, cam_x, cam_y)
            bx_i, by_i = int(bx_s), int(by_s)
            rad = int(max(1, round(b.r)))
            cv2.circle(img, (bx_i, by_i), rad, (255,255,255), -1)
            draw_alpha_circle(img, (bx_i, by_i), rad+5, (0, 180, 255), 0.32, thickness=2)

        # BEAM rendering（遠いほど影を強く/大きく＋深度レイヤーに応じて投影）
        for b in beams:
            for (x1, y1, x2, y2) in b.lines:
                # ワールド座標→擬似3D投影（同一zレイヤー上にビームを描く）
                sx1_w, sy1_w, s1 = project_to_screen(x1, y1, b.z, cam_x, cam_y)
                sx2_w, sy2_w, s2 = project_to_screen(x2, y2, b.z, cam_x, cam_y)
                avg_scale = 0.5 * (s1 + s2)

                # 距離を正規化（画面対角でスケール）
                L = math.hypot(sx2_w - sx1_w, sy2_w - sy1_w)
                diag = math.hypot(SCR_W, SCR_H)
                depth = float(np.clip(L / max(1.0, diag), 0.0, 1.0))
                # 影オフセット/太さ/濃さを距離で決定（遠いほど強調）
                off    = 4.0 + 8.0 * depth   # ピクセル
                th_s   = (6.0 + 10.0 * depth) * (0.6 + 0.8 * avg_scale)  # 深度スケールも反映
                a_s    = 0.15 + 0.35 * depth # 影のアルファ
                # 影の方向：やや右下（光源=左上）
                sx1 = sx1_w + off * 0.3
                sy1 = sy1_w + off * 0.6
                sx2 = sx2_w + off * 0.3
                sy2 = sy2_w + off * 0.6
                # 影（ディープブルー寄り）
                draw_alpha_line(img, (sx1, sy1), (sx2, sy2), (30, 50, 80), a_s, thickness=int(th_s))
                # 本体（外側のグロー→芯）
                # --- Variable thickness beam: thick (near) → thin (far) ---
                segs = 16  # number of segments for the beam
                thick_start = 12.0 * (0.6 + 0.8 * avg_scale)
                thick_end = 3.0 * (0.6 + 0.8 * avg_scale)
                for i in range(segs):
                    t0 = i / segs
                    t1 = (i + 1) / segs
                    px0 = sx1_w + (sx2_w - sx1_w) * t0
                    py0 = sy1_w + (sy2_w - sy1_w) * t0
                    px1 = sx1_w + (sx2_w - sx1_w) * t1
                    py1 = sy1_w + (sy2_w - sy1_w) * t1
                    thickness = thick_start * (1 - t0) + thick_end * t0
                    # --- Beam color intensity fades with t0 (farther = darker) ---
                    # brightness decreases linearly with t0 (min 0.6)
                    brightness = max(0.6, 1.0 - 0.4 * t0)
                    # Outer glow color
                    glow_col = np.clip(np.array([255, 220, 150]) * brightness, 0, 255).astype(np.uint8)
                    # Core color
                    core_col = np.clip(np.array([255, 255, 255]) * brightness, 0, 255).astype(np.uint8)
                    # outer glow
                    cv2.line(
                        img,
                        (int(px0), int(py0)),
                        (int(px1), int(py1)),
                        (int(glow_col[0]), int(glow_col[1]), int(glow_col[2])),
                        int(max(2, round(thickness))),
                        cv2.LINE_AA,
                    )
                    # core
                    core_thick = max(2, round(thickness * 0.33))
                    cv2.line(
                        img,
                        (int(px0), int(py0)),
                        (int(px1), int(py1)),
                        (int(core_col[0]), int(core_col[1]), int(core_col[2])),
                        int(core_thick),
                        cv2.LINE_AA,
                    )

        # 敵弾（深度つき）
        for eb in enemy_bullets:
            ex_s, ey_s, depth_scale = project_to_screen(eb.x, eb.y, eb.z, cam_x, cam_y)
            r = int(max(1, eb.r * depth_scale))
            # 中心を明るく、外縁を濃くして近接感
            cv2.circle(img, (int(ex_s), int(ey_s)), r, (0, 90, 255), -1)
            cv2.circle(img, (int(ex_s), int(ey_s)), r+2, (0, 60, 200), 2)
            # 近づくほど薄いグロー（impact_r 比はそのまま、見かけ半径だけ深度スケール）
            glow_alpha = min(0.5, (eb.r / eb.impact_r) * 0.5)
            draw_alpha_circle(img, (int(ex_s), int(ey_s)), r+6, (0, 120, 255), glow_alpha, thickness=2)
        if crack_sprite is not None:
            a = max(0.0, min(1.0, crack_sprite["life"] / crack_sprite["max"]))
            spr = crack_sprite["img"]
            if spr is not None:
                sh, sw = spr.shape[:2]
                cx_s, cy_s = world_to_screen(crack_sprite["x"], crack_sprite["y"], cam_x, cam_y)
                sx = int(cx_s) - sw // 2
                sy = int(cy_s) - sh // 2
                x1, y1 = max(sx, 0), max(sy, 0)
                x2, y2 = min(sx + sw, SCR_W), min(sy + sh, SCR_H)
                if x1 < x2 and y1 < y2:
                    roi = img[y1:y2, x1:x2]
                    crop = spr[(y1 - sy):(y2 - sy), (x1 - sx):(x2 - sx)]
                    if crop.shape[2] == 4:
                        alpha = (crop[:, :, 3:4].astype(np.float32) / 255.0) * a
                        roi[:] = (roi.astype(np.float32) * (1 - alpha) + crop[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
        # 照準（スプライト）
        crosshair = cv2.imread("img/crosshair_scifi_64.png", cv2.IMREAD_UNCHANGED)
        if crosshair is not None and crosshair.shape[2] == 4:
            ch_h, ch_w = crosshair.shape[:2]
            sx, sy = fps_x - ch_w // 2, fps_y - ch_h // 2
            x1, y1 = max(sx, 0), max(sy, 0)
            x2, y2 = min(sx + ch_w, SCR_W), min(sy + ch_h, SCR_H)
            if x1 < x2 and y1 < y2:
                roi = img[y1:y2, x1:x2]
                ch_crop = crosshair[(y1 - sy):(y2 - sy), (x1 - sx):(x2 - sx)]
                alpha = ch_crop[:, :, 3:4] / 255.0
                roi[:] = (roi * (1 - alpha) + ch_crop[:, :, :3] * alpha).astype(np.uint8)
        else:
            # fallback if missing
            cv2.circle(img, (fps_x, fps_y), 10, (0, 255, 255), 2)

        # シールド表示（画面全体：ATフィールド風）
        if shield_on:
            tfx = max(0.0, time.time() - shield_fx_start)
            # 1) 起動直後フラッシュ（画面全体を軽くオレンジでフラッシュ）
            if tfx <= 0.15:
                overlay = tint_overlay.copy()
                cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

            # 2) 六角グリッド + 弱いオレンジ色のベール
            #    六角は軽いパルスで強弱（2Hz程度）
            pulse = 0.30 + 0.20 * math.sin(2 * math.pi * 2.0 * t)  # 0.10～0.50くらい
            cv2.addWeighted(hex_overlay, float(np.clip(pulse, 0.10, 0.55)), img, 1.0, 0, img)
            cv2.addWeighted(tint_overlay, 0.12, img, 0.88, 0, img)

            # 3) 同心円の干渉縞っぽいリング（中心基準）
            cx0, cy0 = SCR_W//2, SCR_H//2
            base_r = 60
            spacing = 48
            for k in range(1, 10):
                rr = base_r + k * spacing
                a = 0.10 if k % 2 == 0 else 0.05
                draw_alpha_circle(img, (cx0, cy0), rr, (0, 165, 255), a, thickness=2)

        # Minimal HUD elements (HP, Shield, Weapon)
        cv2.putText(img, f"HP: {player_hp}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        cv2.putText(img, f"Shield HP: {shield_hp}/{shield_hp_max}", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 255), 2)
        cv2.putText(img, f"Shield: {'ON' if shield_on else 'OFF'}", (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 240, 255), 2)
        cv2.putText(img, f"Weapon: {['BULLET','SPREAD','BEAM'][weapon_last]}", (12, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 180), 2)
        # Item bar (3 square slots at bottom-left)
        slot_size = 48
        slot_gap = 8
        base_x = 12
        base_y = SCR_H - 80
        for idx in range(3):
            x1 = base_x + idx * (slot_size + slot_gap)
            y1 = base_y
            x2 = x1 + slot_size
            y2 = y1 + slot_size
            col = (80, 80, 80)
            thickness = 1
            if idx == item_selected:
                # Blink highlight for a short time after item use
                if time.time() < item_blink_until:
                    col = (255, 255, 255)
                    thickness = 3
                else:
                    col = (200, 255, 200)
                    thickness = 2
            cv2.rectangle(img, (x1, y1), (x2, y2), col, thickness)
            # draw item icon in the slot if present
            if item_slots[idx]:
                kind = item_slots[idx][-1]
                if 0 <= kind < len(item_slot_icons):
                    icon = item_slot_icons[kind]
                    if icon is not None:
                        ih, iw = icon.shape[:2]
                        cx_s = (x1 + x2) // 2
                        cy_s = (y1 + y2) // 2
                        sx = cx_s - iw // 2
                        sy = cy_s - ih // 2
                        img = blend_rgba(img, icon, xoff=sx, yoff=sy, alpha_scale=1.0)
        cv2.putText(img, "Mouth=Shoot(at cursor/parry)  Brow=Power  Shield(AT-field)=One-eye 90% closed(50f)   Q=Quit",
                    (12, SCR_H-14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 2)

        if wave_clear_timer > 0.0:
            msg = "WAVE CLEARED - NEXT SPOT"
            text_size, _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            tx = (SCR_W - text_size[0]) // 2
            ty = SCR_H // 2
            cv2.putText(img, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow(win, img)
        prev_shoot = shoot_ev
        k = cv2.waitKey(int(1000/FPS)) & 0xFF


        # ワールドの限界: 画面サイズの3倍 WORLD_W x WORLD_H の範囲にクランプ
        # cam_x, cam_y は「画面左上」のワールド座標なので、
        # 0 <= cam_x <= WORLD_W - SCR_W, 0 <= cam_y <= WORLD_H - SCR_H
        max_cam_x = max(0.0, float(WORLD_W - SCR_W))
        max_cam_y = max(0.0, float(WORLD_H - SCR_H))
        if cam_x < 0.0:
            cam_x = 0.0
        elif cam_x > max_cam_x:
            cam_x = max_cam_x
        if cam_y < 0.0:
            cam_y = 0.0
        elif cam_y > max_cam_y:
            cam_y = max_cam_y

        if k in (ord('q'), ord('Q'), 27):
            return_to_home = True
            break

    # --- Show GAME OVER or CLEAR screen if needed ---
    if game_over_flag:
        img = np.zeros((SCR_H, SCR_W, 3), np.uint8)
        msg = "GAME OVER"
        text_size, _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
        tx = (SCR_W - text_size[0]) // 2
        ty = SCR_H // 2
        cv2.putText(img, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,255), 4)
        cv2.imshow(win, img)
        cv2.waitKey(2000)
    elif game_phase == PHASE_CLEAR:
        img = np.zeros((SCR_H, SCR_W, 3), np.uint8)
        msg = "STAGE CLEARED!"
        text_size, _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
        tx = (SCR_W - text_size[0]) // 2
        ty = SCR_H // 2
        cv2.putText(img, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 4)
        cv2.imshow(win, img)
        cv2.waitKey(2000)
    cv2.destroyAllWindows()
    if return_to_home and os.environ.get("SKIP_HOME_RELAUNCH") != "1":
        relaunch_home_screen()

if __name__ == "__main__":
    main()
