#!/usr/bin/env python3
import argparse
import os
import socket, struct, threading, time, math, random
import subprocess
import sys
from pathlib import Path
import cv2
import pygame
import numpy as np
from collections import deque
import src.net as net

from src.face_sprite import capture_opponent_frame
from src.game_objects import (
    PlayerBullet,
    EnemyBullet,
    RemotePlayerBullet,
    Target,
    OpponentPlayer,
    ItemPickup,
    make_crack,
)
from src.render_utils import (
    draw_alpha_circle,
    draw_alpha_ellipse,
    draw_alpha_polyline,
    blend_rgba,
    blend_rgba_tiled_x,
    build_hex_overlay,
    world_to_screen,
    screen_to_world,
    project_to_screen,
    dist_point_to_segment,
)

from src.bullet_render import draw_player_bullet, draw_remote_player_bullet

SCR_W, SCR_H = 960, 540        # ゲーム画面サイズ（自由に変更）
WORLD_W, WORLD_H = SCR_W * 4, SCR_H * 3  # ワールド全体のサイズ
PLAYER1_CAM_START = (0.0, 0.0)
PLAYER2_CAM_START = (max(0.0, float(WORLD_W - SCR_W)), max(0.0, float(WORLD_H - SCR_H)))
# スポーンポイントはワールド座標で管理し、常に2地点がアクティブになる
ENEMY_SPAWN_POINTS = [
    {"pos": (WORLD_W * 0.20, WORLD_H * 0.25)},
    {"pos": (WORLD_W * 0.65, WORLD_H * 0.20)},
    {"pos": (WORLD_W * 0.30, WORLD_H * 0.70)},
    {"pos": (WORLD_W * 0.75, WORLD_H * 0.65)},
]
ACTIVE_SPAWN_LIMIT = 2  # 常時アクティブなスポーンポイント件数（敵数ではない）
# 各スポーンが同時に保持できる敵の最大数（1スポーン=5体）
MAX_ACTIVE_PER_SPAWN = 4
# 常時アクティブ扱いするスポーン（先頭から ACTIVE_SPAWN_LIMIT 個を採用）
ACTIVE_SPAWN_IDS = list(range(min(ACTIVE_SPAWN_LIMIT, len(ENEMY_SPAWN_POINTS))))

# --- P2P face preview error flag (for one-shot error logging) ---
P2P_FACE_PREVIEW_ERROR_ONCE = False

# 顔アイコンを描画する際の統一サイズ（受信/送信問わず、このピクセル数で表示）
FACE_SPRITE_DRAW_SIZE = 128

CAM_STEP = 20.0                          # 矢印キー1回あたりのカメラ移動量[px]
CAM_PAN_SPEED = 1500.0                    # 片目モード時のカメラ移動速度[px/s]（基準：水平方向）

# 擬似3D用のZレイヤー（奥行き）
Z_PLAYER   = 0.0   # 手前（自分）
Z_ENEMY    = 0.5   # 中間（敵モンスター）
Z_OPPONENT = 1.0   # 奥（対戦相手プレイヤー）
FPS = 60
DT = 1.0 / FPS
SHIELD_GRACE_FRAMES = 2  # 目を開けてもこのフレーム数だけシールド維持

GAME_PHASE_WAIT_CALIB_SYNC = "WAIT_CALIB_SYNC"
GAME_PHASE_CALIBRATING = "CALIBRATING"
GAME_PHASE_PLAY = "PLAY"

LAUNCHER_SCRIPT = Path(__file__).resolve().parent / "launcher_pygame.py"


def relaunch_home_screen() -> None:
    """Re-open the pygame launcher so players can return to the home screen."""
    if not LAUNCHER_SCRIPT.exists():
        print(f"[launcher] Missing launcher script: {LAUNCHER_SCRIPT}")
        return
    if os.environ.get("SKIP_HOME_RELAUNCH") == "1":
        return
    try:
        subprocess.Popen([sys.executable, str(LAUNCHER_SCRIPT)])
    except Exception as exc:  # pragma: no cover - best-effort logging
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

# --- Player bullet power from mouth aspect ---
# aspect = mouth_h/mouth_w (縦に開くほど↑)
# 口の開きに応じて「ダメージ↑ / 速度↓」になるように設計する。
ASPECT_MIN = 0.20      # これより小さい aspect は 0.20 扱い（ゼロ除算防止）
ASPECT_MAX = 1.20      # これ以上は「最大開き」とみなす

DAMAGE_CONST = 10.0    # aspect=1.0 のときの基準ダメージ
DAMAGE_MIN  = 1        # ダメージの下限（整数）
DAMAGE_MAX  = 20       # ダメージの上限（整数）

SPEED_CONST = 400    # aspect=1.0 のときの基準速度 [px/s]
SPEED_MIN   = 200      # 速度の下限 [px/s]
SPEED_MAX   = 600    # 速度の上限 [px/s]

# --- 片目モード切替の連続フレーム閾値 ---
ONE_EYE_ENTER_FRAMES = 3        # 両目→片目判定（3連続フレーム）
ONE_EYE_EXIT_FRAMES  = 5     # 片目→両目判定（10連続フレーム）

# --- Player leveling thresholds: level[i] = XP needed to reach level i+2 ---
PLAYER_LEVEL_THRESHOLDS = [2, 4, 6, 8, 10, 12]


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


def get_cam_start(player_id: int) -> tuple[float, float]:
    return PLAYER2_CAM_START if player_id == 2 else PLAYER1_CAM_START


def get_player_center_for_id(player_id: int) -> tuple[float, float]:
    camx, camy = get_cam_start(player_id)
    return camx + SCR_W * 0.5, camy + SCR_H * 0.5


def get_next_level_xp(level: int) -> int | None:
    """Return XP needed to reach the next level, or None if at max."""
    idx = int(level) - 1
    if idx < 0 or idx >= len(PLAYER_LEVEL_THRESHOLDS):
        return None
    return PLAYER_LEVEL_THRESHOLDS[idx]


class EnemySyncTCP:
    """Small TCP helper used to mirror enemy spawns/kill counts between peers."""

    def __init__(self, is_authority: bool):
        self.is_authority = bool(is_authority)
        self.sock: socket.socket | None = None
        self.lock = threading.Lock()
        self.queue_lock = threading.Lock()
        self.queue: deque[tuple[str, list[str]]] = deque()
        self.connected = False

    def start(
        self,
        listen_ip: str,
        listen_port: int | None,
        connect_host: str | None,
        connect_port: int | None,
    ) -> None:
        if self.is_authority:
            if listen_port is None:
                raise RuntimeError("authority requires a TCP listen port")
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((listen_ip, listen_port))
            server.listen(1)
            print(f"[ENEMY-TCP] waiting for peer on {listen_ip}:{listen_port} (authority)")
            conn, addr = server.accept()
            print(f"[ENEMY-TCP] peer connected from {addr}")
            server.close()
            self.sock = conn
        else:
            if not connect_host or connect_port is None:
                raise RuntimeError("non-authority requires --peer host:port")
            while True:
                try:
                    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    conn.connect((connect_host, connect_port))
                    self.sock = conn
                    print(f"[ENEMY-TCP] connected to authority {connect_host}:{connect_port}")
                    break
                except OSError as e:
                    print(f"[ENEMY-TCP] connect failed ({e}); retrying in 1s...")
                    time.sleep(1.0)
        self.connected = True
        threading.Thread(target=self._reader_loop, daemon=True).start()

    def _reader_loop(self) -> None:
        if self.sock is None:
            return
        sock = self.sock
        buf = b""
        try:
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="ignore").strip()
                    if text:
                        self._enqueue_line(text)
        except OSError:
            pass
        finally:
            self.connected = False

    def _enqueue_line(self, text: str) -> None:
        parts = text.split()
        if not parts:
            return
        evt = parts[0].upper()
        args = parts[1:]
        if evt == "KILL":
            if len(args) < 5:
                return
            enemy_id = int(args[0])
            spawn_id = int(args[1])
            total_kills = int(args[2])
            killer_id = int(args[3])
            confirmed = args[4] == "1"
            self.on_kill_received(enemy_id, spawn_id, total_kills, killer_id, confirmed)
        else:
            with self.queue_lock:
                self.queue.append((evt, args))

    def drain_events(self) -> list[tuple[str, list[str]]]:
        out: list[tuple[str, list[str]]] = []
        with self.queue_lock:
            while self.queue:
                out.append(self.queue.popleft())
        return out

    def send_line(self, text: str) -> None:
        if not self.connected or self.sock is None:
            return
        data = (text.strip() + "\n").encode("utf-8")
        with self.lock:
            try:
                self.sock.sendall(data)
            except OSError:
                self.connected = False

    def send_spawn(self, spawn_id: int, spawn_seq: int, enemy_uid: int) -> None:
        self.send_line(f"SPAWN {spawn_id} {spawn_seq} {enemy_uid}")

    def send_kill(
        self,
        enemy_uid: int,
        spawn_id: int,
        total_kills: int | None = None,
        killer_id: int = 0,
        confirmed: bool = False,
    ) -> None:
        total = -1 if total_kills is None else int(total_kills)
        flag = "1" if confirmed else "0"
        self.send_line(f"KILL {enemy_uid} {spawn_id} {total} {killer_id} {flag}")

    def on_kill_received(
        self,
        enemy_uid: int,
        spawn_id: int,
        total_kills: int,
        killer_id: int,
        confirmed: bool,
    ) -> None:
        with self.queue_lock:
            self.queue.append(
                (
                    "KILL",
                    [
                        str(enemy_uid),
                        str(spawn_id),
                        str(total_kills),
                        str(killer_id),
                        "1" if confirmed else "0",
                    ],
                )
            )


def compute_bullet_params_from_aspect(aspect_raw: float):
    """口の縦横比(aspect)から、プレイヤー弾の段階的なダメージ/速度を決定する。"""
    a = float(np.clip(aspect_raw, 0.0, 1.5))
    if a < 0.35:
        tier = 1
    elif a < 0.70:
        tier = 2
    else:
        tier = 3

    damage_i = tier
    speed_table = {1: 600, 2: 500, 3: 400}
    speed_i = speed_table.get(tier, 500)
    return damage_i, speed_i, a



# --- HUD helper function ---
def draw_hud(
    img,
    score: int,
    player_level: int,
    player_xp: int,
    xp_to_next: int | None,
    player_hp: int,
    player_hp_max: int,
    shield_hp: float,
    shield_hp_max: float,
    shield_on: bool,
    calib_done: bool,
    calib_count: int,
    calib_N: int,
    calib_cx: float,
    calib_cy: float,
    gain_xy: float,
    th_open_L: float,
    th_close_L: float,
    th_open_R: float,
    th_close_R: float,
    fps_x: int,
    fps_y: int,
    player_world_x: float,
    player_world_y: float,
    rel_smile: float,
    rel_frown: float,
    rel_browh: float,
) -> None:
    """
    ゲーム画面右上のデバッグ表示、左上のステータス、
    画面下部のアイテムスロットなど HUD 一式をまとめて描画する。
    """
    # Score / HP / Shield basic HUD (left-top)
    cv2.putText(
        img,
        f"Score: {score}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 255, 200),
        2,
    )
    if xp_to_next is None:
        level_text = f"Lv: {player_level} (MAX)"
    else:
        level_text = f"Lv: {player_level}  EXP {player_xp}/{xp_to_next}"
    cv2.putText(
        img,
        level_text,
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 180),
        2,
    )
    cv2.putText(
        img,
        f"HP: {player_hp}/{player_hp_max}",
        (12, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 220, 255),
        2,
    )
    cv2.putText(
        img,
        f"Shield HP: {shield_hp}/{shield_hp_max}",
        (12, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 255, 255),
        2,
    )
    cv2.putText(
        img,
        f"Shield: {'ON' if shield_on else 'OFF'}",
        (12, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (160, 240, 255),
        2,
    )

    # FPS座標とワールド座標のデバッグ表示
    cv2.putText(
        img,
        f"FPS=({fps_x},{fps_y}) PlayerWorld=({player_world_x:.1f},{player_world_y:.1f})",
        (12, 222),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 220, 200),
        2,
    )

    cv2.putText(
        img,
        "Mouth=Shoot(at cursor/parry)  Brow=Power  Shield(AT-field)=One-eye 90% closed(50f)   Q=Quit",
        (12, SCR_H - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        2,
    )





def main():
    # --- Argument parsing for P2P and opponent face ---
    parser = argparse.ArgumentParser(description="PVP shooter with gaze control")
    parser.add_argument('--peer', type=str, default=None, help='peer host:port for P2P (e.g., 192.168.0.5:6001)')
    parser.add_argument('--listen-ip', type=str, default="0.0.0.0", help='listening IP address')
    parser.add_argument('--listen-port', type=int, default=6000, help='listening UDP port')
    parser.add_argument('--gaze-listen-ip', type=str, default="0.0.0.0", help='IP address to bind for gaze UDP input')
    parser.add_argument('--gaze-port', type=int, default=5005, help='UDP port to listen for gaze data (from detect_face)')
    parser.add_argument('--capture-opponent', action='store_true',
                        help='capture opponent face from local camera at startup')
    parser.add_argument('--player-id', type=int, choices=(1, 2), default=None,
                        help='Player slot (auto-detected in --local mode when omitted)')
    args = parser.parse_args()
    return_to_home = False

    peer_host = None
    peer_port = None
    if args.peer is not None:
        peer_host, *peer_port_part = args.peer.split(':')
        if peer_host == "":
            peer_host = "127.0.0.1"
        if peer_port_part and peer_port_part[0].isdigit():
            peer_port = int(peer_port_part[0])
        else:
            peer_port = args.listen_port

    env_local_idx = (
        os.environ.get("PVP_LOCAL_INDEX")
        or os.environ.get("PVP_LOCAL_SLOT")
        or os.environ.get("LOCAL_PVP_INDEX")
    )

    if env_local_idx in ("1", "2"):
        player_id = int(env_local_idx)
    elif args.player_id is not None:
        player_id = args.player_id
    elif peer_port is not None:
        player_id = 1 if args.listen_port <= peer_port else 2
    else:
        player_id = 1

    if player_id == 2 and peer_host is None:
        raise SystemExit("player 2 requires --peer host:port to connect to player 1")

    is_authority = (player_id == 1)
    remote_player_id = 2 if player_id == 1 else 1
    remote_spawn_x, remote_spawn_y = get_player_center_for_id(remote_player_id)

    # --- P2P setup ---
    p2p_sock = None
    p2p_peer_addr = None
    p2p_enabled = False
    remote_state = {"x": remote_spawn_x, "y": remote_spawn_y, "hp": 0.0}
    p2p_lock = threading.Lock()
    p2p_face_sock = None
    p2p_face_peer_addr = None
    if peer_host is not None:
        if peer_port is None:
            peer_port = args.listen_port
        try:
            p2p_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            p2p_sock.settimeout(1.0)
            try:
                p2p_sock.bind((args.listen_ip, args.listen_port))
            except Exception as e:
                print(f"[P2P] WARNING: failed to bind {args.listen_ip}:{args.listen_port}: {e}")
            p2p_peer_addr = (peer_host, peer_port)
            p2p_enabled = True
            threading.Thread(target=net.p2p_recv_loop, args=(p2p_sock, remote_state, p2p_lock), daemon=True).start()
            print(f"[P2P] listening on {args.listen_ip}:{args.listen_port}, peer={peer_host}:{peer_port}")

            try:
                p2p_face_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                p2p_face_sock.settimeout(1.0)
                face_port_offset = 100
                face_listen_port = args.listen_port + face_port_offset
                try:
                    p2p_face_sock.bind((args.listen_ip, face_listen_port))
                except Exception as e:
                    print(f"[P2P_FACE] WARNING: failed to bind {args.listen_ip}:{face_listen_port}: {e}")
                p2p_face_peer_addr = (peer_host, peer_port + face_port_offset)
                threading.Thread(target=net.face_recv_loop, args=(p2p_face_sock,), daemon=True).start()
                print(f"[P2P_FACE] listening on {args.listen_ip}:{face_listen_port}, peer={peer_host}:{peer_port + face_port_offset}")
            except Exception as e:
                print(f"[P2P_FACE] setup failed: {e}")
        except Exception as e:
            print(f"[P2P] setup failed: {e}")

    if is_authority:
        tcp_listen_port = args.listen_port
        tcp_connect_host = None
        tcp_connect_port = None
    else:
        tcp_listen_port = None
        tcp_connect_host = peer_host
        tcp_connect_port = peer_port
    enemy_tcp = EnemySyncTCP(is_authority=is_authority)
    pending_enemy_events: deque[tuple[str, list[str]]] = deque()

    game_phase = GAME_PHASE_WAIT_CALIB_SYNC
    local_calib_ready = False
    remote_calib_ready = False
    calib_active = False
    calib_start_sent = False

    def _start_enemy_tcp():
        try:
            enemy_tcp.start(args.listen_ip, tcp_listen_port, tcp_connect_host, tcp_connect_port)
        except Exception as exc:
            print(f"[ENEMY-TCP] setup failed: {exc}")
    threading.Thread(target=_start_enemy_tcp, daemon=True).start()

    # --- Initialize sound (pygame.mixer) ---
    pygame.mixer.init()
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
    th = threading.Thread(target=net.udp_loop, args=(args.gaze_listen_ip, args.gaze_port), daemon=True)
    th.start()

    # ワールド
    bullets = []
    targets = []
    # 将来のPVP対戦相手（いまはダミーで「現在の画面内」に1体配置）
    # cam_x=0, cam_y=0 でスタートなので、スクリーン座標と同じ値をそのままワールド座標として使う
    opponent = OpponentPlayer(remote_spawn_x, remote_spawn_y)
    score = 0
    player_level = 1
    player_xp = 0
    last_shot_time = 0.0
    shoot_cooldown = 0.12
    shield_until = 0.0
    shield_fx_start = 0.0
    particles = []  # list of dicts: {x,y,vx,vy,life}
    enemy_bullets = []
    enemy_spawn_timer = 0.0
    enemy_spawn_period = 1.0   # 平均1秒に1発
    remote_player_bullets = []
    # --- Crack decals and sprites ---
    cracks = []
    crack_sprites = []
    # --- Field item pickups (dropped from defeated enemies) ---
    items = []
    ITEM_DROP_PROB = 0.05  # 5% drop chance per defeated enemy

    player_hp_max = 30
    player_hp = player_hp_max
    remote_state["hp"] = float(player_hp)
    attack_bonus = 0
    shield_hp_max = 2     # シールドHPの上限
    shield_hp = shield_hp_max
    shield_on = False
    eye_closed_count = 0
    shield_grace = 0

    # --- XY calibration over first N frames ---
    calib_N = 100
    calib_sumx = 0.0
    calib_sumy = 0.0
    calib_count = 0
    calib_done = False
    calib_cx = 0.5
    calib_cy = 0.5
    gain_xy = 5.0  # magnify around calibrated center

    def _spawn_cap_for_id(spawn_id: int) -> int:
        rng = random.Random(spawn_id * 8191 + 17)
        return rng.randint(6, 10)

    # 各スポーンポイントの撃破上限は6～10体の乱数（シード固定で全員一致）
    spawn_states = [
        {"kills": 0, "cap": _spawn_cap_for_id(idx), "active": [], "spawns": 0, "done": False}
        for idx, _ in enumerate(ENEMY_SPAWN_POINTS)
    ]
    enemy_lookup: dict[int, tuple[int, Target]] = {}
    next_enemy_uid = 1

    def alloc_enemy_uid() -> int:
        nonlocal next_enemy_uid
        uid = next_enemy_uid
        next_enemy_uid += 1
        return uid

    def roll_drop_kind() -> int:
        if not item_world_sprites:
            return -1
        if random.random() < ITEM_DROP_PROB:
            return random.randint(0, len(item_world_sprites) - 1)
        return -1

    def apply_item_effect(kind: int) -> None:
        nonlocal player_hp, attack_bonus
        if kind == 0:
            player_hp = min(player_hp_max, player_hp + 2)
        elif kind == 1:
            attack_bonus += 3

    def add_player_exp(points: int) -> None:
        nonlocal player_xp, player_level
        player_xp += points
        while True:
            xp_need = get_next_level_xp(player_level)
            if xp_need is None or player_xp < xp_need:
                break
            player_xp -= xp_need
            player_level += 1

    def grant_kill_rewards(killer_id: int, target_obj: Target | None) -> None:
        nonlocal score, items
        if killer_id != player_id or target_obj is None:
            return
        reward_score = int(100 * getattr(target_obj, "hp_max", 1))
        score += reward_score
        add_player_exp(1)
        drop_kind = roll_drop_kind()
        if 0 <= drop_kind < len(item_world_sprites):
            spr = item_world_sprites[drop_kind]
            if spr is not None:
                items.append(ItemPickup(float(target_obj.x), float(target_obj.y), drop_kind, spr))

    def detach_enemy(enemy_uid: int) -> tuple[int | None, Target | None]:
        info = enemy_lookup.pop(enemy_uid, None)
        if info is None:
            return None, None
        spawn_id, tgt = info
        state = spawn_states[spawn_id]
        state["active"] = [entry for entry in state["active"] if entry["uid"] != enemy_uid]
        tgt._dead = True
        return spawn_id, tgt

    def spawn_enemy_from_point(
        spawn_id: int,
        spawn_seq: int | None = None,
        enemy_uid: int | None = None,
        notify: bool = False,
    ) -> Target | None:
        state = spawn_states[spawn_id]
        if state["done"] or state["kills"] >= state["cap"]:
            return None
        if len(state["active"]) >= MAX_ACTIVE_PER_SPAWN:
            return None
        if spawn_seq is None:
            spawn_seq = state["spawns"]
        seed = spawn_id * 1000 + spawn_seq
        origin = ENEMY_SPAWN_POINTS[spawn_id]["pos"]
        tgt = Target(spawn_point_id=spawn_id, seed=seed, origin=origin)
        tgt.spawn_seq = spawn_seq
        tgt.pending_kill = False
        if enemy_uid is None:
            enemy_uid = alloc_enemy_uid()
        tgt.enemy_uid = enemy_uid
        state["active"].append({"uid": enemy_uid, "target": tgt})
        enemy_lookup[enemy_uid] = (spawn_id, tgt)
        state["spawns"] = max(state["spawns"], spawn_seq + 1)
        targets.append(tgt)
        if snd_enemy_appearance is not None and calib_done:
            try:
                snd_enemy_appearance.play()
            except Exception:
                pass
        if notify and enemy_tcp.connected:
            enemy_tcp.send_spawn(spawn_id, spawn_seq, enemy_uid)
        return tgt

    def fill_spawn_capacity_for(spawn_id: int) -> None:
        if not is_authority:
            return
        state = spawn_states[spawn_id]
        if state["done"] or state["kills"] >= state["cap"]:
            return
        while len(state["active"]) < MAX_ACTIVE_PER_SPAWN and state["kills"] < state["cap"]:
            spawn_enemy_from_point(spawn_id, notify=True)

    def fill_active_spawns() -> None:
        if not is_authority:
            return
        for sid in ACTIVE_SPAWN_IDS:
            fill_spawn_capacity_for(sid)

    def finalize_kill_authority(
        enemy_uid: int,
        killer_id: int,
    ) -> None:
        spawn_id, tgt = detach_enemy(enemy_uid)
        if spawn_id is None or tgt is None:
            return
        state = spawn_states[spawn_id]
        state["kills"] += 1
        total = state["kills"]
        grant_kill_rewards(killer_id, tgt)
        if enemy_tcp.connected:
            enemy_tcp.send_kill(enemy_uid, spawn_id, total_kills=total, killer_id=killer_id, confirmed=True)
        if state["kills"] >= state["cap"]:
            state["done"] = True
        else:
            fill_spawn_capacity_for(spawn_id)

    def handle_kill_confirmation_event(
        enemy_uid: int,
        spawn_id: int,
        total_kills: int,
        killer_id: int,
    ) -> None:
        sid, tgt = detach_enemy(enemy_uid)
        if sid is None:
            sid = spawn_id
        state = spawn_states[sid]
        if total_kills >= 0:
            state["kills"] = total_kills
        grant_kill_rewards(killer_id, tgt)
        if state["kills"] >= state["cap"]:
            state["done"] = True

    def handle_kill_request(enemy_uid: int, killer_id: int) -> None:
        finalize_kill_authority(enemy_uid, killer_id)

    if is_authority:
        fill_active_spawns()

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
    # --- Load glass crack sprite (once) ---
    glass_crack = cv2.imread("img/glass_crack_1024.png", cv2.IMREAD_UNCHANGED)
    if glass_crack is None:
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
    local_face_sprite = None
    if args.capture_opponent:
        # 顔を検出し、blue_cap.csv に合わせて帽子を被せたスプライトをここで作る
        local_face_sprite = capture_opponent_frame(mask_path="img/blue_cap.png")
    if local_face_sprite is None:
        print("[WARN] No local face sprite (capture disabled or failed). Using circle placeholder for self-preview.")

    # --- Start face_send_loop thread if P2P is enabled and we have a valid face sprite ---
    if p2p_enabled and (p2p_face_sock is not None) and (p2p_face_peer_addr is not None) and (local_face_sprite is not None):
        try:
            threading.Thread(
                target=net.face_send_loop,
                args=(p2p_face_sock, p2p_face_peer_addr, local_face_sprite),
                daemon=True
            ).start()
        except Exception as e:
            print(f"[P2P_FACE] send thread start failed: {e}")

    # --- Load item sprites (field pickups only) ---
    def _load_item_rgba(path, world_size=64):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Item sprite not found: {path}")
            return None
        # world pickup icon
        ws = cv2.resize(img, (world_size, world_size), interpolation=cv2.INTER_AREA)
        return ws

    item_world_sprites = [None] * 2
    item_paths = [
        "img/HP_portion.png",     # 0: heal
        "img/Attack_portion.png", # 1: attack up
    ]
    for idx, p in enumerate(item_paths):
        item_world_sprites[idx] = _load_item_rgba(p)

    # --- Prebuild AT-field style overlays ---
    hex_overlay = build_hex_overlay(SCR_W, SCR_H, cell=42, line_th=1, color=(0,165,255))  # orange-ish
    tint_overlay = np.full((SCR_H, SCR_W, 3), (0, 120, 255), dtype=np.uint8)  # light orange tint (BGR)

    prev_time = time.time()

    # --- World (FPS) camera: 画面左上がワールド上のどこかを表す ---
    # ここではひとまず (0,0) スタート。
    # 矢印キーで cam_x, cam_y を動かして「ワールド座標が動いているか」を確認する。
    cam_x, cam_y = get_cam_start(player_id)  # world coords of window top-left
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

    # --- One-eye mode tracking ---
    one_eye_mode = False
    one_eye_consec = 0
    both_eye_consec = 0

    if snd_bgm is not None:
        snd_bgm.play()

    while True:
        now_time = time.time()
        dt = float(np.clip(now_time - prev_time, 1e-4, 0.25))
        prev_time = now_time
        # 定速ループ（OpenCV waitKey とは別に実時間 delta を参照）

        with net.lock:
            x = net.latest["x"]
            y = net.latest["y"]
            raw_x = x
            raw_y = y

            # --- Calibrate center over first calib_N frames, then apply 5x around that center ---
            if calib_active and not calib_done:
                calib_sumx += x
                calib_sumy += y
                calib_count += 1
                if calib_count >= calib_N:
                    calib_cx = calib_sumx / float(calib_count)
                    calib_cy = calib_sumy / float(calib_count)
                    calib_done = True
                    calib_active = False
                    game_phase = GAME_PHASE_PLAY
            elif calib_done:
                dx = x - calib_cx
                dy = y - calib_cy
                x = float(np.clip(0.5 + gain_xy * dx, 0.0, 1.0))
                y = float(np.clip(0.5 + gain_xy * dy, 0.0, 1.0))

            brow = net.latest["brow"]
            # Expression features from sender (for baseline capture)
            smile_val = float(net.latest.get("smile", 0.0))
            frown_val = float(net.latest.get("frown", 0.0))
            browh_val = float(net.latest.get("brow_h", 0.0))
            # Per-eye openness values from sender (detect_face.py)
            eyeL_val = float(net.latest.get("eyeL", net.latest.get("eyeL_open", net.latest.get("eyeL_open_ema", 0.0))))
            eyeR_val = float(net.latest.get("eyeR", net.latest.get("eyeR_open", net.latest.get("eyeR_open_ema", 0.0))))

            # During the initial calibration frames, accumulate baseline EARs and expression features
            if calib_active and not calib_done:
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

        # --- Handle TCP sync events (calibration + gameplay) ---
        tcp_events = enemy_tcp.drain_events()
        for evt, payload in tcp_events:
            if evt == "CALIB_READY":
                remote_calib_ready = True
            elif evt == "CALIB_START":
                calib_active = True
                calib_start_sent = True
                game_phase = GAME_PHASE_CALIBRATING
            else:
                pending_enemy_events.append((evt, payload))

        if (
            is_authority
            and local_calib_ready
            and remote_calib_ready
            and not calib_start_sent
        ):
            enemy_tcp.send_line("CALIB_START")
            calib_start_sent = True
            calib_active = True
            game_phase = GAME_PHASE_CALIBRATING

        # === Setup screen: hold the game until calibration is done ===
        with net.remote_face_lock:
            opp_face = net.remote_face_sprite.copy() if net.remote_face_sprite is not None else None

        # 受信側は「帽子までかぶった完成画像」が届く前提で、描画用サイズにリサイズするだけ
        if opp_face is not None:
            try:
                oh, ow = opp_face.shape[:2]
                if (oh != FACE_SPRITE_DRAW_SIZE) or (ow != FACE_SPRITE_DRAW_SIZE):
                    opp_face = cv2.resize(
                        opp_face,
                        (FACE_SPRITE_DRAW_SIZE, FACE_SPRITE_DRAW_SIZE),
                        interpolation=cv2.INTER_AREA
                    )
            except Exception as e:
                print(f"[P2P_FACE] resize for draw failed (ignored): {e}")
        if game_phase != GAME_PHASE_PLAY:
            img = np.zeros((SCR_H, SCR_W, 3), np.uint8)
            if game_phase == GAME_PHASE_WAIT_CALIB_SYNC:
                if not enemy_tcp.connected:
                    msg1 = "CONNECTING TO OPPONENT..."
                    cv2.putText(img, msg1, (int(SCR_W*0.5 - 260), int(SCR_H*0.45)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                else:
                    msg1 = "WAITING FOR OPPONENT"
                    local_txt = "READY" if local_calib_ready else "NOT READY"
                    remote_txt = "READY" if remote_calib_ready else "NOT READY"
                    cv2.putText(img, msg1, (int(SCR_W*0.5 - 260), int(SCR_H*0.45)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                    cv2.putText(img, f"Local: {local_txt}", (int(SCR_W*0.5 - 160), int(SCR_H*0.45+40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,220,255), 2)
                    cv2.putText(img, f"Opponent: {remote_txt}", (int(SCR_W*0.5 - 200), int(SCR_H*0.45+80)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,255,200), 2)
                    cv2.putText(img, "Press R when YOU are ready", (int(SCR_W*0.5 - 260), int(SCR_H*0.45+120)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            else:
                msg1 = "SETUP / CALIBRATION"
                msg2 = "Please face forward for 100 frames"
                msg3 = f"Progress: {calib_count}/{calib_N}"
                cv2.putText(img, msg1, (int(SCR_W*0.5 - 220), int(SCR_H*0.45)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                cv2.putText(img, msg2, (int(SCR_W*0.5 - 300), int(SCR_H*0.45+40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,220,255), 2)
                cv2.putText(img, msg3, (int(SCR_W*0.5 - 180), int(SCR_H*0.45+80)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,255,200), 2)
            cv2.putText(img, "Q = Quit", (12, SCR_H-14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            # --- Draw local processed face (helmet + face) preview at top-left ---
            if local_face_sprite is not None:
                try:
                    lf = local_face_sprite

                    # 強めのガードは入れるが、「return」でmainを抜けないようにする
                    if lf is None or not isinstance(lf, np.ndarray) or lf.size == 0:
                        raise ValueError("local_face_sprite is empty or invalid")

                    if lf.ndim == 2:
                        # グレースケール → BGR
                        lf = cv2.cvtColor(lf, cv2.COLOR_GRAY2BGR)
                    elif lf.ndim == 3:
                        ch = lf.shape[2]
                        if ch > 4:
                            # 5チャンネルなどの場合は先頭4チャネルだけを使う
                            lf = lf[:, :, :4]
                        elif ch not in (3, 4):
                            raise ValueError(f"local_face_sprite has unexpected channel size={ch}")
                    else:
                        raise ValueError(f"local_face_sprite has unexpected ndim={lf.ndim}")

                    lh, lw = lf.shape[:2]
                    if lh <= 0 or lw <= 0:
                        raise ValueError(f"local_face_sprite has non-positive shape: {lf.shape}")

                    # FACE_SPRITE_DRAW_SIZE を基準に HUD 用サイズにスケール
                    target = FACE_SPRITE_DRAW_SIZE
                    scale = float(target) / float(max(lh, lw))
                    new_w = max(1, int(lw * scale))
                    new_h = max(1, int(lh * scale))

                    lf_resized = cv2.resize(lf, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # 画面の左上に少し余白を空けて合成
                    img = blend_rgba(img, lf_resized, xoff=10, yoff=10, alpha_scale=1.0)
                except Exception as e:
                    # 描画に失敗してもゲーム本体は継続させる
                    global P2P_FACE_PREVIEW_ERROR_ONCE
                    if not P2P_FACE_PREVIEW_ERROR_ONCE:
                        print(f"[P2P_FACE] local preview draw failed (ignored): {e}")
                        P2P_FACE_PREVIEW_ERROR_ONCE = True
                    # ここでは何もせず、単にプレビュー描画をスキップする

            cv2.imshow(win, img)
            k = cv2.waitKey(int(1000/FPS)) & 0xFF

            # On the WAIT_CALIB_SYNC screen, allow the player to mark themselves READY with 'R'
            if game_phase == GAME_PHASE_WAIT_CALIB_SYNC and enemy_tcp.connected:
                if k in (ord('r'), ord('R')) and (not local_calib_ready):
                    local_calib_ready = True
                    enemy_tcp.send_line(f"CALIB_READY {player_id}")

            if k in (ord('q'), ord('Q'), 27):
                break
            # Skip game logic until calibration completes
            continue

        # --- Enemy spawn/kill sync events (TCP authority mirroring) ---
        if game_phase == GAME_PHASE_PLAY:
            while pending_enemy_events:
                evt, payload = pending_enemy_events.popleft()
                if evt == "SPAWN":
                    if len(payload) >= 3:
                        spawn_id = int(payload[0])
                        spawn_seq = int(payload[1])
                        enemy_uid = int(payload[2])
                        spawn_enemy_from_point(spawn_id, spawn_seq=spawn_seq, enemy_uid=enemy_uid, notify=False)
                elif evt == "KILL":
                    if len(payload) >= 5:
                        enemy_uid = int(payload[0])
                        spawn_id = int(payload[1])
                        total = int(payload[2])
                        killer = int(payload[3])
                        confirmed = payload[4] == "1"
                        if confirmed:
                            handle_kill_confirmation_event(enemy_uid, spawn_id, total, killer)
                        elif is_authority:
                            handle_kill_request(enemy_uid, killer)

        if is_authority:
            fill_active_spawns()

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
        shoot_ev = float(net.latest.get("shoot", 0.0)) > 0.5
        winkL = float(net.latest.get("winkL", 0.0)) > 0.5
        winkR = float(net.latest.get("winkR", 0.0)) > 0.5

        # 目状態の基本判定
        eye_open_now = (eyeL_state >= 0.5) and (eyeR_state >= 0.5)
        one_eye_now  = (not eyes_closed_now) and (not eye_open_now)

        # --- 片目モード切替（連続フレーム条件） ---
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


        # --- P2P: 相手から届いた SHOT イベントを RemotePlayerBullet に変換 ---
        if p2p_enabled:
            with net.remote_shot_lock:
                events = list(net.remote_shot_events)
                net.remote_shot_events.clear()
            for event in events:
                if len(event) >= 8:
                    ox, oy, tx, ty, speed_i, damage_i, shooter_id, shooter_level = event
                else:
                    ox, oy, tx, ty, speed_i, damage_i, shooter_id = event
                    shooter_level = 1
                # speed_i / damage_i は送信側でクリップ済みだが、念のため下限だけ軽く守る
                spd = max(float(SPEED_MIN), float(speed_i))
                dmg = int(damage_i)
                lvl = int(np.clip(shooter_level, 1, 5))
                # 発射位置は「相手プレイヤーが今いるワールド座標」に置き換える
                with p2p_lock:
                    opp_x = float(remote_state.get("x", remote_spawn_x))
                    opp_y = float(remote_state.get("y", remote_spawn_y))

                remote_player_bullets.append(
                    RemotePlayerBullet(
                        ox=opp_x,
                        oy=opp_y,
                        tx=tx,
                        ty=ty,
                        speed=spd,
                        damage=dmg,
                        level=lvl,
                    )
                )

        # 発射（クールダウン）：口の開きから速度とダメージを決めた弾を生成
        if shoot_ev and (time.time() - last_shot_time) >= shoot_cooldown:
            # 照準のスクリーン座標 → ワールド座標
            bx, by = norm_to_px(fps_nx, fps_ny)  # crosshair in pixels
            target_wx = cam_x + bx
            target_wy = cam_y + by

            # 発射口（画面下部左右）をワールド座標で決定
            margin_x = int(SCR_W * 0.20)
            margin_y = 80
            muzzles = [
                (float(cam_x + margin_x),           float(cam_y + (SCR_H - margin_y))),
                (float(cam_x + (SCR_W - margin_x)), float(cam_y + (SCR_H - margin_y))),
            ]

            # 口の縦横比からダメージ・速度を計算
            aspect_raw = float(net.latest.get("aspect", net.latest.get("mar", 0.0)))
            damage_i, speed_i, _ = compute_bullet_params_from_aspect(aspect_raw)
            damage_i = max(1, int(damage_i + attack_bonus))
            damage_i = max(1, damage_i + max(0, player_level - 1))
            lvl = int(np.clip(player_level, 1, 5))

            # SE（すべて同じ種類の弾なので、ショット音のみ）
            if snd_shot is not None:
                snd_shot.play()

            # 左右の砲口から同じパラメータの弾を発射
            for muzzle_x, muzzle_y in muzzles:
                # ローカルのプレイヤー弾を生成
                bullets.append(
                    PlayerBullet(
                        ox=muzzle_x,
                        oy=muzzle_y,
                        tx=target_wx,
                        ty=target_wy,
                        speed=speed_i,
                        damage=damage_i,
                        level=lvl,
                        r0=10.0,
                        r_min=2.0,
                    )
                )

                # P2P: 相手にもこの弾の情報をイベントとして送信（UDP）
                # フォーマット: b'SHOT' + struct('!ffffhhB')
                #   ox, oy, tx, ty (float32) / speed (int16) / damage (int16) / shooter_id (uint8)
                if p2p_enabled and (p2p_sock is not None) and (p2p_peer_addr is not None):
                    try:
                        pkt = struct.pack(
                            '!4sffffhhBB',
                            b'SHOT',
                            float(muzzle_x), float(muzzle_y),
                            float(target_wx), float(target_wy),
                            int(speed_i), int(damage_i),
                            0,   # shooter_id
                            int(lvl),
                        )
                        p2p_sock.sendto(pkt, p2p_peer_addr)
                        # デバッグ用：送信内容を標準出力に出す
                        print(f"[P2P-SHOT-SEND] ox={muzzle_x:.1f} oy={muzzle_y:.1f}  tx={target_wx:.1f} ty={target_wy:.1f}  speed={speed_i} dmg={damage_i}")
                    except Exception as e:
                        # ショット通知の失敗は致命的ではないのでログだけ出す
                        print(f"[P2P-SHOT] send error: {e}")

            last_shot_time = time.time()

        # 物理更新
        dt = DT
        # 片目モード中は顔の動きでカメラ(cam_x, cam_y)をパンする
        if one_eye_mode:
            # 中心(0.5,0.5)からのズレを速度ベクトルとして解釈
            dx_norm = float(x) - 0.5
            dy_norm = float(y) - 0.5
            # 水平方向: CAM_PAN_SPEED, 垂直方向: その2倍
            cam_x += dx_norm * CAM_PAN_SPEED * dt
            cam_y += dy_norm * (CAM_PAN_SPEED * 2.0) * dt

        # プレイヤー弾は常に更新
        for b in bullets:
            b.step(dt)
        bullets = [
            b for b in bullets
            if 0 <= b.x < WORLD_W and 0 <= b.y < WORLD_H and b.ttl > 0
        ]

        for tg in targets:
            tg.step(dt)
        # 対戦相手プレイヤーもワールド内を移動（P2P時はネットワーク座標で上書き）
        if opponent is not None:
            if p2p_enabled:
                with p2p_lock:
                    opponent.x = float(remote_state.get("x", opponent.x))
                    opponent.y = float(remote_state.get("y", opponent.y))
            else:
                opponent.step(dt)

        def _target_visible_on_screen(tg_obj: Target, margin: float = 0.0) -> bool:
            """Return True if a target is within the current camera viewport (with optional margin)."""
            if tg_obj is None:
                return False
            return (
                (cam_x - margin) <= tg_obj.x <= (cam_x + SCR_W + margin)
                and (cam_y - margin) <= tg_obj.y <= (cam_y + SCR_H + margin)
            )

        # 敵弾スポーン（視界内の敵のみ発射）
        enemy_spawn_timer += dt
        if enemy_spawn_timer >= enemy_spawn_period and len(targets) > 0:
            enemy_spawn_timer = 0.0
            visible_candidates = [
                tg for tg in targets if _target_visible_on_screen(tg, margin=32.0)
            ]
            if visible_candidates:
                tg = random.choice(visible_candidates)
                enemy_bullets.append(EnemyBullet(tg.x, tg.y))
            enemy_spawn_period = random.uniform(0.8, 1.4)

        # プレイヤーのワールド中心座標（カメラ位置＋画面中心）
        player_world_x = cam_x + SCR_W * 0.5
        player_world_y = cam_y + SCR_H * 0.5
        if p2p_enabled:
            with p2p_lock:
                remote_hp_value = float(remote_state.get("hp", player_hp_max))
        else:
            remote_hp_value = None
        # --- P2P send local player position ---
        if p2p_enabled and p2p_sock is not None and p2p_peer_addr is not None:
            try:
                pkt = struct.pack('3f', float(player_world_x), float(player_world_y), float(player_hp))
                p2p_sock.sendto(pkt, p2p_peer_addr)
            except Exception:
                pass

        # 敵弾更新（ワールド外・寿命切れを捨てる）
        kept = []
        for eb in enemy_bullets:
            eb.step(dt, player_world_x, player_world_y)
            # 寿命切れ
            if eb.ttl <= 0:
                continue
            # ワールド外（少しマージンを持たせる）
            if eb.x < -100 or eb.x > WORLD_W + 100 or eb.y < -100 or eb.y > WORLD_H + 100:
                continue
            kept.append(eb)
        enemy_bullets = kept

        # 相手プレイヤー弾更新（ワールド外のみ捨てる。寿命はヒット処理後にまとめて削除）
        kept_remote = []
        for rb in remote_player_bullets:
            rb.step(dt, player_world_x, player_world_y)
            # ワールド外（少しマージンを持たせる）
            if rb.x < -100 or rb.x > WORLD_W + 100 or rb.y < -100 or rb.y > WORLD_H + 100:
                continue
            kept_remote.append(rb)
        remote_player_bullets = kept_remote

        # 自弾 vs 相手プレイヤー弾：衝突したら双方消滅
        if bullets and remote_player_bullets:
            kept_remote = []
            bullets_changed = False
            for rb in remote_player_bullets:
                collided = False
                for b in bullets:
                    # 簡易：半径和で衝突判定
                    hit_r = float(rb.r + getattr(b, "r", 8.0))
                    dx = b.x - rb.x
                    dy = b.y - rb.y
                    if dx * dx + dy * dy <= hit_r * hit_r:
                        # 双方消滅させ、パリィ風パーティクルを出す
                        b.ttl = 0.0
                        collided = True
                        for _ in range(10):
                            ang = random.uniform(0, 2 * math.pi)
                            spd = random.uniform(200, 480)
                            life = random.uniform(0.05, 0.12)
                            particles.append({
                                "x": float(rb.x), "y": float(rb.y),
                                "vx": math.cos(ang) * spd,
                                "vy": math.sin(ang) * spd,
                                "life": life
                            })
                        break
                if not collided:
                    kept_remote.append(rb)
                else:
                    bullets_changed = True
            remote_player_bullets = kept_remote
            if bullets_changed:
                bullets = [b for b in bullets if b.ttl > 0]

        def _apply_bullet_hit_like_enemy(bullet):
            """Apply hit effects for a single incoming bullet (enemy or remote-player).
            Returns True if the bullet should be removed, False otherwise.
            """
            nonlocal player_hp, shield_hp, shield_on, score, shield_gen_channel, shield_fx_start, cracks, crack_sprites

            # World-space impact position (where the bullet is when it "arrives")
            hit_x = float(getattr(bullet, "x", 0.0))
            hit_y = float(getattr(bullet, "y", 0.0))

            # only show/audio effects if impact is visible on this screen
            hit_visible = (
                cam_x <= hit_x <= (cam_x + SCR_W) and
                cam_y <= hit_y <= (cam_y + SCR_H)
            )

            def _play_hit_sound(snd_obj):
                if not hit_visible or snd_obj is None:
                    return
                try:
                    snd_obj.play()
                except Exception:
                    pass

            if hit_visible:
                try:
                    if cracks:
                        cracks.pop(0)
                    cracks.append(make_crack(hit_x, hit_y))
                except Exception:
                    pass

                if glass_crack is not None:
                    try:
                        if crack_sprites:
                            crack_sprites.pop(0)
                        crack_life = 0.9
                        crack_sprites.append({
                            "img": glass_crack,
                            "x": hit_x,
                            "y": hit_y,
                            "life": crack_life,
                            "max": crack_life,
                        })
                    except Exception:
                        pass

            # If shield is active, consume shield HP first
            if shield_on and shield_hp > 0:
                shield_hp -= 1
                _play_hit_sound(snd_shield_frag)
                # If shield HP is depleted, turn shield off and stop generation sound
                if shield_hp <= 0:
                    shield_hp = 0
                    shield_on = False
                    if shield_gen_channel is not None and snd_shield_gen is not None:
                        try:
                            shield_gen_channel.stop()
                        except Exception:
                            pass
                        shield_gen_channel = None
                # Bullet is always consumed on shield hit
                bullet.ttl = 0.0
                return True

            # No (or depleted) shield: player HP takes damage
            if hit_visible and player_hp > 0:
                player_hp -= 1
                if player_hp < 0:
                    player_hp = 0
                _play_hit_sound(snd_hitted)

            # Bullet consumed after damaging player
            bullet.ttl = 0.0
            return True

        # enemy_bullets のヒット
        tmp = []
        for eb in enemy_bullets:
            if eb.r >= eb.impact_r:
                if _apply_bullet_hit_like_enemy(eb):
                    continue
            tmp.append(eb)
        enemy_bullets = tmp

        # remote_player_bullets のヒット
        tmp = []
        for rb in remote_player_bullets:
            if rb.r >= rb.impact_r:
                if _apply_bullet_hit_like_enemy(rb):
                    continue
            tmp.append(rb)
        remote_player_bullets = tmp
        # ヒット処理後に、寿命切れの相手弾をまとめて削除
        remote_player_bullets = [rb for rb in remote_player_bullets if rb.ttl > 0]


        # パーティクル更新
        new_particles = []
        for p in particles:
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["life"] -= dt
            if p["life"] > 0:
                new_particles.append(p)
        particles = new_particles

        # クラック（ひび）更新
        new_cracks = []
        for c in cracks:
            c["life"] -= dt
            if c["life"] > 0:
                new_cracks.append(c)
        cracks = new_cracks

        # ひび（画像デカール）寿命更新
        new_crack_sprites = []
        for c in crack_sprites:
            c["life"] -= dt
            if c["life"] > 0:
                new_crack_sprites.append(c)
        crack_sprites = new_crack_sprites

        # 当たり判定（円 vs 点/小円） - kills are confirmed by Player1 via TCP
        for tg in targets:
            if getattr(tg, "_dead", False):
                continue
            if getattr(tg, "pending_kill", False):
                continue
            for b in bullets:
                if b.ttl <= 0:
                    continue
                dx_b = b.x - b.ax
                dy_b = b.y - b.ay
                if (dx_b * dx_b + dy_b * dy_b) <= (b.hit_window_px * b.hit_window_px):
                    if (tg.x - b.ax)**2 + (tg.y - b.ay)**2 <= (tg.r + b.aim_radius_px)**2:
                        tg.hp -= b.dmg
                        b.ttl = 0.0
            if tg.spawn_point_id is None or tg.spawn_point_id < 0:
                continue
            if tg.hp <= 0:
                enemy_uid = getattr(tg, "enemy_uid", None)
                if enemy_uid is None:
                    continue
                if is_authority:
                    finalize_kill_authority(enemy_uid, player_id)
                else:
                    tg.pending_kill = True
                    tg.hp = 0.0
                    if enemy_tcp.connected:
                        enemy_tcp.send_kill(enemy_uid, tg.spawn_point_id, killer_id=player_id, confirmed=False)

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
        if targets:
            targets = [tg for tg in targets if not getattr(tg, "_dead", False)]

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
            if dist2 <= pickup_radius * pickup_radius:
                apply_item_effect(it.kind)
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
            else:
                kept_items.append(it)
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
            # ワールド中心座標 → 擬似3D投影（画面座標）。位置だけ奥行きで動かし、サイズは固定。
            tx_s, ty_s, _ = project_to_screen(tg.x, tg.y, tg.z, cam_x, cam_y, SCR_W, SCR_H)
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

            # Draw alien sprite with alpha (randomly assigned per target)
            if enemy_sprites:
                if not hasattr(tg, "sprite"):
                    tg.sprite = random.choice(enemy_sprites)
                ex, ey = int(tx_s), int(ty_s)
                # スプライトサイズも奥に行くほど小さく
                sprite_size = int(base_r * 2.2)
                sprite_size = max(8, sprite_size)
                sprite = cv2.resize(tg.sprite, (sprite_size, sprite_size))
                sh, sw = sprite.shape[:2]
                sx, sy = ex - sw//2, ey - sh//2
                # Alpha blend
                if 0 <= sx < SCR_W and 0 <= sy < SCR_H:
                    x1 = max(sx, 0)
                    y1 = max(sy, 0)
                    x2 = min(sx+sw, SCR_W)
                    y2 = min(sy+sh, SCR_H)
                    sprite_x1 = x1 - sx
                    sprite_y1 = y1 - sy
                    sprite_x2 = sprite_x1 + (x2 - x1)
                    sprite_y2 = sprite_y1 + (y2 - y1)
                    roi = img[y1:y2, x1:x2]
                    spr = sprite[sprite_y1:sprite_y2, sprite_x1:sprite_x2]
                    alpha_spr = spr[:,:,3:4] / 255.0
                    roi[:] = (roi * (1 - alpha_spr) + spr[:,:,:3] * alpha_spr).astype(np.uint8)
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
            draw_player_bullet(img, b, cam_x, cam_y, SCR_W, SCR_H)

        # 敵弾（深度つき）
        for eb in enemy_bullets:
            ex_s, ey_s, depth_scale = project_to_screen(eb.x, eb.y, eb.z, cam_x, cam_y, SCR_W, SCR_H)
            r = int(max(1, eb.r * depth_scale))
            # 中心を明るく、外縁を濃くして近接感
            cv2.circle(img, (int(ex_s), int(ey_s)), r, (0, 90, 255), -1)
            cv2.circle(img, (int(ex_s), int(ey_s)), r+2, (0, 60, 200), 2)
            # 近づくほど薄いグロー（impact_r 比はそのまま、見かけ半径だけ深度スケール）
            glow_alpha = min(0.5, (eb.r / eb.impact_r) * 0.5)
            draw_alpha_circle(img, (int(ex_s), int(ey_s)), r+6, (0, 120, 255), glow_alpha, thickness=2)

        # 対戦相手プレイヤー（将来のPVP用ダミー）を奥レイヤーに描画
        if opponent is not None:
            # 位置は z によるパララックスを反映するが、サイズは z ではスケーリングしない
            ox_s, oy_s, _ = project_to_screen(opponent.x, opponent.y, opponent.z, cam_x, cam_y, SCR_W, SCR_H)
            opp_indicator = compute_edge_indicator(ox_s, oy_s, SCR_W, SCR_H, margin=28)
            if opp_indicator is not None:
                cv2.circle(img, opp_indicator, 6, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
            if opp_face is not None and opp_face.shape[2] == 4:
                # 深度に関わらず、半径ベースの一定スケールで描画
                size = int(max(24, opponent.r * 2.4))
                sprite = cv2.resize(opp_face, (size, size), interpolation=cv2.INTER_AREA)
                sh, sw = sprite.shape[:2]
                sx = int(ox_s) - sw // 2
                sy = int(oy_s) - sh // 2
                img = blend_rgba(img, sprite, xoff=sx, yoff=sy, alpha_scale=1.0)
            else:
                # フォールバック：従来通りのサークル＋"OP"テキスト（サイズは z 非依存）
                o_r = int(max(8, opponent.r))
                cv2.circle(img, (int(ox_s), int(oy_s)), o_r, (255, 180, 80), 2)
                cv2.putText(img, "OP", (int(ox_s - 10), int(oy_s + 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 150), 1)

        # 自分が相手に送る顔画像のプレビューを画面左上に表示
        if local_face_sprite is not None and local_face_sprite.shape[2] == 4:
            preview_size = 96
            preview = cv2.resize(local_face_sprite, (preview_size, preview_size), interpolation=cv2.INTER_AREA)
            pv_x, pv_y = 12, 12  # top-left corner of preview
            img = blend_rgba(img, preview, xoff=pv_x, yoff=pv_y, alpha_scale=1.0)
            # 枠とラベル
            cv2.rectangle(img,
                          (pv_x - 2, pv_y - 2),
                          (pv_x + preview_size + 2, pv_y + preview_size + 2),
                          (80, 80, 80), 1)
            cv2.putText(img, "YOU",
                        (pv_x, pv_y + preview_size + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (230, 230, 230), 1, cv2.LINE_AA)

        # ひび描画（フェードアウト）
        # 専用レイヤーにまとめて描画し、最後に一括ブレンドすることで高速化
        crack_layer = np.zeros_like(img)
        for c in cracks:
            a = max(0.0, min(1.0, c["life"] / c["max_life"]))
            if a <= 0.0:
                continue
            # 外側ほど薄くなるよう、線を2層（濃→薄）で重ねる
            for pts in c["lines"]:
                scr_pts = []
                for (px, py) in pts:
                    sx_p, sy_p = world_to_screen(px, py, cam_x, cam_y)
                    scr_pts.append((sx_p, sy_p))
                if len(scr_pts) < 2:
                    continue
                # 濃い線（外側）
                f1 = 0.45 * a
                col1 = (int(210 * f1), int(220 * f1), int(255 * f1))
                cv2.polylines(
                    crack_layer,
                    [np.array(scr_pts, dtype=np.int32)],
                    isClosed=False,
                    color=col1,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
                # 薄い線（内側）
                f2 = 0.25 * a
                col2 = (int(140 * f2), int(160 * f2), int(200 * f2))
                cv2.polylines(
                    crack_layer,
                    [np.array(scr_pts, dtype=np.int32)],
                    isClosed=False,
                    color=col2,
                    thickness=1,
                    lineType=cv2.LINE_AA
                )
            # 中心の擦りキズ（小さめの円）
            cx_s, cy_s = world_to_screen(c["x"], c["y"], cam_x, cam_y)
            f3 = 0.20 * a
            col3 = (int(220 * f3), int(230 * f3), int(255 * f3))
            cv2.circle(
                crack_layer,
                (int(cx_s), int(cy_s)),
                10,
                col3,
                thickness=2,
                lineType=cv2.LINE_AA
            )
        # 元のフレームと一括ブレンド
        img = cv2.addWeighted(img, 1.0, crack_layer, 1.0, 0, img)

        # ひび（画像デカール）描画（フェードアウト）
        for c in crack_sprites:
            a = max(0.0, min(1.0, c["life"] / c["max"]))
            spr = c["img"]
            sh, sw = spr.shape[:2]
            cx_s, cy_s = world_to_screen(c["x"], c["y"], cam_x, cam_y)
            cx_i, cy_i = int(cx_s), int(cy_s)
            sx, sy = cx_i - sw // 2, cy_i - sh // 2
            x1, y1 = max(sx, 0), max(sy, 0)
            x2, y2 = min(sx + sw, SCR_W), min(sy + sh, SCR_H)
            if x1 < x2 and y1 < y2:
                roi  = img[y1:y2, x1:x2]
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
            pulse = 0.30 + 0.20 * math.sin(2 * math.pi * 2.0 * tfx)  # 0.10～0.50くらい
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

        # ワールド座標は「画面中心のワールド位置」とし、cam_x, cam_y だけで決まる
        player_world_x = cam_x + SCR_W * 0.5
        player_world_y = cam_y + SCR_H * 0.5

        xp_to_next = get_next_level_xp(player_level)
        draw_hud(
            img,
            score=score,
            player_level=player_level,
            player_xp=player_xp,
            xp_to_next=xp_to_next,
            player_hp=player_hp,
            player_hp_max=player_hp_max,
            shield_hp=shield_hp,
            shield_hp_max=shield_hp_max,
            shield_on=shield_on,
            calib_done=calib_done,
            calib_count=calib_count,
            calib_N=calib_N,
            calib_cx=calib_cx,
            calib_cy=calib_cy,
            gain_xy=gain_xy,
            th_open_L=th_open_L,
            th_close_L=th_close_L,
            th_open_R=th_open_R,
            th_close_R=th_close_R,
            fps_x=fps_x,
            fps_y=fps_y,
            player_world_x=player_world_x,
            player_world_y=player_world_y,
            rel_smile=rel_smile,
            rel_frown=rel_frown,
            rel_browh=rel_browh,
        )

        # 相手プレイヤー弾（P2P SHOT）の描画（黄色）
        # 先に発射された弾が「上」に来るように、古い弾ほど後から描画する
        for rb in reversed(remote_player_bullets):
            draw_remote_player_bullet(img, rb, cam_x, cam_y, SCR_W, SCR_H)

        if player_hp <= 0:
            cv2.putText(img, "YOU LOSE", (SCR_W//2 - 180, SCR_H//2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
        elif remote_hp_value is not None and remote_hp_value <= 0:
            cv2.putText(img, "YOU WIN", (SCR_W//2 - 160, SCR_H//2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4, cv2.LINE_AA)

        cv2.imshow(win, img)
        prev_shoot = shoot_ev
        k = cv2.waitKey(int(1000/FPS)) & 0xFF

        # --- Opponent manual movement (ASDW) ---
        if opponent is not None and not p2p_enabled:
            opp_speed = 260.0 * dt  # opponent movement speed in world units per frame
            if k == ord('a'):   # left
                opponent.x -= opp_speed
            if k == ord('d'):   # right
                opponent.x += opp_speed
            if k == ord('w'):   # up
                opponent.y -= opp_speed
            if k == ord('s'):   # down
                opponent.y += opp_speed

            # Clamp opponent inside world bounds (X) and a band in Y
            opponent.x = float(np.clip(opponent.x, opponent.r, WORLD_W - opponent.r))
            y_min = opponent.r
            y_max = WORLD_H - opponent.r
            opponent.y = float(np.clip(opponent.y, y_min, y_max))

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

    cv2.destroyAllWindows()
    if return_to_home:
        relaunch_home_screen()

if __name__ == "__main__":
    main()
