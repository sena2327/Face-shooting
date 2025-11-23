#!/usr/bin/env python3
import argparse
import os
import socket, struct, threading, time, math, random
import cv2
import pygame
import numpy as np
import mediapipe as mp
from collections import deque

SCR_W, SCR_H = 960, 540        # ゲーム画面サイズ（自由に変更）
WORLD_W, WORLD_H = SCR_W * 10, SCR_H * 10  # ワールド全体のサイズ（画面の10倍）

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

EYE_CLOSE_TH = 0.18    # 閉眼しきい値（EAR/虹彩向け）
EYE_OPEN_TH  = 0.26    # 再オープンしきい値（ヒステリシス）


EYE_CLOSE_CONSEC = 30    # 連続フレーム数（この回数以上で発動）

# --- Shield activation window parameters ---
WINDOW_N = 50        # frames
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

# --- 片目→両目モードへ戻すまでの遅延[s]と、その間に必要な片目率 ---
ONE_EYE_TO_DUAL_DELAY = 1.0      # 1.0秒の間は片目扱いを維持する最大時間
ONE_EYE_WINDOW_RATIO = 0.90      # その0.2秒のうち9割以上が片目ならワールド操作を維持

# 受信データ: 10f/11f/12f/13f/16f に対応
# 基本レイアウト: detect_face.py の16fフォーマットは
# [x, y, relZ, mouth, eyeL, eyeR, brow, frown, smile, brow_h, winkL, winkR, shoot, aspect, norm_w, cheek_puff]
# 13fのときは先頭13要素のみ（aspect/norm_w/cheek_puff は送られない）
latest = {
    "x":0.5, "y":0.5, "z":0.0,
    "mouth":0.0, "eyeL":1.0, "eyeR":1.0,
    "brow":0.0,        # eyebrow raise (0..1)
    "winkL":0.0, "winkR":0.0, "shoot":0.0,
    "weapon_id":0.0,     # optional (11f/13f legacy)
    "aspect":0.0,        # mouth_h / mouth_w (16f extras)
    "norm_w":0.0,        # mouth_w / inter_oc_px (16f extras)
    "mar": 0.0,          # Mouth Aspect Ratio (alias, not送信)
    "cheek_puff":0.0,    # 0..1 (16f extras: mouth-shape + cheek layout)
    "smile": 0.0,        # 0..1 smile score (from sender base_vals)
    "frown": 0.0,        # 0..1 frown score (from sender base_vals)
    "brow_h": 0.0        # eyebrow metric / height (from sender base_vals)
}
lock = threading.Lock()

# --- P2P face sprite globals ---
remote_face_sprite = None
remote_face_lock = threading.Lock()
face_ack_event = threading.Event()

# --- P2P face fragmentation / reassembly state ---
# frame_id -> {"total": int, "chunks": {idx: bytes}, "from_addr": (ip, port)}
face_recv_frames = {}
face_frame_id_counter = 1
face_frame_id_lock = threading.Lock()

# --- P2P shot events from opponent (for remote bullets, etc.) ---
remote_shot_events = deque()        # each item: (ox, oy, tx, ty, speed_i, damage_i, shooter_id)
remote_shot_lock = threading.Lock()

def next_face_frame_id():
    global face_frame_id_counter
    with face_frame_id_lock:
        fid = face_frame_id_counter
        face_frame_id_counter += 1
        return fid

def udp_loop(listen_ip, listen_port):
    """Receive gaze/face data over UDP and update the global `latest` dict.

    listen_ip, listen_port はコマンドライン引数から渡される想定で、
    これによりローカルでポート番号を変えた複数インスタンス起動が可能になる。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((listen_ip, listen_port))
    except OSError as e:
        print(f"[UDP] bind failed on {listen_ip}:{listen_port}: {e}")
        return
    sock.settimeout(1.0)
    print(f"[UDP] listening on {listen_ip}:{listen_port}")
    while True:
        try:
            data, _ = sock.recvfrom(4096)
            off = 0
            while off < len(data):
                remaining = len(data) - off
                parsed = False
                # Try 16 floats (64 bytes): extended layout with smile/frown/brow_h
                if remaining >= 64 and not parsed:
                    try:
                        vals = struct.unpack_from("16f", data, off)
                        off += 64; parsed = True
                        with lock:
                            latest["x"] = float(np.clip(vals[0], 0, 1))
                            latest["y"] = float(np.clip(vals[1], 0, 1))
                            latest["z"] = float(vals[2])
                            latest["mouth"] = float(np.clip(vals[3], 0, 1))
                            latest["eyeL"] = float(np.clip(vals[4], 0, 1))
                            latest["eyeR"] = float(np.clip(vals[5], 0, 1))
                            latest["brow"] = float(np.clip(vals[6], 0, 1))         # brow_raise
                            latest["frown"] = float(np.clip(vals[7], 0, 1))        # brow_frown
                            latest["smile"] = float(np.clip(vals[8], 0, 1))        # smile_score
                            latest["brow_h"] = float(vals[9])                      # brow_metric_ema
                            latest["winkL"] = float(vals[10])
                            latest["winkR"] = float(vals[11])
                            latest["shoot"] = float(vals[12])
                            latest["aspect"] = float(vals[13])
                            latest["norm_w"] = float(vals[14])
                            latest["cheek_puff"] = float(np.clip(vals[15], 0.0, 1.0))
                            # MAR は送られてこないので、aspect からの別名として扱う
                            latest["mar"] = latest["aspect"]
                        continue
                    except struct.error:
                        pass
                # Try 13 floats (52 bytes)
                if remaining >= 52 and not parsed:
                    try:
                        vals = struct.unpack_from("13f", data, off)
                        off += 52; parsed = True
                        with lock:
                            latest["x"] = float(np.clip(vals[0],0,1))
                            latest["y"] = float(np.clip(vals[1],0,1))
                            latest["z"] = float(vals[2])
                            latest["mouth"] = float(np.clip(vals[3],0,1))
                            latest["eyeL"]  = float(np.clip(vals[4],0,1))
                            latest["eyeR"]  = float(np.clip(vals[5],0,1))
                            latest["brow"]  = float(np.clip(vals[6],0,1))         # brow_raise
                            latest["frown"] = float(np.clip(vals[7],0,1))         # brow_frown
                            latest["smile"] = float(np.clip(vals[8],0,1))         # smile_score
                            latest["brow_h"] = float(vals[9])                     # brow_metric_ema
                            latest["winkL"] = float(vals[10])
                            latest["winkR"] = float(vals[11])
                            latest["shoot"] = float(vals[12])
                            # aspect/norm_w/cheek_puff は13fでは送られないので0にリセット
                            latest["aspect"] = 0.0
                            latest["norm_w"] = 0.0
                            latest["cheek_puff"] = 0.0
                            # MAR も送られてこないので、mouth_open 相当から別名として扱う
                            latest["mar"] = float(latest["mouth"])
                        continue
                    except struct.error:
                        pass
                # Try 12 floats (48 bytes)
                if remaining >= 48 and not parsed:
                    try:
                        vals = struct.unpack_from("12f", data, off)
                        off += 48; parsed = True
                        with lock:
                            latest["x"] = float(np.clip(vals[0],0,1))
                            latest["y"] = float(np.clip(vals[1],0,1))
                            latest["z"] = float(vals[2])
                            latest["mouth"] = float(np.clip(vals[3],0,1))
                            latest["eyeL"]  = float(np.clip(vals[4],0,1))
                            latest["eyeR"]  = float(np.clip(vals[5],0,1))
                            latest["brow"]  = float(np.clip(vals[6],0,1))
                            latest["winkL"] = float(vals[7])
                            latest["winkR"] = float(vals[8])
                            latest["shoot"] = float(vals[9])
                            latest["aspect"] = float(vals[10])
                            latest["norm_w"] = float(vals[11])
                            latest["cheek_puff"] = 0.0
                            latest["smile"] = 0.0
                            latest["frown"] = 0.0
                            latest["brow_h"] = 0.0
                            latest["mar"] = float(latest["mouth"])
                        continue
                    except struct.error:
                        pass
                # Try 11 floats (44 bytes)
                if remaining >= 44 and not parsed:
                    try:
                        vals = struct.unpack_from("11f", data, off)
                        off += 44; parsed = True
                        with lock:
                            latest["x"] = float(np.clip(vals[0],0,1))
                            latest["y"] = float(np.clip(vals[1],0,1))
                            latest["z"] = float(vals[2])
                            latest["mouth"] = float(np.clip(vals[3],0,1))
                            latest["eyeL"]  = float(np.clip(vals[4],0,1))
                            latest["eyeR"]  = float(np.clip(vals[5],0,1))
                            latest["brow"]  = float(np.clip(vals[6],0,1))
                            latest["winkL"] = float(vals[7])
                            latest["winkR"] = float(vals[8])
                            latest["shoot"] = float(vals[9])
                            latest["weapon_id"] = float(vals[10])
                            latest["aspect"] = 0.0
                            latest["norm_w"] = 0.0
                            latest["cheek_puff"] = 0.0
                            latest["smile"] = 0.0
                            latest["frown"] = 0.0
                            latest["brow_h"] = 0.0
                            latest["mar"] = float(latest["mouth"])
                        continue
                    except struct.error:
                        pass
                # Fallback: 10 floats (40 bytes)
                if remaining >= 40 and not parsed:
                    try:
                        vals = struct.unpack_from("10f", data, off)
                        off += 40; parsed = True
                        with lock:
                            latest["x"] = float(np.clip(vals[0],0,1))
                            latest["y"] = float(np.clip(vals[1],0,1))
                            latest["z"] = float(vals[2])
                            latest["mouth"] = float(np.clip(vals[3],0,1))
                            latest["eyeL"]  = float(np.clip(vals[4],0,1))
                            latest["eyeR"]  = float(np.clip(vals[5],0,1))
                            latest["brow"]  = float(np.clip(vals[6],0,1))
                            latest["winkL"] = float(vals[7])
                            latest["winkR"] = float(vals[8])
                            latest["shoot"] = float(vals[9])
                            latest["aspect"] = 0.0
                            latest["norm_w"] = 0.0
                            latest["cheek_puff"] = 0.0
                            latest["smile"] = 0.0
                            latest["frown"] = 0.0
                            latest["brow_h"] = 0.0
                            latest["mar"] = float(latest["mouth"])
                        continue
                    except struct.error:
                        pass
                # If none matched, break to avoid infinite loop
                break
        except socket.timeout:
            pass
        except Exception as e:
            print("[UDP] err:", e)
            time.sleep(0.2)
def p2p_recv_loop(sock, remote_state, state_lock):
    """Receive opponent player world position and shot events over P2P UDP socket.

    Packet formats:
      - Position: struct('2f') -> (x, y) in world coordinates.
        Updates remote_state['x'], remote_state['y'] under state_lock.
      - Shot event: b'SHOT' + struct('!ffffhhB'):
          ox, oy, tx, ty (float32) / speed (int16) / damage (int16) / shooter_id (uint8)
        Queued into the global remote_shot_events deque under remote_shot_lock.
    """
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            if not data:
                continue

            # --- Shot event packet ---
            # Format: b'SHOT' + struct('!ffffhhB')
            #   ox, oy, tx, ty, speed_i, damage_i, shooter_id
            if len(data) >= 25 and data[:4] == b'SHOT':
                try:
                    _, ox, oy, tx, ty, speed_i, damage_i, shooter_id = struct.unpack('!4sffffhhB', data[:25])

                    # --- Debug print for received shot events ---
                    print(f"[P2P-SHOT-RECV] ox={ox:.1f} oy={oy:.1f}  tx={tx:.1f} ty={ty:.1f}  speed={speed_i} dmg={damage_i} shooter={shooter_id}")

                    with remote_shot_lock:
                        remote_shot_events.append(
                            (float(ox), float(oy), float(tx), float(ty),
                             int(speed_i), int(damage_i), int(shooter_id))
                        )
                except Exception as e:
                    print(f"[P2P] shot unpack error: {e}")
                continue

            # --- Legacy / position-only packet ---
            if len(data) >= 8:
                try:
                    rx, ry = struct.unpack_from('2f', data, 0)
                    with state_lock:
                        remote_state['x'] = float(rx)
                        remote_state['y'] = float(ry)
                except Exception as e:
                    print(f"[P2P] unpack error: {e}")
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[P2P] err: {e}")
            time.sleep(0.1)

#
# --- Face sprite receive loop for P2P ---
def face_recv_loop(sock):
    """Receive opponent face sprite (PNG-encoded RGBA) over UDP.

    Protocol (chunked):
    - Image is PNG-encoded and split into chunks of up to CHUNK_SIZE bytes.
    - Each chunk packet:
      b'FIMG' + uint32(frame_id) + uint16(total_chunks) + uint16(chunk_idx) + payload
    - When all chunks of a frame are received, reassemble and decode PNG.
    - ACK:
      When a frame is successfully decoded, send:
      b'FACE_ACK' + uint32(frame_id)
    - Sender stops retransmitting when it sees FACE_ACK.

    For backward compatibility:
    - If data starts with b'FACE_ACK', it is treated purely as an ACK.
    - If data does not start with b'FIMG' or b'FACE_ACK', it is treated as a
      single-packet PNG (legacy behavior).
    """
    global remote_face_sprite, face_recv_frames
    CHUNK_HDR_MAGIC = b"FIMG"
    ACK_MAGIC = b"FACE_ACK"

    while True:
        try:
            data, addr = sock.recvfrom(65535)
            if not data:
                continue

            # ACK packet: b"FACE_ACK" [+ optional 4-byte frame_id]
            if data.startswith(ACK_MAGIC):
                try:
                    # frame_id は今のところ使わないが、将来の拡張に備えて読むだけ読む
                    if len(data) >= len(ACK_MAGIC) + 4:
                        _ = struct.unpack("!I", data[len(ACK_MAGIC):len(ACK_MAGIC)+4])[0]
                    face_ack_event.set()
                except Exception:
                    # ACK のパース失敗は致命的ではないので無視
                    pass
                continue

            # Chunked image packet: b"FIMG" + header + payload
            if data.startswith(CHUNK_HDR_MAGIC) and len(data) > 12:
                try:
                    # header: magic(4) + frame_id(4) + total(2) + idx(2)
                    _, frame_id, total, idx = struct.unpack("!4sIHH", data[:12])
                    payload = data[12:]

                    entry = face_recv_frames.get(frame_id)
                    if entry is None:
                        entry = {
                            "total": total,
                            "chunks": {},
                            "from_addr": addr,
                        }
                        face_recv_frames[frame_id] = entry

                    # チャンクを保存（同じ idx が再送されてきた場合は上書き）
                    entry["chunks"][idx] = payload

                    # 全チャンク揃ったか？
                    if len(entry["chunks"]) >= entry["total"]:
                        # idx=0..total-1 の順に結合
                        try:
                            buf = b"".join(entry["chunks"][i] for i in range(entry["total"]))
                        except KeyError:
                            # 何らかの理由で欠損していた場合は破棄
                            del face_recv_frames[frame_id]
                            continue

                        # PNG デコード
                        nparr = np.frombuffer(buf, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                        del face_recv_frames[frame_id]

                        if img is None:
                            continue

                        # Ensure 4-channel RGBA for consistency
                        if img.ndim == 3 and img.shape[2] == 3:
                            h, w = img.shape[:2]
                            a = np.full((h, w, 1), 255, dtype=np.uint8)
                            img = np.concatenate([img, a], axis=2)

                        if img.ndim == 3 and img.shape[2] == 4:
                            with remote_face_lock:
                                remote_face_sprite = img

                            # Send ACK with frame_id back to sender
                            try:
                                ack_pkt = ACK_MAGIC + struct.pack("!I", frame_id)
                                sock.sendto(ack_pkt, addr)
                            except Exception as e2:
                                print(f"[P2P_FACE] ack send failed: {e2}")
                except Exception as e:
                    print(f"[P2P_FACE] chunk decode failed: {e}")
                continue

            # --- Legacy single-packet PNG (no header) ---
            try:
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            except Exception:
                img = None

            if img is None:
                continue
            # Ensure 4-channel RGBA
            if img.ndim == 3 and img.shape[2] == 3:
                h, w = img.shape[:2]
                a = np.full((h, w, 1), 255, dtype=np.uint8)
                img = np.concatenate([img, a], axis=2)

            if img.ndim == 3 and img.shape[2] == 4:
                with remote_face_lock:
                    remote_face_sprite = img
                # Legacy モードでは frame_id が無いので、ACK は簡単な固定文字列だけ送る
                try:
                    sock.sendto(ACK_MAGIC, addr)
                except Exception as e2:
                    print(f"[P2P_FACE] ack(send legacy) failed: {e2}")
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[P2P_FACE] err: {e}")
            time.sleep(0.1)


#
# --- Face sprite send loop for P2P ---
def face_send_loop(sock, peer_addr, sprite, interval=0.5, max_retry=20):
    """
    Send the local face sprite (PNG-encoded RGBA) over UDP to the peer.

    - PNG にエンコードした後、CHUNK_SIZE ごとに分割して送信。
    - 各パケットの形式:
      b'FIMG' + uint32(frame_id) + uint16(total_chunks) + uint16(chunk_idx) + payload
    - 受信側が全チャンク復元＆デコードに成功すると、
      b'FACE_ACK' + uint32(frame_id) を返す。
    - ここでは、その ACK を受け取るまで、一定間隔ごとに全チャンクを再送する。
    """
    if sprite is None or sock is None or peer_addr is None:
        return

    # 1. Encode to PNG
    try:
        ret, buf = cv2.imencode('.png', sprite)
        if not ret or buf is None:
            print("[P2P_FACE] encode failed: imencode returned False")
            return
        data = buf.tobytes()
    except Exception as e:
        print(f"[P2P_FACE] encode failed: {e}")
        return

    # 2. Prepare chunked packets
    CHUNK_SIZE = 1300  # UDP-safe payload size
    total = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
    if total <= 0:
        return

    frame_id = next_face_frame_id()
    packets = []
    magic = b"FIMG"
    for idx in range(total):
        start = idx * CHUNK_SIZE
        end = min(len(data), start + CHUNK_SIZE)
        payload = data[start:end]
        header = struct.pack("!4sIHH", magic, frame_id, total, idx)
        packets.append(header + payload)

    # 3. Reset ACK event and retransmit until ACK or max_retry
    face_ack_event.clear()
    retries = 0
    while retries < max_retry and not face_ack_event.is_set():
        for pkt in packets:
            try:
                sock.sendto(pkt, peer_addr)
            except Exception as e:
                print(f"[P2P_FACE] send failed: {e}")
        retries += 1
        time.sleep(interval)

# ----------------- ゲーム状態 -----------------
class PlayerBullet:
    """プレイヤー弾。
    - 口の開き (aspect) から決まるダメージ・速度を持つ
    - 発射地点から照準へ向けて直線移動（速度には依存せず、一定時間で到達）
    - 発射地点から離れるほど小さく・暗く描画される（描画側で t を参照）
    """
    def __init__(self, ox, oy, tx, ty, speed, damage,
                 r0=100.0, r_min=20.0,
                 hit_window_px=20.0, aim_radius_px=50.0):
        # 発射地点
        self.ox = float(ox)
        self.oy = float(oy)
        # 現在位置（初期は発射地点）
        self.x = float(ox)
        self.y = float(oy)
        # 照準位置
        self.tx = float(tx)
        self.ty = float(ty)
        # 速度・ダメージ（整数）: パラメータとして保持（互換性のため）
        self.speed = int(speed)
        self.dmg = int(damage)

        # 描画用の半径
        self.r0 = float(r0)
        self.r_min = float(r_min)
        self.r = self.r0

        # 照準座標と当たり判定ウィンドウ（既存ロジックとの互換のため残す）
        self.ax = float(tx)
        self.ay = float(ty)
        self.hit_window_px = float(hit_window_px)
        self.aim_radius_px = float(aim_radius_px)

        # 時間ベースの進行度（0〜1）と移動時間
        # 発射位置→終点までの移動を travel_time 秒で完了させる（speed には依存しない）
        self.t = 0.0
        self.travel_time = 0.6  # 0.6秒で終点に到達する（必要に応じて調整）

        # TTL は travel_time + α（保険）
        self.ttl = self.travel_time + 0.2

    def step(self, dt):
        # 時間ベースで進行度 t を更新（0〜1）
        if self.travel_time > 1e-6:
            self.t += dt / self.travel_time
        else:
            self.t = 1.0
        if self.t >= 1.0:
            self.t = 1.0
            self.ttl = 0.0
        else:
            self.ttl -= dt

        # 発射位置→終点を線形補間（speed には依存しない）
        self.x = (1.0 - self.t) * self.ox + self.t * self.tx
        self.y = (1.0 - self.t) * self.oy + self.t * self.ty

        # 半径：発射時 r0 → 終点で r_min まで線形に縮小（t にのみ依存）
        self.r = max(self.r_min, self.r0 * (1.0 - self.t))

class EnemyBullet:
    """敵弾：画面中心へ向けて曲線移動＋手前表現の半径拡大。
    中心から半径 cross_radius (=30) を通過した瞬間のベクトルを保持し、それ以降は
    そのベクトルのまま直進（ステアリングしない）。
    """
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        # 奥行き（最初は敵と同じレイヤーからスタートし、プレイヤーに近づくほど手前へ）
        self.z = Z_ENEMY
        # --- 深度表現（手前へ近づく） ---
        self.r = 6.0                 # 初期半径
        self.growth = 10.0           # 成長速度 [px/s]
        self.accel = 10.0            # 成長加速度 [px/s^2]
        self.impact_r = 44.0         # 着弾半径
        # --- 平面移動（中心へ曲線） ---
        self.speed = 80.0            # 初速 [px/s]
        self.speed_accel = 120.0     # 加速度 [px/s^2]
        self.steer = 0.15            # ステアリング強度（0..1）
        self.curve = random.uniform(0.25, 0.60) * (1 if random.random() < 0.5 else -1)
        self.ttl = 6.0               # 保険寿命
        # 進行方向と中心通過管理
        self.dirx = None
        self.diry = None
        self.cross_radius = 30.0
        self.passed_center = False

    def step(self, dt, player_world_x, player_world_y):
        # 視差（半径）成長
        self.growth += self.accel * dt
        self.r += self.growth * dt
        # 平面速度成長
        self.speed += self.speed_accel * dt
        # 見かけ上の奥行き：impact_r に近づくほどプレイヤー側の Z_PLAYER へ補間
        ratio = float(np.clip(self.r / max(1e-6, self.impact_r), 0.0, 1.0))
        self.z = Z_ENEMY * (1.0 - ratio) + Z_PLAYER * ratio

        # プレイヤーのワールド中心座標に向かう
        cx, cy = float(player_world_x), float(player_world_y)
        vx = cx - self.x
        vy = cy - self.y
        n = math.hypot(vx, vy)

        # 初回フレームで方向未設定なら中心方向へ初期化
        if self.dirx is None or self.diry is None:
            if n > 1e-6:
                self.dirx, self.diry = vx / n, vy / n
            else:
                self.dirx, self.diry = 1.0, 0.0

        if not self.passed_center:
            # ステアリング：中心方向＋垂直成分（曲率）を目標に、現在向きとブレンド
            if n > 1e-6:
                ux, uy = vx / n, vy / n
                # 左に90度回転の単位ベクトル
                px, py = -uy, ux
                # 遠いほど曲げる、近いほど直進
                dist_factor = min(1.0, n / (0.5 * max(SCR_W, SCR_H)))
                ex = ux + self.curve * dist_factor * px
                ey = uy + self.curve * dist_factor * py
                en = math.hypot(ex, ey)
                if en > 1e-6:
                    ex, ey = ex / en, ey / en
                else:
                    ex, ey = ux, uy
            else:
                # ほぼ中心：向きを維持
                ex, ey = self.dirx, self.diry

            # 現在向きと目標向きを補間
            dirx = (1.0 - self.steer) * self.dirx + self.steer * ex
            diry = (1.0 - self.steer) * self.diry + self.steer * ey
            dn = math.hypot(dirx, diry)
            if dn > 1e-6:
                self.dirx, self.diry = dirx / dn, diry / dn

            # ここで中心半径を通過したかを判定し、通過したらベクトルを固定
            if n <= self.cross_radius:
                self.passed_center = True
                # self.dirx, self.diry はこの時点の単位ベクトルのまま固定
        else:
            # すでに通過済み：ベクトル固定で直進（ステアリングなし）
            pass

        # 位置更新
        self.x += self.dirx * self.speed * dt
        self.y += self.diry * self.speed * dt

        self.ttl -= dt

class RemotePlayerBullet:
    """相手プレイヤーから飛んでくる弾（P2P 受信した SHOT イベントを元に生成）。
    - 発射地点 (ox, oy) から「受信した照準位置 (tx, ty)」まで直線移動する（速度や距離には依存せず、一定時間で到達）。
    - 移動の進行度に応じて、半径を線形に成長させる。
    - 当たり判定ロジックは今後 EnemyBullet と同様の「到達時着弾」に拡張予定。
    """
    def __init__(self, ox, oy, tx, ty, speed, damage):
        # 発射地点・目標地点
        self.ox = float(ox)
        self.oy = float(oy)
        self.tx = float(tx)
        self.ty = float(ty)

        # 現在位置（初期は発射地点）
        self.x = float(ox)
        self.y = float(oy)

        # 奥行きレイヤー（最初は敵と同じ中間）
        self.z = Z_ENEMY

        # 速度・ダメージ（将来的に利用するため保持しておくが、移動には使わない）
        self.speed = float(speed)
        self.dmg = int(damage)

        # 半径：発射時は小さく、照準到達時に最大になるよう線形補間
        self.r_start = 6.0      # 発射直後の半径
        self.r_end   = 60.0     # 照準到達時の半径
        self.r = self.r_start
        # impact_r も r_end に合わせておく（当たり判定互換用）
        self.impact_r = self.r_end

        # 発射地点→照準位置までの進行度（0〜1）。描画やZ補間に利用。
        self.t = 0.0
        # 発射位置→終点まで travel_time 秒で到達させる（speed には依存しない）
        self.travel_time = 0.6
        # TTL は travel_time + α とする（保険）
        self.ttl = self.travel_time + 0.2

    def step(self, dt, player_world_x, player_world_y):
        """dt 秒だけ前進し、発射地点→照準位置までの進行度に応じて半径を線形に成長させる。
        進行度は speed ではなく「経過時間 / travel_time」で決まる。
        """
        # 進行度を時間ベースで更新
        if self.travel_time > 1e-6:
            self.t += dt / self.travel_time
        else:
            self.t = 1.0

        if self.t >= 1.0:
            self.t = 1.0
            self.ttl = 0.0
        else:
            self.ttl -= dt

        # 発射位置→照準位置を線形補間（speed や距離には直接依存しない）
        self.x = (1.0 - self.t) * self.ox + self.t * self.tx
        self.y = (1.0 - self.t) * self.oy + self.t * self.ty

        # 半径を線形に補間（発射時 r_start → 到達時 r_end）
        self.r = self.r_start + (self.r_end - self.r_start) * self.t
        # impact_r も r_end に合わせておく（当たり判定互換用）
        self.impact_r = self.r_end

        # 奥行きもプレイヤー側に寄ってくるよう線形補間
        self.z = Z_ENEMY * (1.0 - self.t) + Z_PLAYER * self.t
# --- Insert: OpponentPlayer class after Target ---

class Target:
    def __init__(self):
        self.r = random.randint(18, 28)
        self.hp_max = random.randint(1, 15)
        self.hp = float(self.hp_max)
        # 初期位置はスクリーン内上側あたりのワールド座標（従来と同じレンジ）
        self.x = random.uniform(self.r + 10, SCR_W - self.r - 10)
        self.y = random.uniform(self.r + 10, SCR_H * 0.5)
        # 擬似3D空間での奥行きレイヤー：敵は中間レイヤーに固定
        self.z = Z_ENEMY
        ang = random.uniform(0, 2*math.pi)
        spd = random.uniform(40, 120)
        self.vx = math.cos(ang) * spd
        self.vy = math.sin(ang) * spd * 0.6
    def step(self, dt):
        self.x += self.vx*dt
        self.y += self.vy*dt
        # 壁反射
        if self.x < self.r or self.x > SCR_W-self.r: self.vx *= -1; self.x = np.clip(self.x, self.r, SCR_W-self.r)
        if self.y < self.r or self.y > SCR_H*0.85:   self.vy *= -1; self.y = np.clip(self.y, self.r, SCR_H*0.85)

# --- Insert: OpponentPlayer class after Target ---

class OpponentPlayer:
    """Placeholder for future networked opponent.
    現状は見た目だけのダミー：奥(Z_OPPONENT)レイヤーに描画する。
    """
    def __init__(self, x, y, radius=26.0):
        self.x = float(x)
        self.y = float(y)
        self.z = Z_OPPONENT
        self.r = float(radius)
        # vx/vy for movement (future use)
        self.vx = 0.0
        self.vy = 0.0
    def step(self, dt):
        # Move by current velocity (if any)
        self.x += self.vx * dt
        self.y += self.vy * dt
        # Clamp to world bounds (X) and a reasonable Y band
        self.x = float(np.clip(self.x, self.r, WORLD_W - self.r))
        y_min = SCR_H * 0.10
        y_max = SCR_H * 0.70
        self.y = float(np.clip(self.y, y_min, y_max))

# --- Field item spawned from defeated enemies ---
class ItemPickup:
    """Field item spawned from defeated enemies.

    kind: 0=HP potion, 1=Attack potion, 2=Spread armor, 3=Beam armor
    """
    def __init__(self, x, y, kind, sprite):
        self.x = float(x)
        self.y = float(y)
        self.kind = int(kind)
        self.sprite = sprite
        # simple bobbing animation
        self.phase = random.uniform(0.0, 2.0 * math.pi)

    def step(self, dt):
        self.phase += dt * 2.0

    def get_draw_pos(self):
        """Return (x, y) used for rendering / hit detection with a small bob."""
        bob = math.sin(self.phase) * 6.0
        return self.x, self.y + bob

def norm_to_px(xn, yn):
    return int(xn * (SCR_W-1)), int(yn * (SCR_H-1))

def compute_bullet_params_from_aspect(aspect_raw: float):
    """口の縦横比(aspect)から、プレイヤー弾のダメージと速度を決定する。"""
    a = float(np.clip(aspect_raw, ASPECT_MIN, ASPECT_MAX))
    # ダメージは aspect に比例
    damage_f = a * DAMAGE_CONST
    # 速度は 1/aspect に比例（大きく開くほど遅いが重い弾）
    speed_f = (1.0 / a) * SPEED_CONST

    damage_i = int(np.clip(round(damage_f), DAMAGE_MIN, DAMAGE_MAX))
    speed_i  = int(np.clip(round(speed_f),  SPEED_MIN,  SPEED_MAX))
    return damage_i, speed_i, a

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
    # どれくらいパースを強くするか（0.0〜1.0程度で調整）
    DEPTH_STRENGTH = 0.7

    # z に応じたスケール（z=0 → 1.0, z が大きいほど縮小）
    depth_scale = 1.0 / (1.0 + float(z) * DEPTH_STRENGTH)

    # まず通常のカメラ変換（ワールド→画面座標系）
    sx = (wx - cam_x)
    sy = (wy - cam_y)

    # スケールを掛けつつ、画面中心に向かって寄せる
    sx = sx * depth_scale + SCR_W * 0.5 * (1.0 - depth_scale)
    sy = sy * depth_scale + SCR_H * 0.5 * (1.0 - depth_scale)

    return int(sx), int(sy), depth_scale

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



# アルファ合成でサークル描画
def draw_alpha_circle(img, center, radius, color, alpha, thickness=-1):
    """Draw a circle with per-call alpha onto img."""
    if alpha <= 0.0 or radius <= 0:
        return
    overlay = img.copy()
    cv2.circle(overlay, center, int(radius), color, thickness)
    cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0, img)

# アルファ合成で楕円描画（影用）
def draw_alpha_ellipse(img, center, axes, angle_deg, color, alpha, thickness=-1):
    """Draw an ellipse with per-call alpha onto img.
    center: (x, y)
    axes: (axis_x, axis_y)  ※OpenCVのellipseは半径指定
    angle_deg: 回転角（度）
    """
    if alpha <= 0.0 or axes[0] <= 0 or axes[1] <= 0:
        return
    overlay = img.copy()
    cv2.ellipse(overlay,
                (int(center[0]), int(center[1])),
                (int(axes[0]), int(axes[1])),
                float(angle_deg), 0, 360,
                color, thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0, img)


# アルファ合成でポリライン描画
def draw_alpha_polyline(img, pts, color, alpha, thickness=1):
    if alpha <= 0.0 or len(pts) < 2:
        return
    overlay = img.copy()
    cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0, img)

# アルファ合成でライン描画（ビーム影用）
def draw_alpha_line(img, p1, p2, color, alpha, thickness=1):
    if alpha <= 0.0:
        return
    overlay = img.copy()
    cv2.line(overlay, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, int(max(1, thickness)), cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0, img)

def draw_player_bullet(img, bullet, cam_x, cam_y):
    """プレイヤー弾を青い円として描画する。
    発射地点から離れるほど小さく・暗くなる。
    """
    # ワールド座標 → 画面座標
    sx, sy = world_to_screen(bullet.x, bullet.y, cam_x, cam_y)

    # 画面外は描画しない（少しマージンを持たせる）
    if sx < -50 or sx > SCR_W + 50 or sy < -50 or sy > SCR_H + 50:
        return

    # 進行度 t に応じて明るさを変化させる
    t = float(np.clip(getattr(bullet, "t", 0.0), 0.0, 1.0))
    max_brightness = 1.0
    min_brightness = 0.3  # 完全に真っ暗にはしない
    brightness = max_brightness * (1.0 - t) + min_brightness * t

    # ベース色は青 (BGR)
    base_color = (255, 0, 0)
    color = tuple(int(c * brightness) for c in base_color)

    cv2.circle(
        img,
        (int(sx), int(sy)),
        int(max(1.0, bullet.r)),
        color,
        thickness=-1,
        lineType=cv2.LINE_AA
    )
    # 外側の縁取り（少し大きい半径で明るい色を描画）
    outline_radius = int(max(1.0, bullet.r) + 2)
    outline_color = (255, 255, 255)  # 白いアウトライン
    cv2.circle(
        img,
        (int(sx), int(sy)),
        outline_radius,
        outline_color,
        thickness=2,
        lineType=cv2.LINE_AA
    )

def draw_remote_player_bullet(img, bullet, cam_x, cam_y):
    """相手プレイヤー弾を黄色い円として描画する。
    RemotePlayerBullet.r をそのまま半径として使う。
    """
    # ワールド座標 → 画面座標
    sx, sy = world_to_screen(bullet.x, bullet.y, cam_x, cam_y)

    # 画面外は描画しない（少しマージンを持たせる）
    if sx < -50 or sx > SCR_W + 50 or sy < -50 or sy > SCR_H + 50:
        return

    # ベース色は黄色 (BGR)
    base_color = (0, 255, 255)

    cv2.circle(
        img,
        (int(sx), int(sy)),
        int(max(1.0, bullet.r)),
        base_color,
        thickness=-1,
        lineType=cv2.LINE_AA
    )
    # 外側の縁取り（少し大きい半径で明るい色を描画）
    outline_radius = int(max(1.0, bullet.r) + 2)
    outline_color = (255, 255, 255)  # 白いアウトライン
    cv2.circle(
        img,
        (int(sx), int(sy)),
        outline_radius,
        outline_color,
        thickness=2,
        lineType=cv2.LINE_AA
    )

# --- RGBA image blending helpers (for parallax backgrounds) ---
def blend_rgba(dst_bgr, src_rgba, xoff=0, yoff=0, alpha_scale=1.0):
    """Alpha-blend an RGBA image onto a BGR canvas at integer offset.
    Supports partial overlaps. Does not tile automatically.
    """
    if src_rgba is None:
        return dst_bgr
    h, w = dst_bgr.shape[:2]
    sh, sw = src_rgba.shape[:2]
    x1 = max(0, xoff); y1 = max(0, yoff)
    x2 = min(w, xoff + sw); y2 = min(h, yoff + sh)
    if x1 >= x2 or y1 >= y2:
        return dst_bgr
    crop = src_rgba[(y1 - yoff):(y2 - yoff), (x1 - xoff):(x2 - xoff)]
    if crop.shape[2] < 4:
        # No alpha channel; treat as opaque
        dst_bgr[y1:y2, x1:x2] = crop[:, :, :3]
        return dst_bgr
    a = (crop[:, :, 3:4].astype(np.float32) / 255.0) * float(alpha_scale)
    roi = dst_bgr[y1:y2, x1:x2].astype(np.float32)
    dst_bgr[y1:y2, x1:x2] = (roi * (1 - a) + crop[:, :, :3].astype(np.float32) * a).astype(np.uint8)
    return dst_bgr

def blend_rgba_tiled_x(dst_bgr, src_rgba, xoff, yoff=0, alpha_scale=1.0):
    """Horizontally tile an RGBA layer once so scrolling wraps.
    Draws at xoff and xoff - sw to cover wrap-around.
    """
    if src_rgba is None:
        return dst_bgr
    sh, sw = src_rgba.shape[:2]
    xoff_mod = int(xoff) % sw
    dst_bgr = blend_rgba(dst_bgr, src_rgba, xoff=xoff_mod, yoff=yoff, alpha_scale=alpha_scale)
    dst_bgr = blend_rgba(dst_bgr, src_rgba, xoff=xoff_mod - sw, yoff=yoff, alpha_scale=alpha_scale)
    return dst_bgr

# ひび割れパターン生成
def make_crack(x, y, base_r=40, branches=8, seg_min=12, seg_max=28):
    """Create a crack decal composed of multiple jagged polylines.
    Returns a dict with geometry and lifetime for later rendering.
    """
    lines = []
    cx, cy = float(x), float(y)
    for b in range(branches):
        ang = random.uniform(0, 2*math.pi)
        nseg = random.randint(3, 6)
        pts = [(cx, cy)]
        seg_len = random.uniform(seg_min, seg_max)
        dirx, diry = math.cos(ang), math.sin(ang)
        px, py = cx, cy
        for i in range(nseg):
            # 進む＋ジッタでギザギザ
            jitter = random.uniform(-0.6, 0.6)
            jx = -diry * jitter * 6.0
            jy =  dirx * jitter * 6.0
            px += dirx * seg_len + jx
            py += diry * seg_len + jy
            pts.append((int(px), int(py)))
            # 方向を少し曲げる
            ang += random.uniform(-0.35, 0.35)
            dirx, diry = math.cos(ang), math.sin(ang)
            seg_len *= random.uniform(0.85, 1.10)
        lines.append(pts)
    return {
        "x": cx, "y": cy,
        "lines": lines,
        "life": 1.2,          # 表示寿命（秒）
        "max_life": 1.2,
    }

# --- 六角グリッドATフィールド風オーバーレイ生成 ---
def build_hex_overlay(w, h, cell=42, line_th=1, color=(0,165,255)):
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
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=line_th, lineType=cv2.LINE_AA)
            x += dx
        y += dy
        row += 1
    return img

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
    """Capture a single frame from the default camera and return it as an RGBA image.

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
    for i in range(warmup_N):
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
    args = parser.parse_args()

    # --- P2P setup ---
    p2p_sock = None
    p2p_peer_addr = None
    p2p_enabled = False
    remote_state = {"x": SCR_W * 0.7, "y": SCR_H * 0.3}
    p2p_lock = threading.Lock()
    p2p_face_sock = None
    p2p_face_peer_addr = None
    if args.peer is not None:
        peer_host, *peer_port_part = args.peer.split(':')
        if peer_port_part and peer_port_part[0].isdigit():
            peer_port = int(peer_port_part[0])
        else:
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
            threading.Thread(target=p2p_recv_loop, args=(p2p_sock, remote_state, p2p_lock), daemon=True).start()
            print(f"[P2P] listening on {args.listen_ip}:{args.listen_port}, peer={peer_host}:{peer_port}")
            # --- Setup P2P face sprite socket and receiver thread ---
            try:
                p2p_face_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                p2p_face_sock.settimeout(1.0)
                # Use a port range separate from the main P2P port to avoid collisions.
                # Example: if listen_port=6000, face_listen_port=6100.
                face_port_offset = 100
                face_listen_port = args.listen_port + face_port_offset
                try:
                    p2p_face_sock.bind((args.listen_ip, face_listen_port))
                except Exception as e:
                    print(f"[P2P_FACE] WARNING: failed to bind {args.listen_ip}:{face_listen_port}: {e}")
                # The peer's face port is computed with the same offset from its base P2P port.
                p2p_face_peer_addr = (peer_host, peer_port + face_port_offset)
                threading.Thread(target=face_recv_loop, args=(p2p_face_sock,), daemon=True).start()
                print(f"[P2P_FACE] listening on {args.listen_ip}:{face_listen_port}, peer={peer_host}:{peer_port + face_port_offset}")
            except Exception as e:
                print(f"[P2P_FACE] setup failed: {e}")
        except Exception as e:
            print(f"[P2P] setup failed: {e}")

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
    th = threading.Thread(target=udp_loop, args=(args.gaze_listen_ip, args.gaze_port), daemon=True)
    th.start()

    # ワールド
    bullets = []
    targets = [Target() for _ in range(6)]
    # 将来のPVP対戦相手（いまはダミーで「現在の画面内」に1体配置）
    # cam_x=0, cam_y=0 でスタートなので、スクリーン座標と同じ値をそのままワールド座標として使う
    opponent = OpponentPlayer(SCR_W * 0.7, SCR_H * 0.3)
    score = 0
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
    # --- Item system (3 slots bottom-left, stack-based, currently empty) ---
    item_slots = [[], [], []]  # each slot is a stack (list); items not yet implemented
    item_selected = 0          # start at leftmost; 0: left, 1: middle, 2: right
    last_item_move_t = 0.0
    prev_item_use = False
    item_blink_until = 0.0

    player_hp_max = 3
    player_hp = player_hp_max
    ATTACK_BUFF_DURATION = 8.0
    attack_buff_mul = 1.0
    attack_buff_timer = 0.0
    shield_hp_max = 2     # シールドHPの上限
    shield_hp = shield_hp_max
    shield_on = False
    eye_closed_count = 0
    shield_grace = 0

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
                target=face_send_loop,
                args=(p2p_face_sock, p2p_face_peer_addr, local_face_sprite),
                daemon=True
            ).start()
        except Exception as e:
            print(f"[P2P_FACE] send thread start failed: {e}")

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

    # --- Sliding window for one-eye ratio (~0.2 sec at current FPS) ---
    one_eye_window_N = max(1, int(FPS * ONE_EYE_TO_DUAL_DELAY))
    one_eye_buf = deque(maxlen=one_eye_window_N)
    one_eye_sum = 0

    # 直近で片目だった時刻（片目→FPSへの復帰遅延に使用）
    last_one_eye_time = 0.0

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

        # === Setup screen: hold the game until calibration is done ===
        with remote_face_lock:
            opp_face = remote_face_sprite.copy() if remote_face_sprite is not None else None

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
        if not calib_done:
            img = np.zeros((SCR_H, SCR_W, 3), np.uint8)
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

        # --- One-eye sliding window update (0.2秒間での片目率を計測) ---
        if len(one_eye_buf) == one_eye_buf.maxlen:
            oldest = one_eye_buf[0]
            one_eye_sum -= 1 if oldest else 0
        one_eye_buf.append(one_eye_now)
        one_eye_sum += 1 if one_eye_now else 0
        one_eye_ratio_ok = (
            len(one_eye_buf) == one_eye_buf.maxlen and
            (one_eye_sum / float(one_eye_buf.maxlen)) >= ONE_EYE_WINDOW_RATIO
        )

        # --- 片目モード遅延制御 ---
        # now_t はこの少し前で time.time() が代入済み（シールドロジックで使用）
        if one_eye_now:
            # 現在も片目状態なら、その時刻を記録
            last_one_eye_time = now_t

        # one_eye_mode 決定:
        # 1) 現在片目状態 → 片目モード
        # 2) 両目OPENでも
        #    「直前0.2秒のうち9割以上が片目」かつ
        #    「最後に片目だった時刻から0.2秒以内」
        #    なら片目モード維持
        if one_eye_now:
            one_eye_mode = True
        elif eye_open_now and (now_t - last_one_eye_time < ONE_EYE_TO_DUAL_DELAY) and one_eye_ratio_ok:
            one_eye_mode = True
        else:
            one_eye_mode = False

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
                # 1: Attack potion -> temporary attack power up
                elif kind == 1:
                    attack_buff_mul = 1.7
                    attack_buff_timer = ATTACK_BUFF_DURATION
                # 2: Spread armor -> clear all enemy bullets
                elif kind == 2:
                    if enemy_bullets:
                        for eb in enemy_bullets:
                            for _ in range(10):
                                ang = random.uniform(0, 2 * math.pi)
                                spd = random.uniform(220, 520)
                                life = random.uniform(0.05, 0.12)
                                particles.append({
                                    "x": float(eb.x), "y": float(eb.y),
                                    "vx": math.cos(ang) * spd,
                                    "vy": math.sin(ang) * spd,
                                    "life": life
                                })
                        score += 20 * len(enemy_bullets)
                        enemy_bullets = []
                # 3: Beam armor -> heavy damage to all enemies on screen
                elif kind == 3:
                    for tg in targets:
                        tg.hp -= max(1.0, tg.hp_max * 0.75)
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

        # 攻撃アップの残り時間を更新
        if attack_buff_timer > 0.0:
            attack_buff_timer -= dt
            if attack_buff_timer <= 0.0:
                attack_buff_timer = 0.0
                attack_buff_mul = 1.0

        # --- P2P: 相手から届いた SHOT イベントを RemotePlayerBullet に変換 ---
        if p2p_enabled:
            with remote_shot_lock:
                events = list(remote_shot_events)
                remote_shot_events.clear()
            for ox, oy, tx, ty, speed_i, damage_i, shooter_id in events:
                # speed_i / damage_i は送信側でクリップ済みだが、念のため下限だけ軽く守る
                spd = max(float(SPEED_MIN), float(speed_i))
                dmg = int(damage_i)

                # 発射位置は「相手プレイヤーが今いるワールド座標」に置き換える
                with p2p_lock:
                    opp_x = float(remote_state.get("x", SCR_W * 0.7))
                    opp_y = float(remote_state.get("y", SCR_H * 0.3))

                remote_player_bullets.append(
                    RemotePlayerBullet(
                        ox=opp_x,
                        oy=opp_y,
                        tx=tx,
                        ty=ty,
                        speed=spd,
                        damage=dmg,
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
            aspect_raw = float(latest.get("aspect", latest.get("mar", 0.0)))
            damage_i, speed_i, _ = compute_bullet_params_from_aspect(aspect_raw)
            # 攻撃アップバフを乗算
            damage_i = int(damage_i * attack_buff_mul)

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
                            '!4sffffhhB',
                            b'SHOT',
                            float(muzzle_x), float(muzzle_y),
                            float(target_wx), float(target_wy),
                            int(speed_i), int(damage_i),
                            0,  # shooter_id（暫定で0。将来ホスト/クライアントで振り分け可能）
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

        # 敵弾スポーン（だんだん近づいてくる）
        enemy_spawn_timer += dt
        if enemy_spawn_timer >= enemy_spawn_period and len(targets) > 0:
            enemy_spawn_timer = 0.0
            # どれか1体からプレイヤー（現在の照準位置）へ向けて発射
            tg = random.choice(targets)
            enemy_bullets.append(EnemyBullet(tg.x, tg.y))
            # 次回までの間隔に少しランダム性
            enemy_spawn_period = random.uniform(0.8, 1.4)

        # プレイヤーのワールド中心座標（カメラ位置＋画面中心）
        player_world_x = cam_x + SCR_W * 0.5
        player_world_y = cam_y + SCR_H * 0.5
        # --- P2P send local player position ---
        if p2p_enabled and p2p_sock is not None and p2p_peer_addr is not None:
            try:
                pkt = struct.pack('2f', float(player_world_x), float(player_world_y))
                p2p_sock.sendto(pkt, p2p_peer_addr)
            except Exception as e:
                # non-fatal; just log once in a while if needed
                # print(f"[P2P] send err: {e}")
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

        # 相手プレイヤー弾更新（ワールド外・寿命切れを捨てる）
        kept_remote = []
        for rb in remote_player_bullets:
            rb.step(dt, player_world_x, player_world_y)
            # 寿命切れ
            if rb.ttl <= 0:
                continue
            # ワールド外（少しマージンを持たせる）
            if rb.x < -100 or rb.x > WORLD_W + 100 or rb.y < -100 or rb.y > WORLD_H + 100:
                continue
            kept_remote.append(rb)
        remote_player_bullets = kept_remote

        def _apply_bullet_hit_like_enemy(bullet):
            nonlocal player_hp, shield_hp, shield_on, score, shield_gen_channel, shield_fx_start
            # 中身は今の enemy_bullets の hit 処理と同じ

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

        # 当たり判定（円 vs 点/小円）
        hit_idx = []
        for i, tg in enumerate(targets):
            for b in bullets:
                # 照準付近のみ当たり判定：
                # 1) 弾が照準直前（残距離 <= hit_window_px）で、
                # 2) ターゲットが照準中心から aim_radius_px 以内
                rem = max(0.0, 1.0 - float(getattr(b, "t", 0.0)))
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
        for i in sorted(hit_idx, reverse=True):
            tg = targets[i]
            # 5% chance to spawn a field item at the defeated enemy position
            if random.random() < ITEM_DROP_PROB and item_world_sprites:
                kind = random.randint(0, 3)
                spr = item_world_sprites[kind]
                if spr is not None:
                    items.append(ItemPickup(tg.x, tg.y, kind, spr))
            targets.pop(i)
            targets.append(Target())
            if snd_enemy_appearance is not None:
                snd_enemy_appearance.play()

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
            # ワールド中心座標 → 擬似3D投影（画面座標）。位置だけ奥行きで動かし、サイズは固定。
            tx_s, ty_s, _ = project_to_screen(tg.x, tg.y, tg.z, cam_x, cam_y)
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
            draw_player_bullet(img, b, cam_x, cam_y)

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

        # 対戦相手プレイヤー（将来のPVP用ダミー）を奥レイヤーに描画
        if opponent is not None:
            # 位置は z によるパララックスを反映するが、サイズは z ではスケーリングしない
            ox_s, oy_s, _ = project_to_screen(opponent.x, opponent.y, opponent.z, cam_x, cam_y)
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
        for c in cracks:
            a = max(0.0, min(1.0, c["life"] / c["max_life"]))
            # 外側ほど薄くなるよう、線を2層（濃→薄）で重ねる
            for pts in c["lines"]:
                scr_pts = []
                for (px, py) in pts:
                    sx_p, sy_p = world_to_screen(px, py, cam_x, cam_y)
                    scr_pts.append((sx_p, sy_p))
                draw_alpha_polyline(img, scr_pts, (210, 220, 255), 0.45 * a, thickness=2)
                draw_alpha_polyline(img, scr_pts, (140, 160, 200), 0.25 * a, thickness=1)
            # 中心の擦りキズ
            cx_s, cy_s = world_to_screen(c["x"], c["y"], cam_x, cam_y)
            draw_alpha_circle(img, (int(cx_s), int(cy_s)), 10, (220, 230, 255), 0.20 * a, thickness=2)

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

        # Debug: show received mouth-shape features (aspect/norm_w) and relative expression values
        cv2.putText(img, f"shape: aspect={latest.get('aspect',0.0):.2f} norm_w={latest.get('norm_w',0.0):.2f}",
                    (SCR_W-360, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,220,255), 2)
        if calib_done:
            cv2.putText(img,
                        f"rel: smile={rel_smile:.2f} frown={rel_frown:.2f} brow_h={rel_browh:.2f}",
                        (SCR_W-360, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,200,255), 2)
        # HUD
        cv2.putText(img, f"Score: {score}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,255,200), 2)
        cv2.putText(img, f"HP: {player_hp}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,220,255), 2)
        cv2.putText(img, f"Shield HP: {shield_hp}/{shield_hp_max}", (12, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,255,255), 2)
        cv2.putText(img, f"Shield: {'ON' if shield_on else 'OFF'}", (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160,240,255), 2)
        cv2.putText(img, "Weapon: BULLET", (12, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,180), 2)
        cv2.putText(img, f"Calib: {'OK' if calib_done else f'{calib_count}/{calib_N}'}  C=({calib_cx:.2f},{calib_cy:.2f})  gain={gain_xy:.1f}x",
                    (12, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,220), 2)
        if calib_done and (th_open_L > 0.0 or th_open_R > 0.0):
            cv2.putText(img, f"EAR thr  L[o>={th_open_L:.2f}/c<={th_close_L:.2f}]  R[o>={th_open_R:.2f}/c<={th_close_R:.2f}]",
                        (12, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,230), 2)
        # --- FPS座標とワールド座標のデバッグ表示 ---
        # ワールド座標は「画面中心のワールド位置」とし、cam_x, cam_y だけで決まる
        player_world_x = cam_x + SCR_W * 0.5
        player_world_y = cam_y + SCR_H * 0.5
        cv2.putText(img,
                    f"FPS=({fps_x},{fps_y}) PlayerWorld=({player_world_x:.1f},{player_world_y:.1f})",
                    (12, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,220,200), 2)
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

        # 相手プレイヤー弾（P2P SHOT）の描画（黄色）
        # 先に発射された弾が「上」に来るように、古い弾ほど後から描画する
        for rb in reversed(remote_player_bullets):
            draw_remote_player_bullet(img, rb, cam_x, cam_y)
            
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
            y_min = SCR_H * 0.10
            y_max = SCR_H * 0.70
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
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()