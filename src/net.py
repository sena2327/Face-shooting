# net.py - networking and shared state for pvp_shooter

import socket
import struct
import threading
import time

from collections import deque

import numpy as np
import cv2

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
# tuple layout: (ox, oy, tx, ty, speed_i, damage_i, shooter_id, shooter_level)
remote_shot_events = deque()
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
            # Format: b'SHOT' + struct('!ffffhhBB')
            #   ox, oy, tx, ty, speed_i, damage_i, shooter_id, shooter_level
            if len(data) >= 25 and data[:4] == b'SHOT':
                shooter_level = 1
                try:
                    if len(data) >= 26:
                        _, ox, oy, tx, ty, speed_i, damage_i, shooter_id, shooter_level = struct.unpack('!4sffffhhBB', data[:26])
                    else:
                        raise struct.error
                except Exception:
                    try:
                        _, ox, oy, tx, ty, speed_i, damage_i, shooter_id = struct.unpack('!4sffffhhB', data[:25])
                        shooter_level = 1
                    except Exception as e:
                        print(f"[P2P] shot unpack error: {e}")
                        continue

                print(f"[P2P-SHOT-RECV] ox={ox:.1f} oy={oy:.1f}  tx={tx:.1f} ty={ty:.1f}  speed={speed_i} dmg={damage_i} shooter={shooter_id} lvl={shooter_level}")

                with remote_shot_lock:
                    remote_shot_events.append(
                        (float(ox), float(oy), float(tx), float(ty),
                         int(speed_i), int(damage_i), int(shooter_id), int(shooter_level))
                    )
                continue

            # --- Legacy / position-only packet ---
            if len(data) >= 12:
                try:
                    rx, ry, rhp = struct.unpack_from('3f', data, 0)
                    with state_lock:
                        remote_state['x'] = float(rx)
                        remote_state['y'] = float(ry)
                        remote_state['hp'] = float(rhp)
                    continue
                except Exception:
                    pass
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
