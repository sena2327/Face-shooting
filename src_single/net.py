import socket
import struct
import threading
import time

import numpy as np

# UDP 受信設定（detect_face.py 側と揃える）
HOST = "0.0.0.0"
PORT = 5005

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

def udp_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    sock.settimeout(1.0)
    print(f"[UDP] listening on {HOST}:{PORT}")
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