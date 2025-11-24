# main.py
"""
detect_face と single_shooter を同時に起動するエントリポイント。

これまで手動で実行していた

    python3 detect_face.py --flip --udp --udp-host 127.0.0.1 --udp-port 5005
    python single_shooter.py

を、1 本のスクリプトからまとめて起動する。
"""

import subprocess
import sys
import signal
import time
import argparse
from pathlib import Path


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    # --- コマンドライン引数 ---
    parser = argparse.ArgumentParser(
        description="detect_face と shooter(single/pvp) をまとめて起動するランチャー"
    )
    parser.add_argument(
        "--pvp",
        action="store_true",
        help="pvp_shooter.py を起動する（指定なしなら single_shooter.py）",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="ローカルで 2 プレイヤーを起動（pvp 用）。detect_face + pvp_shooter を 2 セット起動する。",
    )
    parser.add_argument(
        "--base-gaze-port",
        type=int,
        default=5005,
        help="detect_face からの UDP 受信ポートのベース番号（デフォルト: 5005）",
    )
    parser.add_argument(
        "--base-listen-port",
        type=int,
        default=6000,
        help="pvp_shooter の P2P リッスンポートのベース番号（デフォルト: 6000）",
    )
    parser.add_argument(
        "--player-id",
        type=int,
        choices=(1, 2),
        default=1,
        help="PVP プレイ時のプレイヤー番号（1=ホスト/2=クライアント）。--local では無視される。",
    )
    parser.add_argument(
        "--peer",
        type=str,
        default=None,
        help="PVP 相手の host:port（player_id=2 では必須）",
    )
    args = parser.parse_args()

    detect_script = base_dir / "detect_face.py"
    single_script = base_dir / "single_shooter.py"
    pvp_script = base_dir / "pvp_shooter.py"

    # プロセスハンドルをまとめて管理
    detect_procs: list[subprocess.Popen] = []
    shooter_procs: list[subprocess.Popen] = []

    # --- ローカル 2 人 PVP モード ---
    if args.local:
        if not args.pvp:
            print("[main] --local が指定されたので PVP モードを有効化します (--pvp を暗黙に有効)")
        # gaze 用ポート（2人分）
        gaze_port1 = args.base_gaze_port
        gaze_port2 = args.base_gaze_port + 1
        # P2P リッスンポート（2人分）
        listen_port1 = args.base_listen_port
        listen_port2 = args.base_listen_port + 1

        # detect_face プレイヤー1
        detect_cmd1 = [
            sys.executable,
            str(detect_script),
            "--flip",
            "--udp",
            "--udp-host",
            "127.0.0.1",
            "--udp-port",
            str(gaze_port1),
        ]
        # detect_face プレイヤー2
        detect_cmd2 = [
            sys.executable,
            str(detect_script),
            "--flip",
            "--udp",
            "--udp-host",
            "127.0.0.1",
            "--udp-port",
            str(gaze_port2),
        ]

        print(f"[main] starting detect_face #1 (port={gaze_port1}) ...")
        detect_procs.append(subprocess.Popen(detect_cmd1))
        print(f"[main] starting detect_face #2 (port={gaze_port2}) ...")
        detect_procs.append(subprocess.Popen(detect_cmd2))

        # detect_face の起動が安定するまで少し待つ
        time.sleep(2.0)

        # pvp_shooter プレイヤー1
        pvp_cmd1 = [
            sys.executable,
            str(pvp_script),
            "--gaze-port",
            str(gaze_port1),
            "--listen-port",
            str(listen_port1),
            "--peer",
            f"127.0.0.1:{listen_port2}",
            "--capture-opponent",
            "--player-id",
            "1",
        ]
        # pvp_shooter プレイヤー2
        pvp_cmd2 = [
            sys.executable,
            str(pvp_script),
            "--gaze-port",
            str(gaze_port2),
            "--listen-port",
            str(listen_port2),
            "--peer",
            f"127.0.0.1:{listen_port1}",
            "--capture-opponent",
            "--player-id",
            "2",
        ]

        try:
            print(f"[main] starting pvp_shooter #1 (gaze={gaze_port1}, listen={listen_port1}) ...")
            shooter_procs.append(subprocess.Popen(pvp_cmd1))
            print(f"[main] starting pvp_shooter #2 (gaze={gaze_port2}, listen={listen_port2}) ...")
            shooter_procs.append(subprocess.Popen(pvp_cmd2))

            # どちらか片方が終わるまで待つ（両方待ちたい場合は両方 wait する）
            for proc in shooter_procs:
                proc.wait()
        except KeyboardInterrupt:
            print("[main] KeyboardInterrupt, shutting down (local PVP)...")
        finally:
            # shooter 側の終了処理
            for proc in shooter_procs:
                if proc.poll() is None:
                    try:
                        proc.send_signal(signal.SIGINT)
                    except Exception:
                        pass
            # detect_face 側の終了処理
            for proc in detect_procs:
                if proc.poll() is None:
                    try:
                        print("[main] terminating detect_face.py ...")
                        proc.send_signal(signal.SIGINT)
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=3.0)
                    except Exception:
                        proc.terminate()
                        try:
                            proc.wait(timeout=2.0)
                        except Exception:
                            proc.kill()

        print("[main] done (local PVP).")
        return

    # --- 通常モード（単体 or 1人PVP） ---
    # detect_face 用ポートは base_gaze_port のみ
    gaze_port = args.base_gaze_port
    detect_cmd = [
        sys.executable,
        str(detect_script),
        "--flip",
        "--udp",
        "--udp-host",
        "127.0.0.1",
        "--udp-port",
        str(gaze_port),
    ]

    if args.pvp:
        listen_port = args.base_listen_port
        if args.player_id == 2 and not args.peer:
            raise SystemExit("--player-id 2 で起動する場合は --peer host:port を指定してください。")
        shooter_cmd = [
            sys.executable,
            str(pvp_script),
            "--gaze-port",
            str(gaze_port),
            "--listen-port",
            str(listen_port),
            "--player-id",
            str(args.player_id),
            "--capture-opponent",
        ]
        if args.peer:
            shooter_cmd += ["--peer", args.peer]
    else:
        shooter_cmd = [sys.executable, str(single_script)]

    print(f"[main] starting detect_face.py (port={gaze_port}) ...")
    detect_proc = subprocess.Popen(detect_cmd)

    # detect_face の起動が安定するまで少し待つ
    time.sleep(2.0)

    try:
        if args.pvp:
            print("[main] starting pvp_shooter.py ...")
        else:
            print("[main] starting single_shooter.py ...")
        shooter_proc = subprocess.Popen(shooter_cmd)

        # shooter が終わるのを待つ
        shooter_proc.wait()
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt, shutting down...")
    finally:
        # shooter 終了 or Ctrl+C で detect_face を終了させる
        if detect_proc.poll() is None:
            try:
                print("[main] terminating detect_face.py ...")
                detect_proc.send_signal(signal.SIGINT)
            except Exception:
                pass

            try:
                detect_proc.wait(timeout=3.0)
            except Exception:
                detect_proc.terminate()
                try:
                    detect_proc.wait(timeout=2.0)
                except Exception:
                    detect_proc.kill()

    print("[main] done.")


if __name__ == "__main__":
    main()
