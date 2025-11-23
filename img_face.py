#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blue_cap.png 上でクリックした 3 点の座標を取得して blue_cap.csv に保存するツール。

使い方:
    python img_face.py

挙動:
    - img/blue_cap.png を読み込んでウィンドウに表示する
    - 左クリックするたびに座標 (x, y) を記録し、画像上に小さな円と番号を描画する
    - 3 点クリックしたら blue_cap.csv に座標を書き出し、自動的に終了する
        フォーマット:
            index,x,y
            0,123,45
            1, ...
            2, ...
"""

import os
import csv
import cv2

# 取得した点を格納するリスト
clicked_points = []  # list of (x, y)

WINDOW_NAME = "Click 3 points on blue_cap (left click)"
IMG_REL_PATH = os.path.join("img", "blue_cap.png")
CSV_OUT_PATH = "blue_cap.csv"


def mouse_callback(event, x, y, flags, param):
    global clicked_points
    img = param["img"]
    disp = param["disp"]

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) >= 3:
            # すでに 3 点とっていたら無視
            return

        # 座標を保存
        clicked_points.append((x, y))
        idx = len(clicked_points) - 1

        # 見やすいように画像に描画（小さな円 + インデックス番号）
        cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)  # 赤い点
        cv2.putText(
            disp,
            str(idx),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, disp)

        if len(clicked_points) == 3:
            print("3 点クリックされました。CSV に保存して終了します。")
            save_points_to_csv(clicked_points, CSV_OUT_PATH)
            print(f"保存先: {os.path.abspath(CSV_OUT_PATH)}")
            cv2.destroyAllWindows()


def save_points_to_csv(points, csv_path):
    """points: list of (x, y) を CSV に保存する"""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x", "y"])
        for idx, (x, y) in enumerate(points):
            writer.writerow([idx, x, y])


def main():
    # スクリプトのディレクトリを基準にパスを解決
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, IMG_REL_PATH)
    csv_path = os.path.join(base_dir, CSV_OUT_PATH)

    if not os.path.exists(img_path):
        print(f"[ERROR] 画像ファイルが見つかりません: {img_path}")
        return

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[ERROR] 画像の読み込みに失敗しました: {img_path}")
        return

    # 表示用のコピー（クリック位置や番号を書き込むため）
    disp = img.copy()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    params = {"img": img, "disp": disp}
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, param=params)

    print("画像が表示されます。左クリックで 3 点を順に指定してください。")
    print("ウィンドウを閉じたい場合は ESC キーを押してください。")

    while True:
        cv2.imshow(WINDOW_NAME, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            print("ESC が押されたので終了します。（クリック点は保存されません）")
            break
        # 3 点クリックし終えると、mouse_callback 内で destroyAllWindows() が呼ばれているので、
        # ここまで来ないでループは終了するはず

        # 念のため、3 点そろっていてウィンドウが閉じられていたら抜ける
        if len(clicked_points) >= 3 and cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
