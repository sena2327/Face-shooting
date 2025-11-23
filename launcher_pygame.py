import sys
import subprocess
from pathlib import Path
import random
import math
import pygame
import cv2
import numpy as np

import test_face_overlay

# --- 設定 & 定数 (ANIME COSMIC THEME) ---
SCREEN_WIDTH = 1000  # 横並びのため少し広く
SCREEN_HEIGHT = 700

# Colors (Anime/Cyber Style)
COLOR_BG = (10, 10, 25)           # Dark Indigo
COLOR_GRID = (20, 0, 40)          # Dark Purple
COLOR_HUD_MAIN = (0, 255, 255)    # Cyan
COLOR_HUD_ACCENT = (255, 0, 150)  # Magenta (Anime Accent)
COLOR_HUD_DIM = (0, 80, 80)
COLOR_ALERT = (255, 50, 50)
COLOR_TEXT_GLOW = (200, 255, 255)
COLOR_PANEL_BG = (0, 0, 20, 200)  # Semi-transparent dark

# --- ランチャー機能 ---
def launch_main(args: list[str]) -> None:
    base_dir = Path(__file__).resolve().parent
    main_script = base_dir / "main.py"
    cmd = [sys.executable, str(main_script)] + args
    print(f"[SYSTEM]: LAUNCHING... {' '.join(cmd)}")
    pygame.quit()
    subprocess.Popen(cmd)
    sys.exit(0)

# --- MediaPipe / Camera クラス ---
class FaceMonitor:
    def __init__(self):
        self.monitor_size = (320, 320)
        self.surface = pygame.Surface(self.monitor_size)
        self.scan_line_y = 0
        self.overlay_ctx = test_face_overlay.create_overlay_context()
        self.face_detection = self.overlay_ctx is not None

    def update(self):
        if not self.overlay_ctx:
            self._draw_no_signal()
            return

        composite_img = test_face_overlay.get_composited_frame(
            self.overlay_ctx["cap"],
            self.overlay_ctx["face_mesh"],
            self.overlay_ctx["base_rgba"],
            self.overlay_ctx["anchor_pts"],
            self.overlay_ctx["face_oval_idx"],
        )

        if composite_img is None:
            self._draw_no_signal()
            return

        display_img = cv2.resize(composite_img, self.monitor_size, interpolation=cv2.INTER_AREA)

        # OpenCV(numpy) -> Pygame Surface（横向きにならないようにシンプルに変換）
        # OpenCV は (H, W, 3)、surfarray は (W, H, 3) を想定するので軸だけ入れ替える
        display_img = np.transpose(display_img, (1, 0, 2))
        surf = pygame.surfarray.make_surface(display_img)
        surf = pygame.transform.scale(surf, self.monitor_size)

        # エフェクト：走査線
        self.surface.blit(surf, (0, 0))
        
        # スキャンラインアニメーション
        self.scan_line_y = (self.scan_line_y + 5) % self.monitor_size[1]
        pygame.draw.line(self.surface, COLOR_HUD_MAIN, (0, self.scan_line_y), (self.monitor_size[0], self.scan_line_y), 2)
        
        # グリッドオーバーレイ
        for y in range(0, self.monitor_size[1], 40):
            pygame.draw.line(self.surface, (0, 50, 50), (0, y), (self.monitor_size[0], y), 1)

    def _draw_no_signal(self):
        self.surface.fill((10, 10, 10))
        font = pygame.font.SysFont(None, 40)
        text = font.render("NO SIGNAL", True, COLOR_ALERT)
        rect = text.get_rect(center=(self.monitor_size[0]//2, self.monitor_size[1]//2))
        self.surface.blit(text, rect)

    def draw(self, screen, center_pos):
        # 枠の描画
        rect = self.surface.get_rect(center=center_pos)
        screen.blit(self.surface, rect)
        
        # コックピット風の枠飾り
        ox, oy = rect.topleft
        w, h = rect.size
        
        # メイン枠
        pygame.draw.rect(screen, COLOR_HUD_MAIN, rect, 2)
        
        # 四隅の装飾
        corn_len = 30
        # 左上
        pygame.draw.line(screen, COLOR_HUD_ACCENT, (ox-5, oy), (ox+corn_len, oy), 4)
        pygame.draw.line(screen, COLOR_HUD_ACCENT, (ox, oy-5), (ox, oy+corn_len), 4)
        # 右下
        pygame.draw.line(screen, COLOR_HUD_ACCENT, (ox+w-corn_len, oy+h), (ox+w+5, oy+h), 4)
        pygame.draw.line(screen, COLOR_HUD_ACCENT, (ox+w, oy+h-corn_len), (ox+w, oy+h+5), 4)

        # ラベル
        font = pygame.font.SysFont("consolas", 16, bold=True)
        text = font.render("PILOT: DETECTED", True, COLOR_HUD_MAIN)
        screen.blit(text, (ox, oy - 20))

    def release(self):
        if self.overlay_ctx:
            test_face_overlay.release_overlay_context(self.overlay_ctx)
            self.overlay_ctx = None

# --- ビジュアルエフェクトクラス ---

class StarParticle:
    """集中線のような効果を出す高速パーティクル"""
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.reset()

    def reset(self):
        # 画面端からスポーン
        if random.choice([True, False]):
            self.x = random.choice([-10, self.w + 10])
            self.y = random.randint(0, self.h)
        else:
            self.x = random.randint(0, self.w)
            self.y = random.choice([-10, self.h + 10])
            
        self.speed = random.uniform(5, 15)
        # 画面中心に向かうベクトル
        self.target_x = self.w // 2
        self.target_y = self.h // 2
        
        angle = math.atan2(self.target_y - self.y, self.target_x - self.x)
        self.vx = math.cos(angle) * self.speed
        self.vy = math.sin(angle) * self.speed
        self.life = random.randint(20, 50)
        self.color = random.choice([COLOR_HUD_MAIN, COLOR_HUD_ACCENT, (255, 255, 255)])

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        
        # 中心に近づきすぎたらリセット
        dist = math.hypot(self.x - self.target_x, self.y - self.target_y)
        if dist < 50 or self.life <= 0:
            self.reset()

    def draw(self, screen):
        # 流線形に描画
        end_x = self.x - self.vx * 2
        end_y = self.y - self.vy * 2
        pygame.draw.line(screen, self.color, (self.x, self.y), (end_x, end_y), 2)

# --- 描画ユーティリティ ---

def draw_neon_text(screen, font, text, pos, color, align="center"):
    x, y = pos
    
    # Glow effect (重ね書き)
    glow_surf = font.render(text, True, color)
    glow_surf.set_alpha(50)
    rect = glow_surf.get_rect()
    
    if align == "center":
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)

    # 少しずらしてぼかし効果風
    for offset in [(-2, -2), (2, 2), (-2, 2), (2, -2)]:
        screen.blit(glow_surf, (rect.x + offset[0], rect.y + offset[1]))
    
    # Main text
    main_surf = font.render(text, True, (255, 255, 255)) # 白い芯
    screen.blit(main_surf, rect)
    
    # Color overlay
    color_surf = font.render(text, True, color)
    color_surf.set_alpha(150)
    screen.blit(color_surf, rect)
    
    return rect

def draw_tech_circle(screen, center, radius, color, timer):
    """回転する魔法陣/HUDリング"""
    pygame.draw.circle(screen, color, center, radius, 1)
    
    # 回転する弧
    start_angle = timer * 2
    end_angle = start_angle + math.pi / 2
    rect = (center[0]-radius, center[1]-radius, radius*2, radius*2)
    pygame.draw.arc(screen, color, rect, start_angle, end_angle, 3)
    pygame.draw.arc(screen, color, rect, start_angle + math.pi, end_angle + math.pi, 3)

# --- メインループ ---

def main():
    pygame.init()
    pygame.display.set_caption("COSMIC LINK SYSTEM // ANIME VER.")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    # --- フォント設定 ---
    def get_font(size):
        fonts = ["impact", "arial black", "noto sans cjk jp", "ms gothic"]
        for f in fonts:
            if f in pygame.font.get_fonts():
                return pygame.font.SysFont(f, size)
        return pygame.font.SysFont(None, size)

    font_title = get_font(60)
    font_btn = get_font(24)
    font_sub = get_font(16)

    # --- メニュー項目（横並び用） ---
    menu_items = [
        {"label": "STORY MODE", "cmd": []},
        {"label": "VERSUS", "cmd": ["--pvp"]},
        {"label": "TRAINING", "cmd": ["--local"]},
        {"label": "EXIT", "cmd": None},
    ]
    selected_idx = 0

    # --- オブジェクト生成 ---
    face_monitor = FaceMonitor()
    particles = [StarParticle(SCREEN_WIDTH, SCREEN_HEIGHT) for _ in range(30)]
    
    timer = 0.0
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        timer += dt
        
        # MediaPipe更新
        face_monitor.update()

        # --- イベント処理 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    selected_idx = (selected_idx - 1) % len(menu_items)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    selected_idx = (selected_idx + 1) % len(menu_items)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    item = menu_items[selected_idx]
                    if item["cmd"] is None:
                        running = False
                    else:
                        face_monitor.release()
                        launch_main(item["cmd"])

        # --- 描画更新 ---
        screen.fill(COLOR_BG)

        # 1. 背景効果（グリッド）
        # 遠近法のあるグリッド（地面）
        for i in range(10):
            y = SCREEN_HEIGHT // 2 + int(math.pow(i/10, 2) * (SCREEN_HEIGHT // 2))
            pygame.draw.line(screen, COLOR_GRID, (0, y), (SCREEN_WIDTH, y), 1)
        
        # 集中線パーティクル
        for p in particles:
            p.update()
            p.draw(screen)

        # 2. 中央：顔モニター表示
        monitor_center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50)
        
        # モニターの背後の回転リング演出
        draw_tech_circle(screen, monitor_center, 200, COLOR_HUD_DIM, timer)
        draw_tech_circle(screen, monitor_center, 180, COLOR_HUD_ACCENT, -timer * 1.5)

        # 顔画像の描画
        face_monitor.draw(screen, monitor_center)

        # 3. タイトル (上部)
        title_y = 60
        draw_neon_text(screen, font_title, "Face Shooter", (SCREEN_WIDTH//2, title_y), COLOR_HUD_MAIN)
        draw_neon_text(screen, font_sub, "- NEURAL SYNC ONLINE -", (SCREEN_WIDTH//2, title_y + 40), COLOR_HUD_ACCENT)

        # 4. ボタン（横並び）
        # ボタンエリアの計算
        btn_area_y = SCREEN_HEIGHT - 120
        btn_width = 200
        btn_height = 50
        gap = 20
        total_width = len(menu_items) * btn_width + (len(menu_items) - 1) * gap
        start_x = (SCREEN_WIDTH - total_width) // 2

        for i, item in enumerate(menu_items):
            bx = start_x + i * (btn_width + gap)
            by = btn_area_y
            rect = pygame.Rect(bx, by, btn_width, btn_height)
            is_selected = (i == selected_idx)

            # ベース色
            base_col = COLOR_HUD_ACCENT if is_selected else COLOR_HUD_DIM
            
            # ボタン背景 (平行四辺形っぽく描画してアニメ感を出す)
            skew = 10
            pts = [
                (bx + skew, by), 
                (bx + btn_width + skew, by), 
                (bx + btn_width - skew, by + btn_height), 
                (bx - skew, by + btn_height)
            ]
            
            # 塗りつぶし
            if is_selected:
                pygame.draw.polygon(screen, base_col, pts) # 明るい
                # 枠線発光
                pygame.draw.lines(screen, (255, 255, 255), True, pts, 2)
            else:
                # 非選択は枠のみ
                pygame.draw.polygon(screen, (0, 0, 0, 100), pts) # 暗い背景
                pygame.draw.lines(screen, base_col, True, pts, 1)

            # テキスト
            text_col = (255, 255, 255) if is_selected else (150, 150, 150)
            text_surf = font_btn.render(item["label"], True, text_col)
            text_rect = text_surf.get_rect(center=(bx + btn_width//2, by + btn_height//2))
            screen.blit(text_surf, text_rect)

            # 選択中のカーソル演出（三角形）
            if is_selected:
                tri_y = by - 15 + math.sin(timer * 10) * 5
                tri_x = bx + btn_width // 2
                pygame.draw.polygon(screen, COLOR_HUD_MAIN, [
                    (tri_x, tri_y + 10),
                    (tri_x - 10, tri_y),
                    (tri_x + 10, tri_y)
                ])

        # 5. フッター
        pygame.draw.rect(screen, (0, 0, 0), (0, SCREEN_HEIGHT-30, SCREEN_WIDTH, 30))
        info_text = f"CPU: NORMAL | MEMORY: OPTIMAL | FACE TRACKING: {'ACTIVE' if face_monitor.face_detection else 'OFFLINE'}"
        draw_neon_text(screen, font_sub, info_text, (SCREEN_WIDTH//2, SCREEN_HEIGHT-15), COLOR_HUD_DIM)

        pygame.display.flip()

    # 終了処理
    face_monitor.release()
    pygame.quit()

if __name__ == "__main__":
    main()
