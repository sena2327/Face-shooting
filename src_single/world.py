# src_single/world.py
import math
import random
import numpy as np

MAX_ENEMY_BULLETS = 15

# 擬似3D用Zレイヤー
Z_PLAYER = 0.0   # 手前（自分）
Z_ENEMY  = 0.5   # 中間（敵モンスター）

class Beam:
    def __init__(self, lines, ttl=0.12, dmg=3.0, aim=(0,0), radius=42.0, dps=None, z=None):
        self.lines = lines  # list of (x1,y1,x2,y2)
        self.ttl = float(ttl)
        # Interpret dmg as base magnitude; define DPS for continuous hit
        self.dmg = float(dmg)
        self.dps = float(dps) if dps is not None else float(dmg * 6.0)  # default: 6x per second
        self.aim = (float(aim[0]), float(aim[1]))
        self.radius = float(radius)
        # 擬似3D用の深度（デフォルトはプレイヤーと敵の中間あたり）
        self.z = float(z) if z is not None else (Z_PLAYER * 0.5 + Z_ENEMY * 0.5)
    def step(self, dt):
        self.ttl -= dt


# ----------------- ゲーム状態 -----------------
class Bullet:
    """プレイヤー弾：発射口→照準までの距離に合わせて半径を線形に縮小し、
    ちょうど照準位置で消える（ほぼ0サイズ）ようにする。
    また、当たり判定は照準付近のみ有効。
    """
    def __init__(self, x, y, vx, vy, dmg, dist_total, r0=6.0, r_min=0.8, ttl=None,
                 aim_x=None, aim_y=None, hit_window_px=20.0, aim_radius_px=36.0):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.dmg = dmg
        self.dist_total = max(1e-3, float(dist_total))  # 発射口→照準の距離
        self.dist_travel = 0.0
        self.r0 = float(r0)
        self.r_min = float(r_min)
        self.r = self.r0
        # 照準座標と当たり判定ウィンドウ
        self.ax = float(aim_x) if aim_x is not None else x
        self.ay = float(aim_y) if aim_y is not None else y
        self.hit_window_px = float(hit_window_px)  # 照準直前の残距離でのみ当たり判定
        self.aim_radius_px = float(aim_radius_px)  # 照準中心周囲の有効半径
        # TTL は保険（到達時に自動で0にする）。未指定なら距離/速さ+少し。
        speed = math.hypot(vx, vy)
        self.ttl = (dist_total / speed + 0.2) if (ttl is None and speed > 1e-6) else (ttl if ttl is not None else 1.5)

    def step(self, dt):
        dx = self.vx * dt
        dy = self.vy * dt
        self.x += dx
        self.y += dy
        self.dist_travel += math.hypot(dx, dy)
        t = min(1.0, self.dist_travel / self.dist_total)  # 0→1
        # 半径：発射時 r0 → 照準で r_min まで線形に縮小
        self.r = max(self.r_min, self.r0 * (1.0 - t))
        # 照準に到達（または超過）したら消滅
        if self.dist_travel >= self.dist_total:
            self.ttl = 0.0
        else:
            self.ttl -= dt

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
        self.ttl = 3.0               # 保険寿命
        # 進行方向と中心通過管理
        self.dirx = None
        self.diry = None
        self.cross_radius = 30.0
        self.passed_center = False
        # Straight shot support (boss bullets)
        self.straight_mode = False
        self.target_time = None
        self.travel_time = 0.0
        self.r_base = self.r

    def set_straight_line(self, target_pos, target_time=2.5):
        px, py = target_pos
        dx = px - self.x
        dy = py - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            dx, dy = 1.0, 0.0
            dist = 1.0
        self.dirx = dx / dist
        self.diry = dy / dist
        self.target_time = max(1e-3, target_time)
        self.speed = dist / self.target_time
        self.speed_accel = 0.0
        self.growth = (self.impact_r - self.r_base) / self.target_time if self.target_time > 0 else 0.0
        self.accel = 0.0
        self.straight_mode = True
        self.travel_time = 0.0
        self.ttl = self.target_time + 0.1
        self.passed_center = True

    def step(self, dt, player_world_x, player_world_y, scr_w, scr_h):
        if self.straight_mode:
            self.x += self.dirx * self.speed * dt
            self.y += self.diry * self.speed * dt
            self.travel_time += dt
            ratio = min(1.0, self.travel_time / (self.target_time or 1.0))
            self.r = self.r_base + (self.impact_r - self.r_base) * ratio
            self.ttl -= dt
            return
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
                dist_factor = min(1.0, n / (0.5 * max(scr_w, scr_h)))
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
class Target:
    def __init__(self, scr_w, scr_h):
        self.scr_w = scr_w
        self.scr_h = scr_h
        self.r = random.randint(18, 28)
        self.hp_max = random.randint(1, 3)
        self.hp = float(self.hp_max)
        # 初期位置はスクリーン内上側あたりのワールド座標
        self.x = random.uniform(self.r + 10, scr_w - self.r - 10)
        self.y = random.uniform(self.r + 10, scr_h * 0.5)
        # 擬似3D空間での奥行きレイヤー：敵は中間レイヤーに固定
        self.z = Z_ENEMY
        ang = random.uniform(0, 2 * math.pi)
        spd = random.uniform(40, 120)
        self.vx = math.cos(ang) * spd
        self.vy = math.sin(ang) * spd * 0.6
    def step(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        # 壁反射
        if self.x < self.r or self.x > self.scr_w - self.r:
            self.vx *= -1
            self.x = float(np.clip(self.x, self.r, self.scr_w - self.r))
        if self.y < self.r or self.y > self.scr_h * 0.85:
            self.vy *= -1
            self.y = float(np.clip(self.y, self.r, self.scr_h * 0.85))


class Boss(Target):
    """High-end boss with multi-phase bullet patterns."""
    def __init__(self, world_w, world_h):
        super().__init__(world_w, world_h)
        self.scr_w = world_w
        self.scr_h = world_h
        self.hp_max = 10.0
        self.hp = self.hp_max
        self.r = 80.0
        self.base_x = world_w * 0.5
        self.base_y = world_h * 0.35
        self.move_amp_x = world_w * 0.18
        self.move_amp_y = world_h * 0.06
        self.motion_t = 0.0
        self.osc_speed = 0.6
        self.attack_timer = 0.0
        self.is_boss = True
        self.has_entered_screen = False

    def phase_index(self):
        ratio = float(self.hp / max(1e-6, self.hp_max))
        if ratio > 0.70:
            return 0
        if ratio > 0.30:
            return 1
        return 2

    def attack_interval(self):
        phase = self.phase_index()
        intervals = (0.25, 0.18, 0.09)
        return intervals[min(phase, len(intervals) - 1)]

    def step(self, dt):
        phase = self.phase_index()
        if phase == 0:
            self.x = self.base_x
            self.y = self.base_y
            return

        self.motion_t += dt
        if phase == 1:
            speed_mul = 1.0
            amp_x = self.move_amp_x
            amp_y = self.move_amp_y
        else:  # phase 2
            speed_mul = 1.6
            amp_x = self.move_amp_x * 1.4
            amp_y = self.move_amp_y * 1.4
        osc = self.osc_speed * speed_mul
        self.x = self.base_x + math.sin(self.motion_t * osc) * amp_x
        self.y = self.base_y + math.sin(self.motion_t * (osc * 1.7) + phase * 0.4) * amp_y
        self.z = Z_ENEMY * 0.4

    def step_ai(self, dt, player_pos, bullet_list, now_t, visible_on_screen=False):
        if not self.has_entered_screen:
            if visible_on_screen:
                self.has_entered_screen = True
            else:
                return
        self.attack_timer -= dt
        if self.attack_timer > 0.0:
            return
        self.attack_timer = self.attack_interval()
        phase = self.phase_index()
        if phase == 0:
            shots = 3
        elif phase == 1:
            shots = 4
        else:
            shots = 5
        for _ in range(shots):
            self._spawn_direct_bullet(bullet_list, player_pos)
        if phase == 2:
            self._spawn_direct_bullet(bullet_list, player_pos)

    def _spawn_direct_bullet(self, bullet_list, player_pos, target_time=0.8):
        if len(bullet_list) >= MAX_ENEMY_BULLETS:
            return
        eb = EnemyBullet(self.x, self.y)
        eb.set_straight_line(player_pos, target_time=target_time)
        bullet_list.append(eb)


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
