# src/game_objects.py
import math
import random
import numpy as np

# pvp_shooter の定数と合わせる（画面サイズに依存するところだけ使う）
SCR_W, SCR_H = 960, 540
WORLD_W, WORLD_H = SCR_W * 10, SCR_H * 10

Z_PLAYER   = 0.0
Z_ENEMY    = 0.5
Z_OPPONENT = 1.0


class PlayerBullet:
    """プレイヤー弾。
    - 口の開き (aspect) から決まるダメージ・速度を持つ
    - 発射地点から照準へ向けて直線移動（速度には依存せず、一定時間で到達）
    - 発射地点から離れるほど小さく・暗く描画される（描画側で t を参照）
    """
    def __init__(self, ox, oy, tx, ty, speed, damage,
                 level=1,
                 r0=100.0, r_min=20.0,
                 hit_window_px=40.0, aim_radius_px=50.0):
        self.ox = float(ox)
        self.oy = float(oy)
        self.x = float(ox)
        self.y = float(oy)
        self.tx = float(tx)
        self.ty = float(ty)
        self.speed = int(speed)
        self.dmg = int(damage)

        self.r0 = float(r0)
        self.r_min = float(r_min)
        self.r = self.r0

        self.ax = float(tx)
        self.ay = float(ty)
        self.hit_window_px = float(hit_window_px)
        self.aim_radius_px = float(aim_radius_px)

        self.level = max(1, int(level))
        self.t = 0.0
        self.travel_time = max(0.4, 0.6 - 0.05 * (self.level - 1))
        self.ttl = self.travel_time + 0.2

    def step(self, dt):
        if self.travel_time > 1e-6:
            self.t += dt / self.travel_time
        else:
            self.t = 1.0
        if self.t >= 1.0:
            self.t = 1.0
            self.ttl = 0.0
        else:
            self.ttl -= dt

        self.x = (1.0 - self.t) * self.ox + self.t * self.tx
        self.y = (1.0 - self.t) * self.oy + self.t * self.ty
        self.r = max(self.r_min, self.r0 * (1.0 - self.t))


class EnemyBullet:
    """敵弾：画面中心へ向けて曲線移動＋手前表現の半径拡大。"""
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.z = Z_ENEMY

        self.r = 6.0
        self.growth = 10.0
        self.accel = 10.0
        self.impact_r = 44.0

        self.speed = 80.0
        self.speed_accel = 120.0
        self.steer = 0.15
        self.curve = random.uniform(0.25, 0.60) * (1 if random.random() < 0.5 else -1)
        self.ttl = 6.0

        self.dirx = None
        self.diry = None
        self.cross_radius = 30.0
        self.passed_center = False

    def step(self, dt, player_world_x, player_world_y):
        self.growth += self.accel * dt
        self.r += self.growth * dt

        self.speed += self.speed_accel * dt

        ratio = float(np.clip(self.r / max(1e-6, self.impact_r), 0.0, 1.0))
        self.z = Z_ENEMY * (1.0 - ratio) + Z_PLAYER * ratio

        cx, cy = float(player_world_x), float(player_world_y)
        vx = cx - self.x
        vy = cy - self.y
        n = math.hypot(vx, vy)

        if self.dirx is None or self.diry is None:
            if n > 1e-6:
                self.dirx, self.diry = vx / n, vy / n
            else:
                self.dirx, self.diry = 1.0, 0.0

        if not self.passed_center:
            if n > 1e-6:
                ux, uy = vx / n, vy / n
                px, py = -uy, ux
                dist_factor = min(1.0, n / (0.5 * max(SCR_W, SCR_H)))
                ex = ux + self.curve * dist_factor * px
                ey = uy + self.curve * dist_factor * py
                en = math.hypot(ex, ey)
                if en > 1e-6:
                    ex, ey = ex / en, ey / en
                else:
                    ex, ey = ux, uy
            else:
                ex, ey = self.dirx, self.diry

            dirx = (1.0 - self.steer) * self.dirx + self.steer * ex
            diry = (1.0 - self.steer) * self.diry + self.steer * ey
            dn = math.hypot(dirx, diry)
            if dn > 1e-6:
                self.dirx, self.diry = dirx / dn, diry / dn

            if n <= self.cross_radius:
                self.passed_center = True
        # else: keep going straight

        self.x += self.dirx * self.speed * dt
        self.y += self.diry * self.speed * dt
        self.ttl -= dt


class RemotePlayerBullet:
    """P2P 相手プレイヤーから飛んでくる弾。"""
    def __init__(self, ox, oy, tx, ty, speed, damage, level=1):
        self.ox = float(ox)
        self.oy = float(oy)
        self.tx = float(tx)
        self.ty = float(ty)

        self.x = float(ox)
        self.y = float(oy)

        self.z = Z_ENEMY
        self.speed = float(speed)
        self.dmg = int(damage)
        self.level = max(1, int(level))

        self.r_start = 6.0
        self.r_end   = 60.0
        self.r = self.r_start
        self.impact_r = self.r_end

        self.t = 0.0
        self.travel_time = max(0.4, 0.6 - 0.05 * (self.level - 1))
        self.ttl = self.travel_time + 0.2

    def step(self, dt, player_world_x, player_world_y):
        if self.travel_time > 1e-6:
            self.t += dt / self.travel_time
        else:
            self.t = 1.0

        if self.t >= 1.0:
            self.t = 1.0
            self.ttl = 0.0
        else:
            self.ttl -= dt

        self.x = (1.0 - self.t) * self.ox + self.t * self.tx
        self.y = (1.0 - self.t) * self.oy + self.t * self.ty

        self.r = self.r_start + (self.r_end - self.r_start) * self.t
        self.impact_r = self.r_end
        self.z = Z_ENEMY * (1.0 - self.t) + Z_PLAYER * self.t


class Target:
    def __init__(self, spawn_point_id=-1, seed=None, origin=None):
        rng = random.Random(seed)
        self.spawn_point_id = spawn_point_id
        self.spawn_seq = 0
        self.pending_kill = False
        self.r = rng.randint(18, 28)
        self.hp_max = rng.randint(1, 15)
        self.hp = float(self.hp_max)
        if origin is None:
            self.x = rng.uniform(self.r + 10, WORLD_W - self.r - 10)
            self.y = rng.uniform(self.r + 10, WORLD_H - self.r - 10)
        else:
            ox, oy = origin
            self.x = float(np.clip(ox + rng.uniform(-40, 40), self.r, WORLD_W - self.r))
            self.y = float(np.clip(oy + rng.uniform(-30, 30), self.r, WORLD_H - self.r))
        self.z = Z_ENEMY
        ang = rng.uniform(0, 2 * math.pi)
        spd = rng.uniform(40, 120)
        self.vx = math.cos(ang) * spd
        self.vy = math.sin(ang) * spd * 0.6

    def step(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.x < self.r or self.x > WORLD_W - self.r:
            self.vx *= -1
            self.x = np.clip(self.x, self.r, WORLD_W - self.r)
        if self.y < self.r or self.y > WORLD_H - self.r:
            self.vy *= -1
            self.y = np.clip(self.y, self.r, WORLD_H - self.r)


class OpponentPlayer:
    """将来のネット対戦相手用プレースホルダ。"""
    def __init__(self, x, y, radius=26.0):
        self.x = float(x)
        self.y = float(y)
        self.z = Z_OPPONENT
        self.r = float(radius)
        self.vx = 0.0
        self.vy = 0.0

    def step(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.x = float(np.clip(self.x, self.r, WORLD_W - self.r))
        y_min = self.r
        y_max = WORLD_H - self.r
        self.y = float(np.clip(self.y, y_min, y_max))


class ItemPickup:
    """敵撃破時に落ちるフィールドアイテム。

    kind: 0=HP potion, 1=Attack potion, 2=Spread armor, 3=Beam armor
    """
    def __init__(self, x, y, kind, sprite):
        self.x = float(x)
        self.y = float(y)
        self.kind = int(kind)
        self.sprite = sprite
        self.phase = random.uniform(0.0, 2.0 * math.pi)

    def step(self, dt):
        self.phase += dt * 2.0

    def get_draw_pos(self):
        bob = math.sin(self.phase) * 6.0
        return self.x, self.y + bob


def make_crack(x, y, base_r=40, branches=8, seg_min=12, seg_max=28):
    """ひび割れパターン生成（ジグザグ線の集合）。"""
    lines = []
    cx, cy = float(x), float(y)
    for _ in range(branches):
        ang = random.uniform(0, 2 * math.pi)
        nseg = random.randint(3, 6)
        pts = [(cx, cy)]
        seg_len = random.uniform(seg_min, seg_max)
        dirx, diry = math.cos(ang), math.sin(ang)
        px, py = cx, cy
        for _ in range(nseg):
            jitter = random.uniform(-0.6, 0.6)
            jx = -diry * jitter * 6.0
            jy = dirx * jitter * 6.0
            px += dirx * seg_len + jx
            py += diry * seg_len + jy
            pts.append((int(px), int(py)))
            ang += random.uniform(-0.35, 0.35)
            dirx, diry = math.cos(ang), math.sin(ang)
            seg_len *= random.uniform(0.85, 1.10)
        lines.append(pts)
    return {
        "x": cx,
        "y": cy,
        "lines": lines,
        "life": 1.2,
        "max_life": 1.2,
    }
