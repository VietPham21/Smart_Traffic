"""Separated Visual Entities for SmartTraffic RL Studio (Pygame 60 FPS).

Strictly decoupled from MDP core logic: reads only the `info` telemetry dict
and visualizes:
1. VisualIntersection: Top-down 4-way intersection with asphalt, dashed divider lines,
   stop lines, pedestrian crosswalks, medians, and landscaped corner scenery.
2. SensorOverlay: 4 translucent green sensor zones ahead of the stop lines displaying
   live queue count `q=X | w=Ys`. Turns amber/red when wait time >= 40s (starvation risk).
3. TrafficLightWidget: 4 physical signal poles with full 3-bulb Red-Yellow-Green housings
   driven by info['light_states'] + smooth 3s yellow clearance transition + tau_t timer.
4. VisualVehicle & VisualTrafficScene: Smooth 60 FPS vehicle animation synced purely to
   info['raw_queues'] and info['step_passed_vehicles'] with diverse vehicle classes
   (car/van/truck), headway collision avoidance, and smooth Bezier straight/right/left turns.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import pygame
import pygame.gfxdraw

from core.config import (
    APPROACHES,
    NORTH,
    SOUTH,
    EAST,
    WEST,
    STARVATION_THRESHOLD,
)


APPROACH_KEYS: dict[int, str] = {
    NORTH: "N",
    SOUTH: "S",
    EAST: "E",
    WEST: "W",
}

CAR_COLORS: list[tuple[int, int, int]] = [
    (56, 189, 248),   # Sky Blue sedan
    (239, 68, 68),    # Crimson sedan
    (168, 85, 247),   # Violet coupe
    (34, 211, 238),   # Cyan sedan
    (59, 130, 246),   # Cobalt sedan
    (16, 185, 129),   # Emerald sedan
]


def _approach_vectors(approach: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return unit forward vector (dx, dy) and right-hand lane normal (nx, ny)."""
    if approach == NORTH:
        return (0.0, 1.0), (-1.0, 0.0)   # Heading Down (+y), right hand is Left (-x)
    if approach == SOUTH:
        return (0.0, -1.0), (1.0, 0.0)   # Heading Up (-y), right hand is Right (+x)
    if approach == WEST:
        return (1.0, 0.0), (0.0, 1.0)    # Heading Right (+x), right hand is Down (+y)
    return (-1.0, 0.0), (0.0, -1.0)      # EAST: Heading Left (-x), right hand is Up (-y)


class VisualIntersection:
    """Renders top-down 4-way road geometry, lane dividers, crosswalks, and corner scenery."""

    COLOR_GROUND = (19, 27, 38)
    COLOR_LAWN = (24, 46, 44)
    COLOR_LAWN_BORDER = (38, 72, 65)
    COLOR_SIDEWALK = (45, 55, 70)
    COLOR_ASPHALT = (32, 38, 49)
    COLOR_MEDIAN = (234, 179, 8)
    COLOR_LANE_DASH = (148, 163, 184)
    COLOR_STOP_LINE = (241, 245, 249)
    COLOR_CROSSWALK = (203, 213, 225)

    def __init__(
        self,
        offset_x: int,
        offset_y: int,
        panel_w: int,
        panel_h: int,
        road_w: int = 128,
    ) -> None:
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.panel_w = panel_w
        self.panel_h = panel_h
        self.road_w = road_w
        self.half_road = road_w // 2
        self.cx = offset_x + panel_w // 2
        self.cy = offset_y + panel_h // 2
        self.stop_margin = 18  # offset of crosswalk + stop line outside junction box

    def draw(self, surface: pygame.Surface) -> None:
        """Draw crisp top-down 4-way intersection inside clipped pane."""
        cx, cy = self.cx, self.cy
        hr = self.half_road
        ox, oy, pw, ph = self.offset_x, self.offset_y, self.panel_w, self.panel_h

        # 1. Background turf / urban block fill
        pygame.draw.rect(surface, self.COLOR_GROUND, (ox, oy, pw, ph))

        # 4 Corner landscape blocks (tiểu cảnh 4 góc ngã tư)
        corners = [
            (ox + 12, oy + 60, cx - hr - ox - 24, cy - hr - oy - 72),
            (cx + hr + 12, oy + 60, (ox + pw) - (cx + hr + 24), cy - hr - oy - 72),
            (ox + 12, cy + hr + 12, cx - hr - ox - 24, (oy + ph) - (cy + hr + 24)),
            (cx + hr + 12, cy + hr + 12, (ox + pw) - (cx + hr + 24), (oy + ph) - (cy + hr + 24)),
        ]
        for bx, by, bw, bh in corners:
            if bw > 20 and bh > 20:
                pygame.draw.rect(surface, self.COLOR_LAWN, (bx, by, bw, bh), border_radius=12)
                pygame.draw.rect(surface, self.COLOR_LAWN_BORDER, (bx, by, bw, bh), 1, border_radius=12)
                # Decorative trees in corners nearest intersection
                tree_x = bx + bw - 22 if bx < cx else bx + 22
                tree_y = by + bh - 22 if by < cy else by + 22
                pygame.draw.circle(surface, (18, 35, 34), (tree_x + 2, tree_y + 3), 12)
                pygame.draw.circle(surface, (36, 98, 72), (tree_x, tree_y), 11)
                pygame.draw.circle(surface, (52, 134, 96), (tree_x - 2, tree_y - 2), 6)

        # 2. Sidewalk curb shadows + Asphalt Roads
        pygame.draw.rect(surface, self.COLOR_SIDEWALK, (cx - hr - 6, oy, self.road_w + 12, ph))
        pygame.draw.rect(surface, self.COLOR_SIDEWALK, (ox, cy - hr - 6, pw, self.road_w + 12))
        pygame.draw.rect(surface, self.COLOR_ASPHALT, (cx - hr, oy, self.road_w, ph))
        pygame.draw.rect(surface, self.COLOR_ASPHALT, (ox, cy - hr, pw, self.road_w))

        # Subtle center junction box outline
        pygame.draw.rect(surface, (42, 50, 65), (cx - hr, cy - hr, self.road_w, self.road_w), 1)

        # 3. Double amber center divider lines (Dải phân cách giữa 2 chiều)
        stop_offset = hr + self.stop_margin
        # North & South medians
        pygame.draw.line(surface, self.COLOR_MEDIAN, (cx - 2, oy), (cx - 2, cy - stop_offset - 8), 2)
        pygame.draw.line(surface, self.COLOR_MEDIAN, (cx + 2, oy), (cx + 2, cy - stop_offset - 8), 2)
        pygame.draw.line(surface, self.COLOR_MEDIAN, (cx - 2, cy + stop_offset + 8), (cx - 2, oy + ph), 2)
        pygame.draw.line(surface, self.COLOR_MEDIAN, (cx + 2, cy + stop_offset + 8), (cx + 2, oy + ph), 2)
        # West & East medians
        pygame.draw.line(surface, self.COLOR_MEDIAN, (ox, cy - 2), (cx - stop_offset - 8, cy - 2), 2)
        pygame.draw.line(surface, self.COLOR_MEDIAN, (ox, cy + 2), (cx - stop_offset - 8, cy + 2), 2)
        pygame.draw.line(surface, self.COLOR_MEDIAN, (cx + stop_offset + 8, cy - 2), (ox + pw, cy - 2), 2)
        pygame.draw.line(surface, self.COLOR_MEDIAN, (cx + stop_offset + 8, cy + 2), (ox + pw, cy + 2), 2)

        # 4. Zebra Pedestrian Crosswalks
        zebra_w = 6
        zebra_len = 13
        for zx in range(cx - hr + 6, cx + hr - 4, 12):
            pygame.draw.rect(surface, self.COLOR_CROSSWALK, (zx, cy - hr - zebra_len - 2, zebra_w, zebra_len))
            pygame.draw.rect(surface, self.COLOR_CROSSWALK, (zx, cy + hr + 2, zebra_w, zebra_len))
        for zy in range(cy - hr + 6, cy + hr - 4, 12):
            pygame.draw.rect(surface, self.COLOR_CROSSWALK, (cx - hr - zebra_len - 2, zy, zebra_len, zebra_w))
            pygame.draw.rect(surface, self.COLOR_CROSSWALK, (cx + hr + 2, zy, zebra_len, zebra_w))

        # 5. Solid Stop Lines (right-hand incoming side only, plus thin clear line)
        pygame.draw.line(surface, self.COLOR_STOP_LINE, (cx - hr + 2, cy - stop_offset), (cx - 2, cy - stop_offset), 3)
        pygame.draw.line(surface, self.COLOR_STOP_LINE, (cx + 2, cy + stop_offset), (cx + hr - 2, cy + stop_offset), 3)
        pygame.draw.line(surface, self.COLOR_STOP_LINE, (cx - stop_offset, cy + 2), (cx - stop_offset, cy + hr - 2), 3)
        pygame.draw.line(surface, self.COLOR_STOP_LINE, (cx + stop_offset, cy - hr + 2), (cx + stop_offset, cy - 2), 3)


class SensorOverlay:
    """Draws translucent rectangular inductive-loop sensor zones & q/w labels ahead of stop lines."""

    def __init__(
        self,
        intersection: VisualIntersection,
        sensor_length: int = 126,
    ) -> None:
        self.intersection = intersection
        self.sensor_length = sensor_length

    def draw(
        self,
        surface: pygame.Surface,
        info: dict[str, Any],
        fonts: dict[str, pygame.font.Font],
        show_overlay: bool = True,
    ) -> None:
        """Render 4 green/amber/red sensor zones with live q=X | w=Ys badges."""
        if not show_overlay:
            return

        cx = self.intersection.cx
        cy = self.intersection.cy
        hr = self.intersection.half_road
        lane_w = hr - 8
        stop_off = hr + self.intersection.stop_margin + 5
        slen = self.sensor_length

        raw_queues = info.get("raw_queues", [0, 0, 0, 0])
        max_waits = info.get("max_waits", [0.0, 0.0, 0.0, 0.0])

        # Sensor rectangles matching incoming right-hand lanes
        zone_rects = {
            NORTH: pygame.Rect(cx - hr + 4, cy - stop_off - slen, lane_w, slen),
            SOUTH: pygame.Rect(cx + 4, cy + stop_off, lane_w, slen),
            WEST: pygame.Rect(cx - stop_off - slen, cy + 4, slen, lane_w),
            EAST: pygame.Rect(cx + stop_off, cy - hr + 4, slen, lane_w),
        }

        label_offsets = {
            NORTH: (cx - hr - 88, cy - stop_off - slen // 2 - 12),
            SOUTH: (cx + hr + 12, cy + stop_off + slen // 2 - 12),
            WEST: (cx - stop_off - slen + 4, cy + hr + 10),
            EAST: (cx + stop_off + 4, cy - hr - 34),
        }

        dir_names = {NORTH: "B", SOUTH: "N", EAST: "Đ", WEST: "T"}

        for app in APPROACHES:
            q_val = int(raw_queues[app])
            w_val = float(max_waits[app])
            rect = zone_rects[app]

            # Starvation risk threshold coloring (w_i >= 40s)
            if w_val >= STARVATION_THRESHOLD:
                fill_rgba = (239, 68, 68, 48)
                border_rgb = (248, 113, 113)
                txt_rgb = (254, 202, 202)
            elif w_val >= 25.0:
                fill_rgba = (245, 158, 11, 42)
                border_rgb = (251, 191, 36)
                txt_rgb = (254, 243, 199)
            else:
                fill_rgba = (34, 197, 94, 34)
                border_rgb = (74, 222, 128)
                txt_rgb = (187, 247, 208)

            zone_surf = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            zone_surf.fill(fill_rgba)
            surface.blit(zone_surf, rect.topleft)
            pygame.draw.rect(surface, border_rgb, rect, 1, border_radius=4)

            # Label pill with q=X | w=Ys
            lx, ly = label_offsets[app]
            label_text = f"{dir_names[app]}: q={q_val} | w={w_val:.0f}s"
            rendered = fonts["small"].render(label_text, True, txt_rgb)
            pill = pygame.Rect(lx, ly, rendered.get_width() + 10, 22)
            # Keep pill inside pane horizontal bounds
            min_x = self.intersection.offset_x + 6
            max_x = self.intersection.offset_x + self.intersection.panel_w - pill.w - 6
            pill.x = max(min_x, min(max_x, pill.x))
            pygame.draw.rect(surface, (15, 23, 36), pill, border_radius=5)
            pygame.draw.rect(surface, border_rgb, pill, 1, border_radius=5)
            surface.blit(rendered, (pill.x + 5, pill.y + 3))


class TrafficLightWidget:
    """Renders 4 physical 3-aspect traffic signal poles and live phase duration clock (tau_t)."""

    def __init__(self, intersection: VisualIntersection) -> None:
        self.intersection = intersection
        self._yellow_timer: float = 0.0
        self._yellow_axis: Optional[tuple[str, str]] = None
        self._last_observed_lights: dict[str, str] = {"N": "GREEN", "S": "GREEN", "E": "RED", "W": "RED"}

    def reset(self) -> None:
        self._yellow_timer = 0.0
        self._yellow_axis = None
        self._last_observed_lights = {"N": "GREEN", "S": "GREEN", "E": "RED", "W": "RED"}

    def on_step(self, info: dict[str, Any]) -> None:
        """Detect phase switch from info['last_step_switched'] to trigger visual yellow transition."""
        light_states = dict(info.get("light_states", {"N": "GREEN", "S": "GREEN", "E": "RED", "W": "RED"}))
        if info.get("last_step_switched", False):
            # Outgoing axis shows YELLOW briefly while incoming waits RED before turning GREEN
            if light_states.get("N") == "RED":
                self._yellow_axis = ("N", "S")
            else:
                self._yellow_axis = ("E", "W")
            self._yellow_timer = 0.22
        self._last_observed_lights = light_states

    def update(self, dt: float) -> None:
        if self._yellow_timer > 0.0:
            self._yellow_timer = max(0.0, self._yellow_timer - dt)
            if self._yellow_timer <= 0.0:
                self._yellow_axis = None

    def get_effective_states(self, info: dict[str, Any]) -> dict[str, str]:
        """Return effective 4-way bulb states, applying visual yellow clearance if active."""
        states = dict(info.get("light_states", self._last_observed_lights))
        if self._yellow_timer > 0.0 and self._yellow_axis is not None:
            # Outgoing axis shows YELLOW, incoming axis remains RED during yellow clearance
            for k in ("N", "S", "E", "W"):
                states[k] = "YELLOW" if k in self._yellow_axis else "RED"
        return states

    @staticmethod
    def _draw_housing(
        surface: pygame.Surface,
        x: int,
        y: int,
        state: str,
        horizontal: bool = False,
    ) -> None:
        """Draw a physical 3-bulb (Red, Yellow, Green) traffic signal head."""
        if horizontal:
            rect = pygame.Rect(x - 19, y - 7, 38, 14)
            centers = [(x - 12, y), (x, y), (x + 12, y)]
        else:
            rect = pygame.Rect(x - 7, y - 19, 14, 38)
            centers = [(x, y - 12), (x, y), (x, y + 12)]

        pygame.draw.rect(surface, (14, 18, 26), rect, border_radius=5)
        pygame.draw.rect(surface, (82, 96, 115), rect, 1, border_radius=5)

        bulbs = [
            ("RED", (239, 68, 68), (95, 25, 25)),
            ("YELLOW", (250, 204, 21), (105, 85, 18)),
            ("GREEN", (34, 197, 94), (20, 88, 45)),
        ]
        for (bx, by), (name, active_c, dim_c) in zip(centers, bulbs):
            is_on = (state == name)
            if is_on:
                pygame.gfxdraw.filled_circle(surface, bx, by, 6, dim_c)
                pygame.gfxdraw.aacircle(surface, bx, by, 6, dim_c)
                pygame.gfxdraw.filled_circle(surface, bx, by, 4, active_c)
                pygame.gfxdraw.aacircle(surface, bx, by, 4, (255, 255, 255))
            else:
                pygame.gfxdraw.filled_circle(surface, bx, by, 3, (40, 46, 58))

    def draw(
        self,
        surface: pygame.Surface,
        info: dict[str, Any],
        fonts: dict[str, pygame.font.Font],
    ) -> None:
        """Render 4 signal poles at junction corners + phase duration badge (tau_t)."""
        cx = self.intersection.cx
        cy = self.intersection.cy
        hr = self.intersection.half_road
        states = self.get_effective_states(info)

        # Placed cleanly at each incoming corner just beside the stop line
        poles = [
            ("N", cx - hr - 12, cy - hr - 18, False),
            ("S", cx + hr + 12, cy + hr + 18, False),
            ("W", cx - hr - 18, cy + hr + 12, True),
            ("E", cx + hr + 18, cy - hr - 12, True),
        ]
        for key, px, py, horiz in poles:
            self._draw_housing(surface, px, py, states.get(key, "RED"), horizontal=horiz)

        # Phase duration counter tau_t centered right below the junction box
        tau = float(info.get("phase_duration", 0.0))
        if self._yellow_timer > 0.0:
            axis_label = "ĐỆM ĐÈN VÀNG 3s"
            tau_color = (250, 204, 21)
        elif states.get("N") == "GREEN":
            axis_label = f"XANH B–N · τ={tau:.0f}s"
            tau_color = (74, 222, 128)
        else:
            axis_label = f"XANH Đ–T · τ={tau:.0f}s"
            tau_color = (56, 189, 248)

        txt = fonts["bold"].render(axis_label, True, tau_color)
        pill = pygame.Rect(cx - (txt.get_width() + 18) // 2, cy - 12, txt.get_width() + 18, 24)
        pygame.draw.rect(surface, (15, 23, 36, 220), pill, border_radius=6)
        pygame.draw.rect(surface, tau_color, pill, 1, border_radius=6)
        surface.blit(txt, (pill.x + 9, pill.y + 4))


class VisualVehicle:
    """Animated 2D vehicle supporting car/van/truck types, car-following & smooth curved turns."""

    SPECS: dict[str, tuple[int, int]] = {
        "car": (22, 13),
        "van": (26, 14),
        "truck": (32, 15),
    }

    def __init__(
        self,
        uid: int,
        approach: int,
        dist: float,
        queue_idx: int,
        color: tuple[int, int, int],
        v_type: str = "car",
        maneuver: str = "straight",
    ) -> None:
        self.uid = uid
        self.approach = approach
        self.dist = float(dist)
        self.speed = 0.0
        self.queue_idx = queue_idx
        self.is_discharged = False
        self.is_braking = False
        self.v_type = v_type
        self.maneuver = maneuver
        self.color = color
        self.yield_timer = 0.0
        self.length, self.width = self.SPECS.get(v_type, (22, 13))

    def step_kinematics(self, dt: float, target_dist: float, v_max: float = 420.0) -> None:
        a_acc = 1400.0
        b_brk = 1200.0
        gap = max(0.0, target_dist - self.dist)
        v_des = min(v_max, math.sqrt(2.0 * b_brk * gap)) if gap > 0.5 else 0.0

        if self.speed < v_des:
            self.speed = min(v_des, self.speed + a_acc * dt)
            self.is_braking = False
        else:
            self.speed = max(v_des, self.speed - b_brk * dt)
            self.is_braking = (self.speed < 12.0) or (v_des < self.speed - 14.0)

        # Strictly clamp position to target_dist for BOTH queued and discharged vehicles (prevents rear-end overlap!)
        next_dist = self.dist + self.speed * dt
        if next_dist >= target_dist - 0.05:
            self.dist = max(self.dist, target_dist) if target_dist >= self.dist else target_dist
            if gap <= 1.0:
                self.speed = 0.0
                self.is_braking = True
        else:
            self.dist = next_dist


class VisualTrafficScene:
    """Coordinates VisualIntersection, SensorOverlay, TrafficLightWidget, and VisualVehicles.

    Reads ONLY from environment `info` dicts; zero MDP state mutation or math runs here.
    """

    def __init__(
        self,
        offset_x: int,
        offset_y: int,
        panel_w: int,
        panel_h: int,
        road_w: int = 128,
    ) -> None:
        self.intersection = VisualIntersection(offset_x, offset_y, panel_w, panel_h, road_w=road_w)
        self.sensors = SensorOverlay(self.intersection)
        self.lights = TrafficLightWidget(self.intersection)

        self.lane_offset = road_w * 0.25
        self.spawn_margin = 40.0
        self.min_gap = 10.0  # Minimum bumper-to-bumper clear gap (px)
        self.car_spacing = 36.0

        self.vehicles: list[VisualVehicle] = []
        self._next_uid: int = 1
        self._rr_cursor: int = 0
        self._sprite_cache: dict[tuple[Any, ...], pygame.Surface] = {}
        self.last_info: dict[str, Any] = {
            "raw_queues": [0, 0, 0, 0],
            "max_waits": [0.0, 0.0, 0.0, 0.0],
            "phase": 0,
            "phase_duration": 0.0,
            "step_passed_vehicles": 0,
            "light_states": {"N": "GREEN", "S": "GREEN", "E": "RED", "W": "RED"},
            "last_step_switched": False,
        }

    def _maneuver_junction_len(self, maneuver: str) -> float:
        """Return true 2D arc length inside junction box so 1D speed matches 2D travel speed 1:1."""
        inter = self.intersection
        H = float(inter.half_road + inter.stop_margin)
        off = float(self.lane_offset)
        if maneuver == "straight":
            return 2.0 * H
        elif maneuver == "right":
            # Corner Bezier (0, -H) -> (0, 0) corner approximation length
            return 0.5 * math.pi * max(16.0, H - off)
        else:
            # Left turn arc length around intersection center
            return 0.5 * math.pi * (H + off)

    def _entry_and_stop_dist(self, approach: int, maneuver: str = "straight") -> tuple[float, float, float]:
        """Return (stop_dist, junction_len, exit_dist) along 1D path."""
        inter = self.intersection
        stop_off = inter.half_road + inter.stop_margin
        if approach in (NORTH, SOUTH):
            half_span = inter.panel_h / 2.0 + self.spawn_margin
            stop_dist = half_span - stop_off
            max_straight_exit = inter.panel_h + 2.0 * self.spawn_margin + 60.0
        else:
            half_span = inter.panel_w / 2.0 + self.spawn_margin
            stop_dist = half_span - stop_off
            max_straight_exit = inter.panel_w + 2.0 * self.spawn_margin + 60.0
        junction_len = self._maneuver_junction_len(maneuver)
        # Ensure turned cars exiting along perpendicular leg clear the panel cleanly
        exit_dist = max(max_straight_exit, stop_dist + junction_len + max(inter.panel_w, inter.panel_h) * 0.5 + 80.0)
        return stop_dist, junction_len, exit_dist

    def _spawn_vehicle(self, approach: int, dist: float, queue_idx: int, speed: float = 0.0) -> VisualVehicle:
        uid = self._next_uid
        self._next_uid += 1

        mod_type = uid % 10
        if mod_type == 0:
            v_type = "truck"
            color = (245, 158, 11)
        elif mod_type in (3, 7):
            v_type = "van"
            color = (226, 232, 240)
        else:
            v_type = "car"
            color = CAR_COLORS[uid % len(CAR_COLORS)]

        # Maneuver distribution: 65% straight, 20% right, 15% left
        r_m = uid % 20
        if r_m < 13:
            maneuver = "straight"
        elif r_m < 17:
            maneuver = "right"
        else:
            maneuver = "left"

        vv = VisualVehicle(
            uid=uid,
            approach=approach,
            dist=dist,
            queue_idx=queue_idx,
            color=color,
            v_type=v_type,
            maneuver=maneuver,
        )
        vv.speed = speed
        return vv

    @staticmethod
    def _opposite_approach(approach: int) -> int:
        if approach == NORTH:
            return SOUTH
        if approach == SOUTH:
            return NORTH
        if approach == EAST:
            return WEST
        return EAST

    def _safe_pair_spacing(self, lead: VisualVehicle, follower: VisualVehicle) -> float:
        """Exact center-to-center safe spacing accounting for varying car/van/truck lengths."""
        return 0.5 * float(lead.length) + 0.5 * float(follower.length) + self.min_gap

    def reset(self, info: dict[str, Any]) -> None:
        """Reset all visual vehicles and signal states from initial info dictionary."""
        self.vehicles.clear()
        self.lights.reset()
        self.last_info = dict(info)

        raw_queues = info.get("raw_queues", [0, 0, 0, 0])
        for app in APPROACHES:
            stop_dist, _, _ = self._entry_and_stop_dist(app)
            q_cnt = int(raw_queues[app])
            prev_v: Optional[VisualVehicle] = None
            for idx in range(q_cnt):
                nv = self._spawn_vehicle(app, dist=0.0, queue_idx=idx, speed=0.0)
                if prev_v is None:
                    nv.dist = max(0.0, stop_dist - 0.5 * nv.length - 4.0)
                else:
                    nv.dist = prev_v.dist - self._safe_pair_spacing(prev_v, nv)
                self.vehicles.append(nv)
                prev_v = nv

    def sync_from_info(self, info: dict[str, Any]) -> None:
        """Synchronize vehicle queues, discharges, and signal lights purely from `info`."""
        self.last_info = dict(info)
        self.lights.on_step(info)

        raw_queues = info.get("raw_queues", [0, 0, 0, 0])
        step_passed = int(info.get("step_passed_vehicles", 0))
        per_app_passed = info.get("step_passed_by_approach")
        light_states = info.get("light_states", {"N": "GREEN", "S": "GREEN", "E": "RED", "W": "RED"})

        green_apps = [app for app in APPROACHES if light_states.get(APPROACH_KEYS[app]) == "GREEN"]
        queued_by_app: dict[int, list[VisualVehicle]] = {
            app: sorted(
                [v for v in self.vehicles if v.approach == app and not v.is_discharged],
                key=lambda v: v.queue_idx,
            )
            for app in APPROACHES
        }

        # Step 1: Discharge head vehicles accurately per approach (using exact counts when available)
        if per_app_passed is not None and len(per_app_passed) == len(APPROACHES):
            for app in APPROACHES:
                pass_cnt = max(0, int(per_app_passed[app]))
                for _ in range(pass_cnt):
                    if queued_by_app[app]:
                        lead_v = queued_by_app[app].pop(0)
                        lead_v.is_discharged = True
                        lead_v.queue_idx = -1
                        lead_v.speed = max(lead_v.speed, 260.0)
                    else:
                        stop_dist, _, _ = self._entry_and_stop_dist(app)
                        min_dist = min((v.dist for v in self.vehicles if v.approach == app), default=stop_dist)
                        expr = self._spawn_vehicle(app, dist=0.0, queue_idx=-1, speed=320.0)
                        expr.dist = min(stop_dist - 0.5 * expr.length - 4.0, min_dist - expr.length - self.min_gap)
                        expr.is_discharged = True
                        self.vehicles.append(expr)
        elif step_passed > 0 and green_apps:
            remaining_pass = step_passed
            safety_guard = 0
            while remaining_pass > 0 and safety_guard < 100:
                safety_guard += 1
                # Prefer green approach that currently has surplus vehicles vs target raw_queues
                surplus_apps = [
                    a for a in green_apps if len(queued_by_app[a]) > int(raw_queues[a])
                ]
                if surplus_apps:
                    app = surplus_apps[self._rr_cursor % len(surplus_apps)]
                else:
                    app = green_apps[self._rr_cursor % len(green_apps)]
                self._rr_cursor += 1
                if queued_by_app[app]:
                    lead_v = queued_by_app[app].pop(0)
                    lead_v.is_discharged = True
                    lead_v.queue_idx = -1
                    lead_v.speed = max(lead_v.speed, 260.0)
                    remaining_pass -= 1
                elif all(not queued_by_app[a] for a in green_apps):
                    stop_dist, _, _ = self._entry_and_stop_dist(app)
                    min_dist = min((v.dist for v in self.vehicles if v.approach == app), default=stop_dist)
                    expr = self._spawn_vehicle(app, dist=0.0, queue_idx=-1, speed=320.0)
                    expr.dist = min(stop_dist - 0.5 * expr.length - 4.0, min_dist - expr.length - self.min_gap)
                    expr.is_discharged = True
                    self.vehicles.append(expr)
                    remaining_pass -= 1

        # Step 2: Reconcile remaining queued count per approach with exact info['raw_queues']
        for app in APPROACHES:
            target_q = int(raw_queues[app])
            q_list = queued_by_app[app]

            while len(q_list) > target_q:
                removed = q_list.pop(0)
                if light_states.get(APPROACH_KEYS[app]) == "GREEN":
                    removed.is_discharged = True
                    removed.queue_idx = -1
                else:
                    if removed in self.vehicles:
                        self.vehicles.remove(removed)

            while len(q_list) < target_q:
                new_idx = len(q_list)
                # Ensure newly spawned car never overlaps ANY existing vehicle on this approach
                all_app_dists = [v.dist for v in self.vehicles if v.approach == app]
                min_existing_dist = min(all_app_dists, default=40.0)
                nv = self._spawn_vehicle(app, dist=0.0, queue_idx=new_idx, speed=300.0)
                nv.dist = min(0.0, min_existing_dist - nv.length - self.min_gap)
                self.vehicles.append(nv)
                q_list.append(nv)

            for i, v in enumerate(q_list):
                v.queue_idx = i

    def _update_single_substep(self, dt: float) -> None:
        """Advance one small kinematic substep with intersection clearance & conflict avoidance."""
        self.lights.update(dt)
        eff_states = self.lights.get_effective_states(self.last_info)

        # Strict junction conflict box bounds: only vehicles physically inside asphalt conflict box
        # (past stop_dist - 2.0 and before clearing box at stop_dist + junction_len - 8.0) block entry.
        poses: dict[int, tuple[float, float, float]] = {}
        in_junction_by_app: dict[int, list[VisualVehicle]] = {app: [] for app in APPROACHES}

        for v in self.vehicles:
            poses[v.uid] = self._compute_world_pose(v)
            stop_dist, junction_len, _ = self._entry_and_stop_dist(v.approach, v.maneuver)
            if v.is_discharged and (stop_dist + 6.0 < v.dist < stop_dist + junction_len - 4.0):
                in_junction_by_app[v.approach].append(v)

        # Courtesy turn-slot for opposing left turns: prevents both deadlock and blind entry into oncoming stream
        left_slot_claims: dict[int, Optional[int]] = {app: None for app in APPROACHES}
        left_in_box: dict[int, bool] = {app: False for app in APPROACHES}
        for v in self.vehicles:
            if not v.is_discharged or v.maneuver != "left":
                continue
            sdist, jlen, _ = self._entry_and_stop_dist(v.approach, "left")
            hold_d = max(0.0, sdist - 0.5 * v.length - 4.0)
            if sdist + 6.0 <= v.dist <= sdist + jlen * 0.96:
                left_in_box[v.approach] = True
            elif (hold_d - 12.0 <= v.dist < sdist + 6.0) and v.yield_timer >= 0.25:
                if eff_states.get(APPROACH_KEYS[v.approach]) == "GREEN":
                    if left_slot_claims[v.approach] is None or v.uid < left_slot_claims[v.approach]:
                        left_slot_claims[v.approach] = v.uid

        surviving: list[VisualVehicle] = []
        for app in APPROACHES:
            stop_dist, _, _ = self._entry_and_stop_dist(app, "straight")
            is_green_now = (eff_states.get(APPROACH_KEYS[app]) == "GREEN")

            # Cross-axis approaches (perpendicular traffic actively inside intersection box)
            perp_apps = (EAST, WEST) if app in (NORTH, SOUTH) else (NORTH, SOUTH)
            perp_in_junction = any(len(in_junction_by_app[p]) > 0 for p in perp_apps)

            opp_app = self._opposite_approach(app)
            opp_in_junction = len(in_junction_by_app[opp_app]) > 0
            opp_stop_dist, _, _ = self._entry_and_stop_dist(opp_app, "straight")
            opp_straight_entering = any(
                ov.is_discharged
                and ov.maneuver != "left"
                and ov.speed > 50.0
                and (opp_stop_dist - 22.0 <= ov.dist <= opp_stop_dist + 40.0)
                for ov in self.vehicles
                if ov.approach == opp_app
            )
            opp_left_in_box = left_in_box[opp_app]
            my_claim = left_slot_claims[app]
            opp_claim = left_slot_claims[opp_app]
            opp_holds_left_slot = (
                opp_claim is not None
                and (my_claim is None or opp_claim < my_claim)
            )

            app_vehs = sorted(
                [v for v in self.vehicles if v.approach == app],
                key=lambda v: v.dist,
                reverse=True,
            )

            lead_v: Optional[VisualVehicle] = None

            # Compute deterministic stop target stack behind stop line for queued vehicles
            queued_cars = sorted(
                (v for v in app_vehs if not v.is_discharged),
                key=lambda x: x.queue_idx,
            )
            queue_targets: dict[int, float] = {}
            prev_qv: Optional[VisualVehicle] = None
            for qv in queued_cars:
                if prev_qv is None:
                    qt = max(0.0, stop_dist - 0.5 * qv.length - 4.0)
                else:
                    qt = max(0.0, queue_targets[prev_qv.uid] - self._safe_pair_spacing(prev_qv, qv))
                queue_targets[qv.uid] = qt
                prev_qv = qv

            for v in app_vehs:
                v_stop_dist, v_junction_len, exit_dist = self._entry_and_stop_dist(app, v.maneuver)
                stop_hold_dist = max(0.0, v_stop_dist - 0.5 * v.length - 4.0)

                if v.is_discharged:
                    target = exit_dist
                    # 1. Before entering asphalt conflict zone (crosswalk gate <= v_stop_dist + 4.0):
                    if v.dist <= v_stop_dist + 4.0:
                        hold_pt = min(stop_hold_dist, v.dist) if v.dist > stop_hold_dist else stop_hold_dist
                        if not is_green_now or perp_in_junction:
                            target = hold_pt
                            v.yield_timer = 0.0
                        elif v.maneuver == "left":
                            opp_priority_left = any(
                                ov.is_discharged
                                and ov.maneuver == "left"
                                and ov.uid < v.uid
                                and (opp_stop_dist - 10.0 <= ov.dist <= opp_stop_dist + v_junction_len * 0.96)
                                for ov in self.vehicles
                                if ov.approach == opp_app
                            )
                            if opp_in_junction or opp_left_in_box or opp_straight_entering or opp_priority_left or opp_holds_left_slot:
                                target = hold_pt
                                if (v.dist >= stop_hold_dist - 12.0) and not opp_holds_left_slot:
                                    v.yield_timer += dt
                            else:
                                # Keep yield_timer lock active across [hold_pt .. v_stop_dist + 6.0] so
                                # opp stays yielded while this left-turn launches into `left_in_box`!
                                if v.dist >= v_stop_dist + 4.0:
                                    v.yield_timer = 0.0
                        else:
                            # Straight / right traffic waits before crosswalk if opposing left turn
                            # is still inside the box OR has claimed the next turn slot
                            if opp_left_in_box or opp_holds_left_slot:
                                target = hold_pt
                            v.yield_timer = 0.0
                else:
                    target = queue_targets.get(v.uid, stop_hold_dist)
                    v.yield_timer = 0.0

                # 2. Active 2D Spatial Yielding inside junction box against ANY other approach's vehicle
                if v.dist > stop_hold_dist + 2.0 and v.dist < v_stop_dist + v_junction_len + 12.0:
                    vx, vy, vdeg = poses[v.uid]
                    v_rad = math.radians(vdeg)
                    fwd_x, fwd_y = math.cos(v_rad), math.sin(v_rad)
                    look_dist = 36.0 + 0.06 * v.speed
                    look_x = vx + fwd_x * look_dist
                    look_y = vy + fwd_y * look_dist
                    u_self = (v.dist - v_stop_dist) / max(1.0, v_junction_len)

                    for other_app in APPROACHES:
                        if other_app == app:
                            continue
                        for ov in in_junction_by_app[other_app]:
                            ox, oy, _ = poses[ov.uid]
                            dx_o, dy_o = ox - vx, oy - vy
                            proj_fwd = dx_o * fwd_x + dy_o * fwd_y
                            dist_center = math.hypot(dx_o, dy_o)
                            dist_look = math.hypot(ox - look_x, oy - look_y)
                            min_2d_safe = 0.5 * (v.length + ov.length) + 12.0
                            if (dist_center < min_2d_safe or dist_look < min_2d_safe) and proj_fwd > -4.0:
                                o_stop, o_jlen, _ = self._entry_and_stop_dist(ov.approach, ov.maneuver)
                                u_other = (ov.dist - o_stop) / max(1.0, o_jlen)

                                # Strict antisymmetric priority: prevents deadlocks while clearing front vehicle
                                if abs(u_other - u_self) > 0.025:
                                    other_has_priority = (u_other > u_self)
                                else:
                                    other_has_priority = (ov.uid < v.uid)

                                if other_has_priority:
                                    target = min(target, v.dist)
                                    break

                # 3. Same-lane car-following spacing constraint
                if lead_v is not None:
                    max_follow_dist = lead_v.dist - self._safe_pair_spacing(lead_v, v)
                    target = min(target, max_follow_dist)

                v.step_kinematics(dt, target)
                lead_v = v

                if v.dist < exit_dist:
                    surviving.append(v)

        self.vehicles = surviving

    def update(self, dt: float, speed_mult: float = 1.0) -> None:
        """Advance smooth vehicle kinematics with sub-stepping for collision-free motion at any speed."""
        total_dt = min(max(float(dt), 0.0), 0.08) * max(1.0, float(speed_mult))
        if total_dt <= 0.0:
            return
        max_sub = 0.02
        steps = max(1, int(math.ceil(total_dt / max_sub)))
        sub_dt = total_dt / steps
        for _ in range(steps):
            self._update_single_substep(sub_dt)

    def _compute_world_pose(self, vehicle: VisualVehicle) -> tuple[float, float, float]:
        """Compute (screen_x, screen_y, heading_deg) including Bezier curve turns inside junction."""
        inter = self.intersection
        cx, cy = float(inter.cx), float(inter.cy)
        stop_dist, junction_len, _ = self._entry_and_stop_dist(vehicle.approach, vehicle.maneuver)
        dx, dy = _approach_vectors(vehicle.approach)[0]
        nx, ny = _approach_vectors(vehicle.approach)[1]
        H = float(inter.half_road + inter.stop_margin)
        off = float(self.lane_offset)

        p0_x = cx - H * dx + off * nx
        p0_y = cy - H * dy + off * ny

        d = vehicle.dist
        if d <= stop_dist:
            # Approaching stop line along straight incoming lane
            rem = stop_dist - d
            return p0_x - dx * rem, p0_y - dy * rem, math.degrees(math.atan2(dy, dx))

        if d < stop_dist + junction_len and vehicle.maneuver != "straight":
            # Smooth quadratic Bezier curve through junction center
            u = (d - stop_dist) / max(1.0, junction_len)
            if vehicle.maneuver == "right":
                p2_x = cx + H * nx - off * dx
                p2_y = cy + H * ny - off * dy
                pc_x = cx - off * dx + off * nx
                pc_y = cy - off * dy + off * ny
            else:  # left turn
                p2_x = cx - H * nx + off * dx
                p2_y = cy - H * ny + off * dy
                pc_x = cx + off * dx + off * nx
                pc_y = cy + off * dy + off * ny

            omu = 1.0 - u
            px = (omu * omu) * p0_x + 2.0 * omu * u * pc_x + (u * u) * p2_x
            py = (omu * omu) * p0_y + 2.0 * omu * u * pc_y + (u * u) * p2_y
            tx = 2.0 * omu * (pc_x - p0_x) + 2.0 * u * (p2_x - pc_x)
            ty = 2.0 * omu * (pc_y - p0_y) + 2.0 * u * (p2_y - pc_y)
            return px, py, math.degrees(math.atan2(ty, tx))

        # Straight or post-junction outgoing lane
        if vehicle.maneuver == "right":
            dx_out, dy_out = nx, ny
            p_exit_x = cx + H * nx - off * dx
            p_exit_y = cy + H * ny - off * dy
            extra = d - (stop_dist + junction_len)
            return p_exit_x + dx_out * extra, p_exit_y + dy_out * extra, math.degrees(math.atan2(dy_out, dx_out))
        elif vehicle.maneuver == "left":
            dx_out, dy_out = -nx, -ny
            p_exit_x = cx - H * nx + off * dx
            p_exit_y = cy - H * ny + off * dy
            extra = d - (stop_dist + junction_len)
            return p_exit_x + dx_out * extra, p_exit_y + dy_out * extra, math.degrees(math.atan2(dy_out, dx_out))
        else:
            extra = d - stop_dist
            return p0_x + dx * extra, p0_y + dy * extra, math.degrees(math.atan2(dy, dx))

    def _get_vehicle_sprite(self, v: VisualVehicle) -> pygame.Surface:
        """Build cached UP-facing high-detail vehicle sprite."""
        key = (v.v_type, v.color, v.is_braking)
        if key in self._sprite_cache:
            return self._sprite_cache[key]

        size = 42
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        cx, cy = size // 2, size // 2
        w, l = v.width, v.length

        # Note: base sprite faces RIGHT (+x, angle = 0 deg) so atan2(dy, dx) rotation aligns cleanly!
        # Let length be along X, width along Y:
        body_rect = pygame.Rect(cx - l // 2, cy - w // 2, l, w)
        shadow_rect = body_rect.move(2, 2)
        pygame.draw.rect(surf, (10, 14, 22, 110), shadow_rect, border_radius=4)
        pygame.draw.rect(surf, v.color, body_rect, border_radius=4)
        border_c = tuple(max(0, c - 60) for c in v.color)
        pygame.draw.rect(surf, border_c, body_rect, 1, border_radius=4)

        if v.v_type == "truck":
            # Cargo cab distinction
            cab_x = cx + l // 2 - 8
            pygame.draw.rect(surf, (254, 240, 138), (cab_x, cy - w // 2 + 1, 7, w - 2), border_radius=2)
            pygame.draw.rect(surf, (25, 35, 50), (cab_x + 3, cy - w // 2 + 2, 3, w - 4))
        elif v.v_type == "van":
            pygame.draw.rect(surf, (30, 41, 59), (cx + l // 2 - 7, cy - w // 2 + 2, 4, w - 4), border_radius=1)
            pygame.draw.rect(surf, (203, 213, 225), (cx - l // 2 + 3, cy - w // 2 + 2, l - 11, w - 4), border_radius=2)
        else:
            # Sedan windshields & cabin
            pygame.draw.rect(surf, (24, 34, 48), (cx + l // 2 - 7, cy - w // 2 + 2, 4, w - 4), border_radius=1)
            pygame.draw.rect(surf, (24, 34, 48), (cx - l // 2 + 2, cy - w // 2 + 2, 3, w - 4), border_radius=1)

        # Front headlights (on right side +x)
        pygame.draw.circle(surf, (255, 250, 215), (cx + l // 2 - 1, cy - w // 2 + 3), 2)
        pygame.draw.circle(surf, (255, 250, 215), (cx + l // 2 - 1, cy + w // 2 - 3), 2)

        # Rear brake lights (on left side -x)
        tail_c = (255, 45, 45) if v.is_braking else (150, 25, 25)
        pygame.draw.circle(surf, tail_c, (cx - l // 2 + 1, cy - w // 2 + 3), 2)
        pygame.draw.circle(surf, tail_c, (cx - l // 2 + 1, cy + w // 2 - 3), 2)

        self._sprite_cache[key] = surf
        return surf

    def draw(
        self,
        surface: pygame.Surface,
        info: dict[str, Any],
        fonts: dict[str, pygame.font.Font],
        show_overlay: bool = True,
    ) -> None:
        """Render entire clipped intersection pane (road + sensors + lights + vehicles)."""
        inter = self.intersection
        old_clip = surface.get_clip()
        surface.set_clip(pygame.Rect(inter.offset_x, inter.offset_y, inter.panel_w, inter.panel_h))

        inter.draw(surface)
        self.sensors.draw(surface, info, fonts, show_overlay=show_overlay)

        # Render moving vehicles
        for v in self.vehicles:
            sx, sy, angle_deg = self._compute_world_pose(v)
            if (
                sx < inter.offset_x - 50
                or sx > inter.offset_x + inter.panel_w + 50
                or sy < inter.offset_y - 50
                or sy > inter.offset_y + inter.panel_h + 50
            ):
                continue
            sprite = self._get_vehicle_sprite(v)
            rotated = pygame.transform.rotate(sprite, -angle_deg)
            rw, rh = rotated.get_size()
            surface.blit(rotated, (int(sx - rw // 2), int(sy - rh // 2)))

        self.lights.draw(surface, info, fonts)
        surface.set_clip(old_clip)
