"""Isometric (2.5D) traffic scene renderer and smooth vehicle animation."""

from __future__ import annotations

import math
from typing import Any

import pygame

from core.config import APPROACHES, NORTH, SOUTH, EAST, WEST, PHASE_NS
from core.env import TrafficLightEnv


CAR_COLORS = [
    (45, 123, 225), (221, 74, 92), (39, 174, 126), (238, 167, 55),
    (45, 190, 205), (225, 232, 239), (139, 157, 177), (148, 105, 218),
    (236, 115, 55),
]
CAR_SPACING = 38
V_MAX = 210.0
A_ACCEL = 320.0
B_COMFORT = 380.0


class VisualVehicle:
    """A vehicle with smooth one-dimensional motion along its approach."""

    def __init__(self, vehicle_id: int, approach: int, color: tuple[int, int, int],
                 is_taxi: bool = False, is_suv: bool = False,
                 initial_dist: float = 0.0) -> None:
        self.vehicle_id = vehicle_id
        self.approach = approach
        self.color = color
        self.is_taxi = is_taxi
        self.is_suv = is_suv
        self.dist = float(initial_dist)
        self.speed = 0.0
        self.target_dist = float(initial_dist)
        self.is_discharged = False
        self.is_braking = False
        self.queue_index = 0

    def update_physics(self, dt: float, target_dist: float) -> None:
        self.target_dist = target_dist
        gap = max(0.0, target_dist - self.dist)
        desired_speed = min(V_MAX, math.sqrt(2.0 * B_COMFORT * gap)) if gap > 0.8 else 0.0
        if self.speed < desired_speed:
            self.speed = min(desired_speed, self.speed + A_ACCEL * dt)
            self.is_braking = False
        else:
            self.speed = max(desired_speed, self.speed - B_COMFORT * dt)
            self.is_braking = self.speed < 12.0 or desired_speed < self.speed - 15.0

        next_dist = self.dist + self.speed * dt
        if next_dist >= target_dist - 0.05:
            self.dist = max(self.dist, target_dist) if target_dist >= self.dist else target_dist
            if gap <= 1.0:
                self.speed = 0.0
                self.is_braking = True
        else:
            self.dist = next_dist
            if self.is_discharged:
                self.is_braking = False


class TrafficRenderer:
    """Render a four-way intersection in an isometric projection.

    Vehicle physics remains in approach-distance space. Only the drawing layer
    projects world coordinates, so the Gymnasium environment is never mutated.
    """

    SKY = (17, 27, 43)
    GROUND = (24, 39, 54)
    ROAD = (46, 57, 69)
    ROAD_EDGE = (75, 91, 105)
    MARKING = (211, 221, 226)
    MUTED = (164, 183, 196)

    def __init__(self, offset_x: int, offset_y: int, panel_w: int,
                 panel_h: int, road_w: int = 140) -> None:
        self.offset_x, self.offset_y = offset_x, offset_y
        self.panel_w, self.panel_h = panel_w, panel_h
        self.road_w = road_w
        self.cx = offset_x + panel_w // 2
        self.cy = offset_y + int(panel_h * 0.54)
        self.iso_x = 0.707
        self.iso_y = 0.37
        self.lane_offset = 34.0
        self.stop_distance = 305.0
        self.exit_distance = 760.0
        self.spawn_margin = 380.0
        self.vehicles: dict[int, VisualVehicle] = {}
        self.current_phase: int = PHASE_NS

    def project(self, wx: float, wy: float) -> tuple[float, float]:
        """Project world-plane coordinates onto the isometric screen plane."""
        return (int(round(self.cx + (wx - wy) * self.iso_x)),
                int(round(self.cy + (wx + wy) * self.iso_y)))

    def get_stop_distance(self, approach: int) -> float:
        return self.stop_distance

    def get_exit_distance(self, approach: int) -> float:
        return self.exit_distance

    def _world_position(self, approach: int, distance: float) -> tuple[float, float]:
        lane = self.lane_offset
        if approach == NORTH:
            return -lane, -self.spawn_margin + distance
        if approach == SOUTH:
            return lane, self.spawn_margin - distance
        if approach == WEST:
            return -self.spawn_margin + distance, lane
        return self.spawn_margin - distance, -lane

    @staticmethod
    def _direction(approach: int) -> tuple[float, float]:
        if approach == NORTH:
            return 0.0, 1.0
        if approach == SOUTH:
            return 0.0, -1.0
        if approach == WEST:
            return 1.0, 0.0
        return -1.0, 0.0

    def get_screen_pos(self, approach: int, dist: float) -> tuple[float, float, float]:
        wx, wy = self._world_position(approach, dist)
        sx, sy = self.project(wx, wy)
        dx, dy = self._direction(approach)
        # Angle of the projected travel direction, used by callers that relied
        # on this helper in the earlier 2D renderer.
        ex, ey = self.project(wx + dx, wy + dy)
        return sx, sy, math.atan2(ey - sy, ex - sx)

    def reset(self, env: TrafficLightEnv) -> None:
        self.vehicles.clear()
        self.current_phase = int(env.phase)
        for approach in APPROACHES:
            queue = env.queues[approach]
            for index, vehicle in enumerate(queue):
                distance = max(0.0, self.stop_distance - 16.0 - index * CAR_SPACING)
                self.vehicles[vehicle.vehicle_id] = VisualVehicle(
                    vehicle.vehicle_id, approach,
                    CAR_COLORS[vehicle.vehicle_id % len(CAR_COLORS)],
                    vehicle.vehicle_id % 7 == 0, vehicle.vehicle_id % 5 == 0,
                    distance,
                )
                self.vehicles[vehicle.vehicle_id].queue_index = index

    def sync_with_env(self, env: TrafficLightEnv) -> None:
        self.current_phase = int(env.phase)
        queued: set[int] = set()
        for approach in APPROACHES:
            for index, vehicle in enumerate(env.queues[approach]):
                queued.add(vehicle.vehicle_id)
                visual = self.vehicles.get(vehicle.vehicle_id)
                if visual is None:
                    min_dist = min(
                        (v.dist for v in self.vehicles.values() if v.approach == approach),
                        default=float(CAR_SPACING),
                    )
                    visual = VisualVehicle(
                        vehicle.vehicle_id, approach,
                        CAR_COLORS[vehicle.vehicle_id % len(CAR_COLORS)],
                        vehicle.vehicle_id % 7 == 0, vehicle.vehicle_id % 5 == 0,
                        initial_dist=min(0.0, min_dist - float(CAR_SPACING)),
                    )
                    visual.speed = V_MAX * 0.75
                    self.vehicles[vehicle.vehicle_id] = visual
                visual.queue_index = index
        for vehicle_id, visual in list(self.vehicles.items()):
            if not visual.is_discharged and vehicle_id not in queued:
                visual.is_discharged = True
                visual.queue_index = -1

    def update(self, dt: float) -> None:
        dt = min(max(float(dt), 0.0), 0.08)
        remove: list[int] = []

        # Determine junction box occupancy per approach to prevent cross-axis collisions
        junction_end = self.stop_distance + float(self.road_w) + 30.0
        in_box: dict[int, bool] = {app: False for app in APPROACHES}
        for v in self.vehicles.values():
            if self.stop_distance - 10.0 < v.dist < junction_end:
                in_box[v.approach] = True

        for approach in APPROACHES:
            perp_apps = (EAST, WEST) if approach in (NORTH, SOUTH) else (NORTH, SOUTH)
            perp_busy = any(in_box[p] for p in perp_apps)
            is_green = (self.current_phase == PHASE_NS) if approach in (NORTH, SOUTH) else (self.current_phase != PHASE_NS)
            stop_hold = self.stop_distance - 16.0

            vehicles = sorted(
                (v for v in self.vehicles.values() if v.approach == approach),
                key=lambda v: v.dist, reverse=True,
            )
            lead_vehicle: VisualVehicle | None = None
            for vehicle in vehicles:
                if vehicle.is_discharged:
                    target = self.exit_distance
                    if vehicle.dist <= stop_hold + 2.0 and (not is_green or perp_busy):
                        target = stop_hold
                else:
                    target = max(0.0, stop_hold - vehicle.queue_index * CAR_SPACING)

                if lead_vehicle is not None:
                    req_gap = 40.0 if (vehicle.is_suv or lead_vehicle.is_suv) else float(CAR_SPACING)
                    safe_target = min(target, lead_vehicle.dist - req_gap)
                else:
                    safe_target = target

                vehicle.update_physics(dt, safe_target)
                lead_vehicle = vehicle
                if vehicle.dist >= self.exit_distance:
                    remove.append(vehicle.vehicle_id)
        for vehicle_id in remove:
            self.vehicles.pop(vehicle_id, None)

    def _world_rect(self, cx: float, cy: float, half_w: float,
                    half_h: float) -> list[tuple[float, float]]:
        return [self.project(cx - half_w, cy - half_h),
                self.project(cx + half_w, cy - half_h),
                self.project(cx + half_w, cy + half_h),
                self.project(cx - half_w, cy + half_h)]

    def _draw_ground(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, self.SKY,
                         (self.offset_x, self.offset_y, self.panel_w, self.panel_h))
        # Quiet city-block texture and rounded green medians.
        for wx, wy in [(-270, -270), (270, -270), (-270, 270), (270, 270)]:
            pts = self._world_rect(wx, wy, 72, 72)
            pygame.draw.polygon(surface, (30, 55, 64), pts)
            pygame.draw.polygon(surface, (39, 70, 77), pts, 2)
            for ox, oy in [(-32, -28), (20, -22), (-18, 24), (34, 29)]:
                x, y = self.project(wx + ox, wy + oy)
                pygame.draw.circle(surface, (45, 91, 78), (int(x), int(y)), 5)

        # A dark offset polygon beneath the roads gives the intersection depth.
        for hw, hh in [(self.road_w / 2 + 11, 450), (450, self.road_w / 2 + 11)]:
            shadow = [(x + 2, y + 8) for x, y in self._world_rect(0, 0, hw, hh)]
            pygame.draw.polygon(surface, (12, 20, 30), shadow)
            pygame.draw.polygon(surface, self.ROAD_EDGE,
                                self._world_rect(0, 0, hw, hh))
        for hw, hh in [(self.road_w / 2, 450), (450, self.road_w / 2)]:
            pygame.draw.polygon(surface, self.ROAD,
                                self._world_rect(0, 0, hw, hh))

        # Dashed center guide marks on each approach.
        for value in range(-440, 441, 42):
            if abs(value) < self.road_w / 2 + 13:
                continue
            for a, b in [((0, value), (0, value + 17)),
                         ((value, 0), (value + 17, 0))]:
                pygame.draw.line(surface, (116, 133, 145),
                                 self.project(*a), self.project(*b), 2)

        # Zebra crossings made from short, consistently projected stripes.
        half = self.road_w / 2
        for center, horizontal in [(-half - 24, True), (half + 24, True),
                                   (-half - 24, False), (half + 24, False)]:
            for offset in range(-52, 53, 17):
                if horizontal:
                    a, b = (offset, center), (offset + 8, center)
                else:
                    a, b = (center, offset), (center, offset + 8)
                pygame.draw.line(surface, (203, 215, 221),
                                 self.project(*a), self.project(*b), 4)

        # Stop bars.
        for y in (-half - 5, half + 5):
            pygame.draw.line(surface, (235, 239, 240),
                             self.project(-half, y), self.project(half, y), 3)
        for x in (-half - 5, half + 5):
            pygame.draw.line(surface, (235, 239, 240),
                             self.project(x, -half), self.project(x, half), 3)

    def _draw_car(self, surface: pygame.Surface, vehicle: VisualVehicle) -> None:
        wx, wy = self._world_position(vehicle.approach, vehicle.dist)
        dx, dy = self._direction(vehicle.approach)
        px, py = -dy, dx
        length = 27.0 if not vehicle.is_suv else 31.0
        width = 15.0 if not vehicle.is_suv else 17.0

        def shape(half_l: float, half_w: float, forward: float = 0.0,
                  side: float = 0.0, screen_dy: float = 0.0) -> list[tuple[float, float]]:
            center_x = wx + dx * forward + px * side
            center_y = wy + dy * forward + py * side
            points = []
            for along, across in [(half_l, half_w), (half_l, -half_w),
                                  (-half_l, -half_w), (-half_l, half_w)]:
                sx, sy = self.project(center_x + dx * along + px * across,
                                      center_y + dy * along + py * across)
                points.append((int(sx), int(sy + screen_dy)))
            return points

        pygame.draw.polygon(surface, (12, 17, 24), shape(length * .58, width * .58, screen_dy=5))
        body = (243, 173, 62) if vehicle.is_taxi else vehicle.color
        pygame.draw.polygon(surface, body, shape(length * .5, width * .5))
        pygame.draw.polygon(surface, (25, 35, 47), shape(length * .29, width * .37, forward=-1.5))
        pygame.draw.polygon(surface, (68, 92, 112), shape(length * .12, width * .31, forward=length * .20))
        pygame.draw.polygon(surface, (18, 27, 38), shape(length * .5, width * .5), 1)

        # Head lamps and brake lamps are rendered in world coordinates.
        for side in (-1, 1):
            hx, hy = self.project(wx + dx * (length * .43) + px * side * width * .30,
                                  wy + dy * (length * .43) + py * side * width * .30)
            pygame.draw.circle(surface, (255, 236, 177), (int(hx), int(hy)), 2)
            tx, ty = self.project(wx - dx * (length * .43) + px * side * width * .30,
                                  wy - dy * (length * .43) + py * side * width * .30)
            tail = (255, 75, 65) if vehicle.is_braking else (145, 53, 57)
            pygame.draw.circle(surface, tail, (int(tx), int(ty)), 2)

    def _draw_signal(self, surface: pygame.Surface, x: float, y: float,
                     green: bool) -> None:
        sx, sy = self.project(x, y)
        color = (53, 218, 143) if green else (246, 82, 88)
        pygame.draw.ellipse(surface, (13, 20, 29), (sx - 10, sy + 5, 20, 8))
        pygame.draw.line(surface, (157, 173, 178), (sx, sy + 9), (sx, sy - 9), 3)
        pygame.draw.circle(surface, (23, 31, 42), (int(sx), int(sy - 12)), 8)
        pygame.draw.circle(surface, color, (int(sx), int(sy - 12)), 4)
        pygame.draw.circle(surface, (255, 255, 255), (int(sx - 1), int(sy - 13)), 1)

    @staticmethod
    def _badge(surface: pygame.Surface, font: pygame.font.Font, text: str,
               pos: tuple[int, int], width: int = 112) -> None:
        x, y = pos
        rect = pygame.Rect(x, y, width, 25)
        pygame.draw.rect(surface, (16, 27, 40), rect, border_radius=7)
        pygame.draw.rect(surface, (67, 89, 105), rect, 1, border_radius=7)
        label = font.render(text, True, (233, 241, 244))
        surface.blit(label, (x + 8, y + 5))

    def render(self, surface: pygame.Surface, env: TrafficLightEnv,
               fonts: dict[str, pygame.font.Font]) -> None:
        previous_clip = surface.get_clip()
        surface.set_clip(pygame.Rect(self.offset_x, self.offset_y,
                                     self.panel_w, self.panel_h))
        self._draw_ground(surface)

        # Signals sit just outside each corner of the junction. Opposing arms
        # share the same phase, matching the environment's two-phase contract.
        half = self.road_w / 2 + 18
        north_south_green = env.phase == PHASE_NS
        for x, y, green in [(-half, -half, north_south_green),
                             (half, half, north_south_green),
                             (half, -half, not north_south_green),
                             (-half, half, not north_south_green)]:
            self._draw_signal(surface, x, y, green)

        # Painter's order follows projected depth, keeping vehicles visually
        # grounded while the 1D headway model prevents overlap on each approach.
        ordered = sorted(self.vehicles.values(), key=lambda v: sum(self._world_position(v.approach, v.dist)))
        for vehicle in ordered:
            self._draw_car(surface, vehicle)

        # Direction queue badges use live environment state and remain legible
        # over the scene rather than being baked into the simulation renderer.
        queues = {NORTH: "Bắc", SOUTH: "Nam", EAST: "Đông", WEST: "Tây"}
        badge_positions = {
            NORTH: (self.cx - 57, self.cy - 175),
            SOUTH: (self.cx - 57, self.cy + 145),
            WEST: (self.cx - 224, self.cy - 18),
            EAST: (self.cx + 112, self.cy - 18),
        }
        for approach, direction in queues.items():
            queue = env.queues[approach]
            wait = queue[0].get_wait_time(env.current_time) if queue else 0.0
            self._badge(surface, fonts["small"],
                        f"{direction}  {len(queue):02d} xe  {wait:.0f}s",
                        badge_positions[approach])

        green_axis = "BẮC—NAM" if north_south_green else "ĐÔNG—TÂY"
        phase_label = fonts["small"].render(
            f"XANH {green_axis}  ·  {env.phase_duration:.0f}s", True,
            (110, 235, 184) if north_south_green else (110, 214, 246),
        )
        phase_bg = pygame.Rect(self.cx - 83, self.cy + 184, 166, 25)
        pygame.draw.rect(surface, (16, 27, 40), phase_bg, border_radius=7)
        surface.blit(phase_label, (phase_bg.centerx - phase_label.get_width() // 2,
                                   phase_bg.y + 5))
        surface.set_clip(previous_clip)
