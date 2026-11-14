"""Traffic Renderer and Continuous Vehicle Animation Engine.

Provides smooth, physically realistic 60 FPS continuous 2D vehicle motion for
Gymnasium TrafficLightEnv simulation:
- Smooth continuous motion: Vehicles spawn at screen edges, drive in, queue up,
  and cross through the intersection when lights turn green.
- Right-hand traffic (Vietnam standard):
  * North -> South: Top to bottom (moves along right-hand lane).
  * South -> North: Bottom to top (moves along right-hand lane).
  * West -> East: Left to right (moves along right-hand lane).
  * East -> West: Right to left (moves along right-hand lane).
- Intelligent Driver Model (IDM) kinematics: Smooth acceleration, comfortable
  braking, collision avoidance, and queue compaction.
- Realistic visual aesthetics: Headlight beams illuminating the asphalt, glowing
  taillights when braking, windshields, roof contours, zebra crossings, and
  traffic signal halos.
"""

from __future__ import annotations

import math
from typing import Any
import pygame
import pygame.gfxdraw

from core.config import (
    APPROACHES,
    NORTH,
    SOUTH,
    EAST,
    WEST,
    PHASE_NS,
    PHASE_EW,
)
from core.env import TrafficLightEnv


# Palette of realistic vehicle body colors
CAR_COLORS = [
    (37, 99, 235),    # Sapphire Blue
    (225, 29, 72),    # Crimson Red
    (16, 185, 129),   # Emerald Green
    (245, 158, 11),   # Taxi Yellow / Amber
    (6, 182, 212),    # Cyber Teal
    (241, 245, 249),  # Pearl White
    (148, 163, 184),  # Slate Silver
    (139, 92, 246),   # Royal Violet
    (249, 115, 22),   # Sunset Orange
]

# Vehicle dimensions
CAR_LENGTH = 26
CAR_WIDTH = 15
CAR_SPACING = 34  # Headway distance between stopped cars (26px length + 8px gap)

# Kinematic parameters (pixels and seconds)
V_MAX = 135.0         # Max cruise speed (px/s)
A_ACCEL = 170.0       # Acceleration rate (px/s^2)
B_COMFORT = 210.0     # Comfortable braking rate (px/s^2)


class VisualVehicle:
    """Represents an animated vehicle with continuous physical kinematics."""

    def __init__(
        self,
        vehicle_id: int,
        approach: int,
        color: tuple[int, int, int],
        is_taxi: bool = False,
        is_suv: bool = False,
        initial_dist: float = 0.0,
    ) -> None:
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
        """Advance vehicle kinematics towards target distance with smooth deceleration."""
        self.target_dist = target_dist
        delta_d = max(0.0, target_dist - self.dist)

        # Compute safe desired speed
        if delta_d > 0.8:
            v_target = min(V_MAX, math.sqrt(2.0 * B_COMFORT * delta_d))
        else:
            v_target = 0.0

        # Adjust current speed with acceleration or braking limits
        if self.speed < v_target:
            self.speed = min(v_target, self.speed + A_ACCEL * dt)
            self.is_braking = False
        else:
            self.speed = max(v_target, self.speed - B_COMFORT * dt)
            self.is_braking = (self.speed < 12.0 or v_target < self.speed - 15.0)

        # Advance position
        if not self.is_discharged:
            self.dist = min(target_dist, self.dist + self.speed * dt)
            if self.dist >= target_dist - 0.05:
                self.dist = target_dist
                self.speed = 0.0
                self.is_braking = True
        else:
            self.dist += self.speed * dt
            self.is_braking = False


class TrafficRenderer:
    """Manages 2D intersection rendering and continuous vehicle physics."""

    def __init__(
        self,
        offset_x: int,
        offset_y: int,
        panel_w: int,
        panel_h: int,
        road_w: int = 140,
    ) -> None:
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.panel_w = panel_w
        self.panel_h = panel_h
        self.road_w = road_w

        self.cx = offset_x + panel_w // 2
        self.cy = offset_y + panel_h // 2
        self.lane_offset = road_w // 4

        # Spawn margins (just outside visible boundary)
        self.spawn_margin = 40

        # Stop line positions
        self.stop_y_north = self.cy - road_w // 2
        self.stop_y_south = self.cy + road_w // 2
        self.stop_x_west = self.cx - road_w // 2
        self.stop_x_east = self.cx + road_w // 2

        # 1D stop distances measured from spawn point
        self.dist_stop_north = (self.stop_y_north) - (self.offset_y - self.spawn_margin)
        self.dist_stop_south = (self.offset_y + self.panel_h + self.spawn_margin) - (self.stop_y_south)
        self.dist_stop_west = (self.stop_x_west) - (self.offset_x - self.spawn_margin)
        self.dist_stop_east = (self.offset_x + self.panel_w + self.spawn_margin) - (self.stop_x_east)

        # Total exit distances to fully leave the pane
        self.dist_exit_vert = self.panel_h + 2 * self.spawn_margin + 50
        self.dist_exit_horiz = self.panel_w + 2 * self.spawn_margin + 50

        # Vehicle registry
        self.vehicles: dict[int, VisualVehicle] = {}

        # Cached sprite templates for crisp rendering
        self._car_surfaces: dict[tuple[int, int, int, bool, bool], pygame.Surface] = {}

    def get_stop_distance(self, approach: int) -> float:
        if approach == NORTH:
            return float(self.dist_stop_north)
        elif approach == SOUTH:
            return float(self.dist_stop_south)
        elif approach == WEST:
            return float(self.dist_stop_west)
        else:
            return float(self.dist_stop_east)

    def get_exit_distance(self, approach: int) -> float:
        if approach in (NORTH, SOUTH):
            return float(self.dist_exit_vert)
        return float(self.dist_exit_horiz)

    def get_screen_pos(self, approach: int, dist: float) -> tuple[float, float, float]:
        """Convert 1D distance along approach lane into 2D (x, y, heading_rad)."""
        if approach == NORTH:
            # Traveling DOWN along x = cx - lane_offset
            x = self.cx - self.lane_offset
            y = (self.offset_y - self.spawn_margin) + dist
            return x, y, math.pi  # Facing South (180 deg)
        elif approach == SOUTH:
            # Traveling UP along x = cx + lane_offset
            x = self.cx + self.lane_offset
            y = (self.offset_y + self.panel_h + self.spawn_margin) - dist
            return x, y, 0.0  # Facing North (0 deg)
        elif approach == WEST:
            # Traveling RIGHT along y = cy + lane_offset
            x = (self.offset_x - self.spawn_margin) + dist
            y = self.cy + self.lane_offset
            return x, y, -math.pi / 2.0  # Facing East (-90 deg in Pygame coords)
        else:  # EAST
            # Traveling LEFT along y = cy - lane_offset
            x = (self.offset_x + self.panel_w + self.spawn_margin) - dist
            y = self.cy - self.lane_offset
            return x, y, math.pi / 2.0  # Facing West (90 deg)

    def reset(self, env: TrafficLightEnv) -> None:
        """Clear all active vehicles and populate from initial env queues."""
        self.vehicles.clear()
        for app in APPROACHES:
            stop_dist = self.get_stop_distance(app)
            for idx, veh in enumerate(env.queues[app]):
                color = CAR_COLORS[veh.vehicle_id % len(CAR_COLORS)]
                is_taxi = (veh.vehicle_id % 7 == 0)
                is_suv = (veh.vehicle_id % 5 == 0)

                # Pre-position initial cars stationary at their stop positions
                slot_dist = max(0.0, stop_dist - 16.0 - idx * CAR_SPACING)
                vv = VisualVehicle(
                    vehicle_id=veh.vehicle_id,
                    approach=app,
                    color=color,
                    is_taxi=is_taxi,
                    is_suv=is_suv,
                    initial_dist=slot_dist,
                )
                vv.queue_index = idx
                vv.target_dist = slot_dist
                self.vehicles[veh.vehicle_id] = vv

    def sync_with_env(self, env: TrafficLightEnv) -> None:
        """Synchronize with Gymnasium env state after a decision step."""
        current_queued_ids = set()
        for app in APPROACHES:
            for idx, veh in enumerate(env.queues[app]):
                current_queued_ids.add(veh.vehicle_id)
                if veh.vehicle_id not in self.vehicles:
                    # New arrival: spawn at dist = 0 (outside screen)
                    color = CAR_COLORS[veh.vehicle_id % len(CAR_COLORS)]
                    is_taxi = (veh.vehicle_id % 7 == 0)
                    is_suv = (veh.vehicle_id % 5 == 0)
                    vv = VisualVehicle(
                        vehicle_id=veh.vehicle_id,
                        approach=app,
                        color=color,
                        is_taxi=is_taxi,
                        is_suv=is_suv,
                        initial_dist=0.0,
                    )
                    vv.speed = V_MAX * 0.7  # enter with initial speed
                    vv.queue_index = idx
                    self.vehicles[veh.vehicle_id] = vv
                else:
                    self.vehicles[veh.vehicle_id].queue_index = idx

        # Any vehicle previously in queue that is no longer in env.queues was discharged!
        for vid, vv in self.vehicles.items():
            if not vv.is_discharged and vid not in current_queued_ids:
                vv.is_discharged = True
                vv.queue_index = -1

    def update(self, dt: float) -> None:
        """Update continuous physical motion for all active vehicles."""
        to_delete = []

        for app in APPROACHES:
            stop_dist = self.get_stop_distance(app)
            exit_dist = self.get_exit_distance(app)

            # Gather all vehicles on this approach
            app_vehs = [v for v in self.vehicles.values() if v.approach == app]

            # Sort descending by distance (vehicle furthest along lane is first)
            app_vehs.sort(key=lambda v: v.dist, reverse=True)

            lead_dist = exit_dist + 200.0

            for v in app_vehs:
                if v.is_discharged:
                    target_dist = exit_dist
                else:
                    target_dist = max(0.0, stop_dist - 16.0 - v.queue_index * CAR_SPACING)

                # Safe following distance: cannot exceed lead vehicle minus spacing
                safe_target = min(target_dist, lead_dist - CAR_SPACING)
                v.update_physics(dt, safe_target)

                lead_dist = v.dist

                # Remove vehicle once fully past exit boundary
                if v.dist >= exit_dist:
                    to_delete.append(v.vehicle_id)

        for vid in to_delete:
            self.vehicles.pop(vid, None)

    def _get_car_sprite(
        self,
        color: tuple[int, int, int],
        is_taxi: bool,
        is_suv: bool,
        is_braking: bool,
    ) -> pygame.Surface:
        """Generate or retrieve a detailed pre-rendered vehicle sprite facing UP."""
        key = (color[0], color[1], color[2], is_taxi, is_braking)
        if key in self._car_surfaces:
            return self._car_surfaces[key]

        w = 36
        h = 36
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        cx, cy = w // 2, h // 2
        car_w = CAR_WIDTH
        car_l = CAR_LENGTH

        # 1. Soft Drop Shadow
        shadow_rect = (cx - car_w // 2 + 1, cy - car_l // 2 + 3, car_w, car_l)
        pygame.draw.rect(surf, (0, 0, 0, 90), shadow_rect, border_radius=4)

        # 2. Main Car Body
        body_rect = (cx - car_w // 2, cy - car_l // 2, car_w, car_l)
        body_color = (245, 158, 11) if is_taxi else color
        pygame.draw.rect(surf, body_color, body_rect, border_radius=4)
        # Subtle rim outline
        rim_color = (max(0, body_color[0] - 50), max(0, body_color[1] - 50), max(0, body_color[2] - 50))
        pygame.draw.rect(surf, rim_color, body_rect, 1, border_radius=4)

        # 3. Windshield & Glass (Tinted cyan/dark)
        glass_color = (25, 35, 50)
        # Front Windshield
        pygame.draw.rect(surf, glass_color, (cx - car_w // 2 + 2, cy - car_l // 2 + 5, car_w - 4, 4), border_radius=1)
        # Rear Window
        pygame.draw.rect(surf, glass_color, (cx - car_w // 2 + 2, cy + car_l // 2 - 8, car_w - 4, 3), border_radius=1)
        # Roof panel
        pygame.draw.rect(surf, body_color, (cx - car_w // 2 + 2, cy - car_l // 2 + 9, car_w - 4, 6))

        # 4. Taxi Top Sign
        if is_taxi:
            pygame.draw.rect(surf, (255, 255, 255), (cx - 4, cy - 2, 8, 4), border_radius=1)
            pygame.draw.rect(surf, (0, 0, 0), (cx - 4, cy - 2, 8, 4), 1, border_radius=1)

        # 5. Dual Front Headlights (Bright warm white)
        pygame.draw.circle(surf, (255, 255, 230), (cx - car_w // 2 + 3, cy - car_l // 2 + 1), 2)
        pygame.draw.circle(surf, (255, 255, 230), (cx + car_w // 2 - 3, cy - car_l // 2 + 1), 2)

        # 6. Dual Rear Taillights
        tail_c = (255, 40, 40) if is_braking else (180, 20, 20)
        pygame.draw.circle(surf, tail_c, (cx - car_w // 2 + 3, cy + car_l // 2 - 1), 2)
        pygame.draw.circle(surf, tail_c, (cx + car_w // 2 - 3, cy + car_l // 2 - 1), 2)

        self._car_surfaces[key] = surf
        return surf

    def render(
        self,
        surface: pygame.Surface,
        env: TrafficLightEnv,
        fonts: dict[str, pygame.font.Font],
    ) -> None:
        """Render the complete intersection scene with continuous traffic animation."""
        # Clip drawing to intersection pane
        old_clip = surface.get_clip()
        surface.set_clip(pygame.Rect(self.offset_x, self.offset_y, self.panel_w, self.panel_h))

        road_w = self.road_w
        cx, cy = self.cx, self.cy
        stop = road_w // 2

        # 1. Asphalt Road Surfaces
        color_road = (34, 40, 52)
        color_marking = (240, 243, 246)
        color_dash = (150, 165, 185)

        # North-South Vertical Road
        pygame.draw.rect(surface, color_road, (cx - stop, self.offset_y, road_w, self.panel_h))
        # East-West Horizontal Road
        pygame.draw.rect(surface, color_road, (self.offset_x, cy - stop, self.panel_w, road_w))

        # 2. Dashed Lane Markings
        dash_len = 16
        gap_len = 12
        # North leg
        for dy in range(self.offset_y + 10, cy - stop - 15, dash_len + gap_len):
            pygame.draw.line(surface, color_dash, (cx, dy), (cx, dy + dash_len), 2)
        # South leg
        for dy in range(cy + stop + 15, self.offset_y + self.panel_h - 10, dash_len + gap_len):
            pygame.draw.line(surface, color_dash, (cx, dy), (cx, dy + dash_len), 2)
        # West leg
        for dx in range(self.offset_x + 10, cx - stop - 15, dash_len + gap_len):
            pygame.draw.line(surface, color_dash, (dx, cy), (dx + dash_len, cy), 2)
        # East leg
        for dx in range(cx + stop + 15, self.offset_x + self.panel_w - 10, dash_len + gap_len):
            pygame.draw.line(surface, color_dash, (dx, cy), (dx + dash_len, cy), 2)

        # 3. Zebra Crossings (Pedestrian Walkways)
        # North crosswalk
        for zx in range(cx - stop + 4, cx + stop - 4, 12):
            pygame.draw.rect(surface, (200, 210, 225, 180), (zx, cy - stop - 14, 7, 12))
        # South crosswalk
        for zx in range(cx - stop + 4, cx + stop - 4, 12):
            pygame.draw.rect(surface, (200, 210, 225, 180), (zx, cy + stop + 2, 7, 12))
        # West crosswalk
        for zy in range(cy - stop + 4, cy + stop - 4, 12):
            pygame.draw.rect(surface, (200, 210, 225, 180), (cx - stop - 14, zy, 12, 7))
        # East crosswalk
        for zy in range(cy - stop + 4, cy + stop - 4, 12):
            pygame.draw.rect(surface, (200, 210, 225, 180), (cx + stop + 2, zy, 12, 7))

        # 4. Solid White Stop Lines
        pygame.draw.line(surface, color_marking, (cx - stop, cy - stop), (cx + stop, cy - stop), 3)
        pygame.draw.line(surface, color_marking, (cx - stop, cy + stop), (cx + stop, cy + stop), 3)
        pygame.draw.line(surface, color_marking, (cx - stop, cy - stop), (cx - stop, cy + stop), 3)
        pygame.draw.line(surface, color_marking, (cx + stop, cy - stop), (cx + stop, cy + stop), 3)

        # 5. Traffic Signal Lights
        is_ns_green = (env.phase == PHASE_NS)
        c_green = (34, 197, 94)
        c_green_glow = (20, 90, 45)
        c_red = (239, 68, 68)
        c_red_glow = (100, 30, 30)

        c_ns = c_green if is_ns_green else c_red
        glow_ns = c_green_glow if is_ns_green else c_red_glow
        c_ew = c_red if is_ns_green else c_green
        glow_ew = c_red_glow if is_ns_green else c_green_glow

        # Draw anti-aliased signal circles with outer halo
        for sx, sy, sc, sg in [
            (cx - self.lane_offset - 15, cy - stop - 24, c_ns, glow_ns),  # North signal
            (cx + self.lane_offset + 15, cy + stop + 24, c_ns, glow_ns),  # South signal
            (cx + stop + 24, cy - self.lane_offset - 15, c_ew, glow_ew),  # East signal
            (cx - stop - 24, cy + self.lane_offset + 15, c_ew, glow_ew),  # West signal
        ]:
            pygame.gfxdraw.filled_circle(surface, int(sx), int(sy), 15, sg)
            pygame.gfxdraw.aacircle(surface, int(sx), int(sy), 15, sg)
            pygame.gfxdraw.filled_circle(surface, int(sx), int(sy), 10, sc)
            pygame.gfxdraw.aacircle(surface, int(sx), int(sy), 10, (255, 255, 255))

        # 6. Render Active Moving Vehicles
        # Sort so vehicles closer to center render nicely
        sorted_vehs = sorted(self.vehicles.values(), key=lambda v: (v.is_discharged, v.dist))

        for v in sorted_vehs:
            vx, vy, heading_rad = self.get_screen_pos(v.approach, v.dist)

            # Skip if fully off-screen
            if (
                vx < self.offset_x - 50
                or vx > self.offset_x + self.panel_w + 50
                or vy < self.offset_y - 50
                or vy > self.offset_y + self.panel_h + 50
            ):
                continue

            # Base sprite
            sprite = self._get_car_sprite(v.color, v.is_taxi, v.is_suv, v.is_braking)

            # Rotation angle in degrees
            angle_deg = math.degrees(heading_rad)
            rotated_sprite = pygame.transform.rotate(sprite, angle_deg)
            rw, rh = rotated_sprite.get_size()

            surface.blit(rotated_sprite, (vx - rw // 2, vy - rh // 2))

        # 7. Approach Statistics Overlay Badges
        q = [len(env.queues[app]) for app in APPROACHES]
        w = [
            env.queues[app][0].get_wait_time(env.current_time) if env.queues[app] else 0.0
            for app in APPROACHES
        ]

        b_north = fonts["bold"].render(f"Bắc: {q[0]} xe | {w[0]:.0f}s", True, (248, 250, 252))
        b_south = fonts["bold"].render(f"Nam: {q[1]} xe | {w[1]:.0f}s", True, (248, 250, 252))
        b_east = fonts["bold"].render(f"Đông: {q[2]} xe | {w[2]:.0f}s", True, (248, 250, 252))
        b_west = fonts["bold"].render(f"Tây: {q[3]} xe | {w[3]:.0f}s", True, (248, 250, 252))

        surface.blit(b_north, (cx - 130, cy - stop - 48))
        surface.blit(b_south, (cx + 20, cy + stop + 38))
        surface.blit(b_east, (cx + stop + 42, cy - 42))
        surface.blit(b_west, (cx - stop - 180, cy + 30))

        surface.set_clip(old_clip)
