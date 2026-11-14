"""Interactive Environment Inspector for TrafficLightEnv (Thực hành 3 Demo).

Standalone interactive demonstration of the Gymnasium TrafficLightEnv without
any RL algorithms or baselines. Allows manual or automated step-by-step
stepping to verify MDP dynamics, Action Masking safety rules, Poisson arrivals,
queue discharges, and reward breakdown.

Rendered in High-DPI crystal-clear 60 FPS graphics.
"""

from __future__ import annotations

import os
import sys

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Enable Windows Per-Monitor V2 High-DPI Awareness (fixes blurry rendering on Windows 10/11)
if sys.platform == "win32":
    import ctypes
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

from typing import Any
import numpy as np
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
    ACTION_KEEP,
    ACTION_SWITCH,
    Q_MAX,
    W_MAX,
    T_MIN,
    T_MAX,
    DELTA_T_GREEN,
    DELTA_T_YELLOW,
    SCENARIOS,
    ALPHA_QUEUE,
    BETA_WAIT_SQ,
    C_SWITCH,
    ETA_PASSED,
    DELTA_V_MAX,
)
from core.env import TrafficLightEnv


# Premium High-Contrast Color Palette
COLOR_BG = (16, 22, 34)
COLOR_PANEL_BG = (22, 29, 45)
COLOR_CARD = (28, 36, 56)
COLOR_CARD_BORDER = (48, 62, 92)
COLOR_ROAD = (36, 42, 54)
COLOR_ROAD_MARKING = (240, 243, 246)
COLOR_LANE_DASH = (160, 175, 195)
COLOR_GREEN = (34, 197, 94)
COLOR_GREEN_GLOW = (20, 90, 45)
COLOR_YELLOW = (245, 158, 11)
COLOR_YELLOW_GLOW = (110, 75, 15)
COLOR_RED = (239, 68, 68)
COLOR_RED_GLOW = (100, 30, 30)
COLOR_TEXT_PRIMARY = (248, 250, 252)
COLOR_TEXT_MUTED = (148, 163, 184)
COLOR_ACCENT = (56, 189, 248)
COLOR_ALERT_BG = (45, 25, 35)
COLOR_ALERT_TEXT = (251, 113, 133)
COLOR_SUCCESS_BG = (20, 45, 35)
COLOR_SUCCESS_TEXT = (74, 222, 128)


class EnvironmentInspectorApp:
    """Standalone Pygame GUI to inspect and demonstrate TrafficLightEnv."""

    def __init__(
        self,
        scenario_id: str = "SC3_ASYMMETRIC",
        seed: int = 42,
        headless: bool = False,
    ) -> None:
        self.scenario_id = scenario_id
        self.seed = seed
        self.headless = headless

        self.env = TrafficLightEnv(scenario_id=scenario_id, seed=seed)
        self.obs, self.info = self.env.reset(seed=seed)

        self.auto_step = False
        self.step_delay_ms = 400
        self.last_step_time = 0
        self.last_action = ACTION_KEEP
        self.last_reward = 0.0
        self.is_last_action_rejected = False
        self.status_message = "Môi trường sẵn sàng. Nhấn [SPACE] để thử đổi pha, [A] để tự động chạy."

        self.window_width = 1200
        self.window_height = 740
        self.is_running = True

        if not self.headless:
            pygame.init()
            pygame.display.set_caption("SmartTraffic RL Studio — Kiểm Thử Môi Trường (Thực Hành 3)")
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()

            # High-resolution vector system fonts
            self.fonts = {
                "header": pygame.font.SysFont("Segoe UI", 18, bold=True),
                "title": pygame.font.SysFont("Segoe UI", 15, bold=True),
                "section": pygame.font.SysFont("Segoe UI", 13, bold=True),
                "normal": pygame.font.SysFont("Segoe UI", 12),
                "bold": pygame.font.SysFont("Segoe UI", 12, bold=True),
                "mono": pygame.font.SysFont("Consolas", 12, bold=True),
                "mono_small": pygame.font.SysFont("Consolas", 11),
            }

    def reset_env(self, scenario_id: str | None = None) -> None:
        if scenario_id:
            self.scenario_id = scenario_id
        self.env = TrafficLightEnv(scenario_id=self.scenario_id, seed=self.seed)
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.last_action = ACTION_KEEP
        self.last_reward = 0.0
        self.is_last_action_rejected = False
        self.status_message = f"Đã reset môi trường! Kịch bản: {self.scenario_id}"

    def step(self, requested_action: int) -> None:
        mask = self.info["action_mask"]
        self.last_action = requested_action

        if mask[requested_action] == 0:
            self.is_last_action_rejected = True
            if requested_action == ACTION_SWITCH and mask[ACTION_KEEP] == 1:
                self.status_message = (
                    f"⚠️ [ACTION REJECTED] tau = {self.info['phase_duration']:.0f}s < T_MIN (10s): "
                    f"Khóa đổi pha, ép GIỮ PHA để chống nhấp nháy đèn!"
                )
            elif requested_action == ACTION_KEEP and mask[ACTION_SWITCH] == 1:
                self.status_message = (
                    f"⚠️ [SAFETY OVERRIDE] tau >= T_MAX (60s): Ép buộc ĐỔI PHA để chống đói đèn!"
                )
        else:
            self.is_last_action_rejected = False
            action_name = "ĐỔI PHA (3s vàng + 5s xanh)" if requested_action == ACTION_SWITCH else "GIỮ PHA (+5s xanh)"
            self.status_message = f"✅ Thực hiện {action_name} thành công."

        self.obs, self.last_reward, term, trunc, self.info = self.env.step(requested_action)
        if term or trunc:
            self.status_message = "🏁 Tập mô phỏng đã hoàn thành! Bấm [R] để reset."

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.step(ACTION_SWITCH)
                elif event.key == pygame.K_k:
                    self.step(ACTION_KEEP)
                elif event.key == pygame.K_a:
                    self.auto_step = not self.auto_step
                    self.status_message = f"Chế độ Auto-step: {'BẬT' if self.auto_step else 'TẮT'}"
                elif event.key == pygame.K_r:
                    self.reset_env()
                elif event.key == pygame.K_1:
                    self.reset_env("SC1_LOW")
                elif event.key == pygame.K_2:
                    self.reset_env("SC2_HIGH")
                elif event.key == pygame.K_3:
                    self.reset_env("SC3_ASYMMETRIC")
                elif event.key == pygame.K_4:
                    self.reset_env("SC4_DYNAMIC_SURGE")
                elif event.key == pygame.K_UP:
                    self.step_delay_ms = max(50, self.step_delay_ms - 50)
                elif event.key == pygame.K_DOWN:
                    self.step_delay_ms = min(1000, self.step_delay_ms + 50)
        return True

    def render(self) -> None:
        self.screen.fill(COLOR_BG)

        # 1. Khung Ngã Tư Mô Phỏng (Bên trái: 640 x 740)
        self._render_intersection_panel(0, 0, 650, self.window_height)

        # 2. Khung Chi Tiết MDP & Thông số Môi trường (Bên phải: 530 x 740)
        self._render_telemetry_panel(660, 15, 525, self.window_height - 30)

        pygame.display.flip()

    def _draw_smooth_circle(self, x: int, y: int, radius: int, color: tuple[int, int, int], glow: bool = False, glow_color: tuple[int, int, int] | None = None) -> None:
        """Draw an anti-aliased circle with optional halo glow."""
        if glow and glow_color:
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius + 5, glow_color)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius + 5, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, (255, 255, 255) if glow else (60, 60, 60))

    def _render_intersection_panel(self, x: int, y: int, w: int, h: int) -> None:
        cx, cy = x + w // 2, y + h // 2
        road_w = 140

        # Nền đường nhựa
        pygame.draw.rect(self.screen, COLOR_ROAD, (cx - road_w // 2, 0, road_w, h))
        pygame.draw.rect(self.screen, COLOR_ROAD, (x, cy - road_w // 2, w, road_w))

        # Vạch kẻ tim đường nét đứt (Dashed center lane markings)
        dash_len = 16
        gap_len = 12
        # Đường dọc Bắc - Nam
        for dy in range(10, cy - road_w // 2, dash_len + gap_len):
            pygame.draw.line(self.screen, COLOR_LANE_DASH, (cx, dy), (cx, dy + dash_len), 2)
        for dy in range(cy + road_w // 2 + 10, h - 80, dash_len + gap_len):
            pygame.draw.line(self.screen, COLOR_LANE_DASH, (cx, dy), (cx, dy + dash_len), 2)
        # Đường ngang Đông - Tây
        for dx in range(10, cx - road_w // 2, dash_len + gap_len):
            pygame.draw.line(self.screen, COLOR_LANE_DASH, (dx, cy), (dx + dash_len, cy), 2)
        for dx in range(cx + road_w // 2 + 10, w - 10, dash_len + gap_len):
            pygame.draw.line(self.screen, COLOR_LANE_DASH, (dx, cy), (dx + dash_len, cy), 2)

        # Vạch dừng xe (Solid Stop Lines)
        stop = road_w // 2
        pygame.draw.line(self.screen, COLOR_ROAD_MARKING, (cx - stop, cy - stop), (cx + stop, cy - stop), 3)
        pygame.draw.line(self.screen, COLOR_ROAD_MARKING, (cx - stop, cy + stop), (cx + stop, cy + stop), 3)
        pygame.draw.line(self.screen, COLOR_ROAD_MARKING, (cx - stop, cy - stop), (cx - stop, cy + stop), 3)
        pygame.draw.line(self.screen, COLOR_ROAD_MARKING, (cx + stop, cy - stop), (cx + stop, cy + stop), 3)

        # Đèn tín hiệu giao thông (Sắc nét khử răng cưa với hiệu ứng Glow)
        is_ns_green = (self.env.phase == PHASE_NS)
        c_ns = COLOR_GREEN if is_ns_green else COLOR_RED
        glow_ns = COLOR_GREEN_GLOW if is_ns_green else COLOR_RED_GLOW

        c_ew = COLOR_RED if is_ns_green else COLOR_GREEN
        glow_ew = COLOR_RED_GLOW if is_ns_green else COLOR_GREEN_GLOW

        # Đèn Bắc & Nam
        self._draw_smooth_circle(cx, cy - stop - 22, 13, c_ns, glow=True, glow_color=glow_ns)
        self._draw_smooth_circle(cx, cy + stop + 22, 13, c_ns, glow=True, glow_color=glow_ns)
        # Đèn Đông & Tây
        self._draw_smooth_circle(cx + stop + 22, cy, 13, c_ew, glow=True, glow_color=glow_ew)
        self._draw_smooth_circle(cx - stop - 22, cy, 13, c_ew, glow=True, glow_color=glow_ew)

        # Vẽ hàng xe đang chờ (Queues)
        self._render_vehicles(cx, cy, road_w)

        # Banner Tiêu Đề Trên Cùng
        t_title = self.fonts["header"].render("MÔ HÌNH HÓA NGÃ TƯ 4 HƯỚNG (TrafficLightEnv)", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(t_title, (x + 24, 20))

        sub = self.fonts["bold"].render(
            f"Pha đèn hiện tại: {'BẮC - NAM XANH' if is_ns_green else 'ĐÔNG - TÂY XANH'}  |  Thời gian duy trì pha tau: {self.env.phase_duration:.0f}s",
            True,
            COLOR_ACCENT,
        )
        self.screen.blit(sub, (x + 24, 48))

        # Hộp thông báo trạng thái & Cảnh báo an toàn (Bottom Notification Pill)
        box_bg = COLOR_ALERT_BG if self.is_last_action_rejected else COLOR_SUCCESS_BG
        box_border = (120, 40, 55) if self.is_last_action_rejected else (35, 95, 60)
        box_txt_color = COLOR_ALERT_TEXT if self.is_last_action_rejected else COLOR_SUCCESS_TEXT

        box_y = h - 68
        pygame.draw.rect(self.screen, box_bg, (20, box_y, w - 40, 48), border_radius=6)
        pygame.draw.rect(self.screen, box_border, (20, box_y, w - 40, 48), 1, border_radius=6)
        msg_txt = self.fonts["bold"].render(self.status_message, True, box_txt_color)
        self.screen.blit(msg_txt, (35, box_y + 14))

    def _render_vehicles(self, cx: int, cy: int, road_w: int) -> None:
        """Draw queued vehicles with clean rounded corners and windshield details."""
        stop = road_w // 2
        car_w, car_h = 24, 15

        car_body = (56, 189, 248)
        car_border = (20, 100, 150)
        windshield = (210, 240, 255)

        # North queue (chạy về hướng Nam)
        for idx, veh in enumerate(self.env.queues[NORTH][:11]):
            vy = (cy - stop - 18) - idx * (car_w + 5)
            rx, ry = cx - 18, vy - car_w
            pygame.draw.rect(self.screen, car_body, (rx, ry, car_h, car_w), border_radius=4)
            pygame.draw.rect(self.screen, car_border, (rx, ry, car_h, car_w), 1, border_radius=4)
            pygame.draw.rect(self.screen, windshield, (rx + 2, ry + car_w - 8, car_h - 4, 4), border_radius=1)

        # South queue (chạy về hướng Bắc)
        for idx, veh in enumerate(self.env.queues[SOUTH][:11]):
            vy = (cy + stop + 18) + idx * (car_w + 5)
            rx, ry = cx + 4, vy
            pygame.draw.rect(self.screen, car_body, (rx, ry, car_h, car_w), border_radius=4)
            pygame.draw.rect(self.screen, car_border, (rx, ry, car_h, car_w), 1, border_radius=4)
            pygame.draw.rect(self.screen, windshield, (rx + 2, ry + 4, car_h - 4, 4), border_radius=1)

        # East queue (chạy về hướng Tây)
        for idx, veh in enumerate(self.env.queues[EAST][:11]):
            vx = (cx + stop + 18) + idx * (car_w + 5)
            rx, ry = vx, cy - 18
            pygame.draw.rect(self.screen, car_body, (rx, ry, car_w, car_h), border_radius=4)
            pygame.draw.rect(self.screen, car_border, (rx, ry, car_w, car_h), 1, border_radius=4)
            pygame.draw.rect(self.screen, windshield, (rx + 4, ry + 2, 4, car_h - 4), border_radius=1)

        # West queue (chạy về hướng Đông)
        for idx, veh in enumerate(self.env.queues[WEST][:11]):
            vx = (cx - stop - 18) - idx * (car_w + 5)
            rx, ry = vx - car_w, cy + 4
            pygame.draw.rect(self.screen, car_body, (rx, ry, car_w, car_h), border_radius=4)
            pygame.draw.rect(self.screen, car_border, (rx, ry, car_w, car_h), 1, border_radius=4)
            pygame.draw.rect(self.screen, windshield, (rx + car_w - 8, ry + 2, 4, car_h - 4), border_radius=1)

        # Nhãn thông tin từng nhánh
        q = self.info["raw_queues"]
        w = self.info["max_waits"]
        self.screen.blit(self.fonts["bold"].render(f"Bắc (N): {q[0]} xe | Đợi: {w[0]:.1f}s", True, COLOR_TEXT_PRIMARY), (cx - 95, cy - stop - 48))
        self.screen.blit(self.fonts["bold"].render(f"Nam (S): {q[1]} xe | Đợi: {w[1]:.1f}s", True, COLOR_TEXT_PRIMARY), (cx - 95, cy + stop + 42))
        self.screen.blit(self.fonts["bold"].render(f"Đông (E): {q[2]} xe | Đợi: {w[2]:.1f}s", True, COLOR_TEXT_PRIMARY), (cx + stop + 40, cy - 25))
        self.screen.blit(self.fonts["bold"].render(f"Tây (W): {q[3]} xe | Đợi: {w[3]:.1f}s", True, COLOR_TEXT_PRIMARY), (cx - stop - 195, cy + 30))

    def _render_telemetry_panel(self, x: int, y: int, w: int, h: int) -> None:
        """Render detailed MDP state vector, action mask, and reward breakdown."""
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, (x, y, w, h), border_radius=10)
        pygame.draw.rect(self.screen, COLOR_CARD_BORDER, (x, y, w, h), 1, border_radius=10)

        cur_y = y + 16
        # Header
        t = self.fonts["header"].render("TELEMETRY & HỢP ĐỒNG MDP v2.0", True, COLOR_ACCENT)
        self.screen.blit(t, (x + 20, cur_y))
        cur_y += 34

        # 1. Action Masking Card
        card_h = 80
        pygame.draw.rect(self.screen, COLOR_CARD, (x + 15, cur_y, w - 30, card_h), border_radius=8)
        pygame.draw.rect(self.screen, COLOR_CARD_BORDER, (x + 15, cur_y, w - 30, card_h), 1, border_radius=8)

        m_title = self.fonts["title"].render("1. MẶT NẠ HÀNH ĐỘNG AN TOÀN (Action Masking)", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(m_title, (x + 25, cur_y + 10))

        mask = self.info["action_mask"]
        k_color = COLOR_GREEN if mask[0] == 1 else COLOR_RED
        s_color = COLOR_GREEN if mask[1] == 1 else COLOR_RED
        k_txt = f"Keep (0): {'HỢP LỆ' if mask[0] == 1 else 'BỊ CẤM'}"
        s_txt = f"Switch (1): {'HỢP LỆ' if mask[1] == 1 else 'BỊ CẤM'}"

        self.screen.blit(self.fonts["bold"].render(k_txt, True, k_color), (x + 35, cur_y + 36))
        self.screen.blit(self.fonts["bold"].render(s_txt, True, s_color), (x + 260, cur_y + 36))

        if self.env.phase_duration < T_MIN:
            rule_t = f"-> tau < {T_MIN}s: KHÓA ĐỔI PHA (Bảo đảm đèn xanh tối thiểu, chống nhấp nháy)"
        elif self.env.phase_duration >= T_MAX:
            rule_t = f"-> tau >= {T_MAX}s: ÉP BUỘC ĐỔI PHA (Đạt ngưỡng xanh tối đa, chống đói đèn)"
        else:
            rule_t = f"-> {T_MIN}s <= tau < {T_MAX}s: TỰ DO LỰA CHỌN (Tác tử toàn quyền quyết định)"
        self.screen.blit(self.fonts["mono_small"].render(rule_t, True, (210, 220, 235)), (x + 25, cur_y + 58))

        cur_y += card_h + 12

        # 2. Vector Quan Sát 10 Chiều s_t
        obs_h = 180
        pygame.draw.rect(self.screen, COLOR_CARD, (x + 15, cur_y, w - 30, obs_h), border_radius=8)
        pygame.draw.rect(self.screen, COLOR_CARD_BORDER, (x + 15, cur_y, w - 30, obs_h), 1, border_radius=8)
        self.screen.blit(self.fonts["title"].render("2. VECTOR TRẠNG THÁI CHUẨN HÓA s_t ∈ [0, 1]^10", True, COLOR_TEXT_PRIMARY), (x + 25, cur_y + 10))

        obs_labels = [
            f"q_B: {self.obs[0]:.2f} (Q_B/30)",
            f"q_N: {self.obs[1]:.2f} (Q_N/30)",
            f"q_Đ: {self.obs[2]:.2f} (Q_E/30)",
            f"q_T: {self.obs[3]:.2f} (Q_W/30)",
            f"w_B: {self.obs[4]:.2f} (W_B/120)",
            f"w_N: {self.obs[5]:.2f} (W_N/120)",
            f"w_Đ: {self.obs[6]:.2f} (W_E/120)",
            f"w_T: {self.obs[7]:.2f} (W_W/120)",
            f"p_t: {self.obs[8]:.1f} ({'B-N' if self.obs[8] == 0.0 else 'Đ-T'})",
            f"tau: {self.obs[9]:.2f} ({self.env.phase_duration:.0f}s/60s)",
        ]

        for i, lbl in enumerate(obs_labels):
            col = i % 2
            row = i // 2
            bx = x + 30 + col * 240
            by = cur_y + 36 + row * 27
            self.screen.blit(self.fonts["mono"].render(lbl, True, COLOR_TEXT_PRIMARY), (bx, by))
            # Thanh đo tiến trình chuẩn hóa
            val = float(np.clip(self.obs[i], 0.0, 1.0))
            pygame.draw.rect(self.screen, (40, 50, 72), (bx + 145, by + 4, 75, 9), border_radius=3)
            pygame.draw.rect(self.screen, COLOR_ACCENT, (bx + 145, by + 4, int(val * 75), 9), border_radius=3)

        cur_y += obs_h + 12

        # 3. Hàm Phần Thưởng Tức Thời r_t
        rw_h = 105
        pygame.draw.rect(self.screen, COLOR_CARD, (x + 15, cur_y, w - 30, rw_h), border_radius=8)
        pygame.draw.rect(self.screen, COLOR_CARD_BORDER, (x + 15, cur_y, w - 30, rw_h), 1, border_radius=8)
        self.screen.blit(self.fonts["title"].render("3. HÀM PHẦN THƯỞNG ĐA MỤC TIÊU (Reward r_t)", True, COLOR_TEXT_PRIMARY), (x + 25, cur_y + 10))

        rw_str = f"r_t = {self.last_reward:.4f}"
        self.screen.blit(self.fonts["header"].render(rw_str, True, COLOR_GREEN if self.last_reward > -1.0 else COLOR_YELLOW), (x + 35, cur_y + 34))

        formula_text = [
            "Công thức: r_t = - [ 1.0 * sum(q_i) + 1.5 * sum((w_i)^2) + 0.25 * 1(switch) ]",
            "                 + 0.4 * (v_passed / 10.0)",
        ]
        for idx, line in enumerate(formula_text):
            self.screen.blit(self.fonts["mono_small"].render(line, True, COLOR_TEXT_MUTED), (x + 35, cur_y + 64 + idx * 17))

        cur_y += rw_h + 12

        # 4. Điều Khiển & Phím Tắt
        ctrl_h = 165
        pygame.draw.rect(self.screen, COLOR_CARD, (x + 15, cur_y, w - 30, ctrl_h), border_radius=8)
        pygame.draw.rect(self.screen, COLOR_CARD_BORDER, (x + 15, cur_y, w - 30, ctrl_h), 1, border_radius=8)
        self.screen.blit(self.fonts["title"].render("4. ĐIỀU KHIỂN & KỊCH BẢN (THỰC HÀNH 3)", True, COLOR_TEXT_PRIMARY), (x + 25, cur_y + 10))

        sc_info = f"Kịch bản: {self.scenario_id}  |  Thời gian mô phỏng: {self.env.current_time:.0f}s  |  Bước: {self.env.step_count}"
        self.screen.blit(self.fonts["bold"].render(sc_info, True, COLOR_ACCENT), (x + 30, cur_y + 34))

        stats_txt = (
            f"Tổng xe xả: {self.env.total_passed_vehicles} xe  |  "
            f"Thời gian chờ trung bình: {self.env.total_wait_time_accumulated / max(1, self.env.total_passed_vehicles):.1f}s"
        )
        self.screen.blit(self.fonts["normal"].render(stats_txt, True, COLOR_TEXT_PRIMARY), (x + 30, cur_y + 56))

        keys = [
            "[SPACE]: Yêu cầu ĐỔI PHA (a_t = 1)",
            "[K]: Yêu cầu GIỮ PHA (a_t = 0)",
            "[A]: Bật / Tắt Auto-Step",
            "[R]: Reset Môi trường",
            "[1]-[4]: Chọn kịch bản SC1 -> SC4",
            "[UP]/[DOWN]: Tăng / Giảm tốc độ Auto",
        ]
        for idx, k in enumerate(keys):
            col = idx % 2
            row = idx // 2
            self.screen.blit(self.fonts["mono_small"].render(k, True, (220, 225, 238)), (x + 30 + col * 240, cur_y + 82 + row * 24))

    def run(self, max_steps: int | None = None) -> None:
        if self.headless:
            steps = max_steps or 5
            for i in range(steps):
                mask = self.info["action_mask"]
                chosen = ACTION_KEEP if mask[ACTION_KEEP] == 1 else ACTION_SWITCH
                self.step(chosen)
                print(f"[Headless Step {i+1}] time={self.env.current_time:.1f}s, phase={self.env.phase}, queues={self.info['raw_queues']}")
            self.env.close()
            return

        step_idx = 0
        while self.is_running:
            if not self.handle_events():
                break

            now = pygame.time.get_ticks()
            if self.auto_step and (now - self.last_step_time >= self.step_delay_ms):
                mask = self.info["action_mask"]
                chosen = ACTION_KEEP if mask[ACTION_KEEP] == 1 else ACTION_SWITCH
                self.step(chosen)
                self.last_step_time = now
                step_idx += 1
                if max_steps and step_idx >= max_steps:
                    break

            self.render()
            self.clock.tick(60)

        pygame.quit()
        self.env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone Environment Inspector")
    parser.add_argument(
        "--scenario",
        type=str,
        default="SC3_ASYMMETRIC",
        choices=["SC1_LOW", "SC2_HIGH", "SC3_ASYMMETRIC", "SC4_DYNAMIC_SURGE"],
        help="Initial traffic scenario",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master PRNG seed")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--steps", type=int, default=None, help="Max steps")
    args = parser.parse_args()

    app = EnvironmentInspectorApp(
        scenario_id=args.scenario,
        seed=args.seed,
        headless=args.headless,
    )
    app.run(max_steps=args.steps)
