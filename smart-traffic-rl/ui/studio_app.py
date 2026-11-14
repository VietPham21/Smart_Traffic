"""Split-Screen Studio Application (Pygame 1280x720).

Visualizes side-by-side A/B comparison between Baseline and RL signal controllers
receiving identical synchronized vehicle arrival streams in real time.

Panels:
- Left Pane (500x720): Baseline Intersection (Fixed-Time / Webster / Max-Pressure)
- Mid Pane (500x720): RL Agent Intersection (Double DQN / DQN / Tabular Q-Learning)
- Right Pane (280x720): Telemetry Dashboard & Interactive Controls (SC1-SC4, Speed Multiplier)

Maintainer: Cao Đức Đạo (MSV: 2351170581)
"""

from __future__ import annotations

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Enable Windows Per-Monitor V2 High-DPI Awareness (fixes blurry graphics on Windows 10/11)
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

from typing import Any, Optional
import numpy as np

from core.config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    LEFT_PANE_WIDTH,
    MID_PANE_WIDTH,
    RIGHT_PANE_WIDTH,
    FPS,
    APPROACHES,
    NORTH,
    SOUTH,
    EAST,
    WEST,
    PHASE_NS,
    PHASE_EW,
    ACTION_KEEP,
    ACTION_SWITCH,
    COLOR_BG,
    COLOR_PANE_BORDER,
    COLOR_ROAD,
    COLOR_ROAD_MARKING,
    COLOR_GREEN_LIGHT,
    COLOR_YELLOW_LIGHT,
    COLOR_RED_LIGHT,
    COLOR_CAR_BODY,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_MUTED,
    COLOR_ACCENT_RL,
    COLOR_ACCENT_BASELINE,
    SCENARIOS,
)
from core.env import TrafficLightEnv
from core.traffic_generator import TrafficGenerator
from agents.base_agent import BaseAgent
from agents.baselines import MaxPressureAgent, FixedTimeAgent
from agents.dqn_agent import DQNAgent


class SplitScreenStudio:
    """Side-by-side Pygame 1280x720 A/B Testing Studio for Traffic Signal Control."""

    def __init__(
        self,
        baseline_agent: Optional[BaseAgent] = None,
        rl_agent: Optional[BaseAgent] = None,
        scenario_id: str = "SC3_ASYMMETRIC",
        seed: int = 42,
        headless: bool = False,
    ) -> None:
        """Initialize Split-Screen Studio.

        Args:
            baseline_agent: Controller for the left intersection (default: MaxPressureAgent).
            rl_agent: Controller for the middle intersection (default: DQNAgent with Double DQN).
            scenario_id: Starting traffic scenario ID ('SC1_LOW', 'SC2_HIGH', etc.).
            seed: Shared seed ensuring both environments receive identical vehicle streams.
            headless: If True, operates in off-screen mode for headless tests.
        """
        # TODO [Cao Đức Đạo]: Khởi tạo ứng dụng đồ họa và tham số màn hình
        self.scenario_id: str = scenario_id
        self.seed: int = seed
        self.headless: bool = headless
        self.is_running: bool = False
        self.is_paused: bool = False
        self.sim_speed: int = 1  # 1x, 2x, 5x, 10x

        # Agents
        self.baseline_agent: BaseAgent = baseline_agent or MaxPressureAgent(name="Max-Pressure")
        self.rl_agent: BaseAgent = rl_agent or DQNAgent(use_double_dqn=True, name="Double-DQN")

        # Initialize Synchronized Dual Environments
        self.env_baseline: TrafficLightEnv
        self.env_rl: TrafficLightEnv
        self.obs_baseline: np.ndarray
        self.obs_rl: np.ndarray
        self.info_baseline: dict[str, Any] = {}
        self.info_rl: dict[str, Any] = {}

        self.setup_environments(self.scenario_id, self.seed)

        # Pygame Initialization
        self.pygame_module: Any = None
        self.screen: Any = None
        self.clock: Any = None
        self.fonts: dict[str, Any] = {}

        if not self.headless:
            self._init_pygame()

    def _init_pygame(self) -> None:
        """Initialize Pygame display, window surface, and fonts."""
        try:
            import pygame
            self.pygame_module = pygame
            pygame.init()
            pygame.display.set_caption("SmartTraffic RL Studio — Split-Screen Benchmark")
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.clock = pygame.time.Clock()

            # Fonts
            pygame.font.init()
            self.fonts["title"] = pygame.font.SysFont("Segoe UI, Roboto, Arial", 18, bold=True)
            self.fonts["body"] = pygame.font.SysFont("Segoe UI, Roboto, Arial", 14)
            self.fonts["metric"] = pygame.font.SysFont("Segoe UI, Roboto, Arial", 22, bold=True)
            self.fonts["small"] = pygame.font.SysFont("Segoe UI, Roboto, Arial", 12)
        except Exception as e:
            print(f"[SplitScreenStudio] Falling back to headless mode: {e}")
            self.headless = True

    def setup_environments(self, scenario_id: str, seed: int = 42) -> None:
        """Create two synchronized environments receiving identical vehicle streams."""
        # TODO [Cao Đức Đạo]: Thiết lập 2 môi trường nhận cùng luồng xe
        self.scenario_id = scenario_id
        self.seed = seed

        gen_base = TrafficGenerator(scenario_id=scenario_id, seed=seed)
        gen_rl = gen_base.clone_synchronized_generator()

        self.env_baseline = TrafficLightEnv(scenario_id=scenario_id, seed=seed, traffic_generator=gen_base)
        self.env_rl = TrafficLightEnv(scenario_id=scenario_id, seed=seed, traffic_generator=gen_rl)

        self.obs_baseline, self.info_baseline = self.env_baseline.reset(seed=seed)
        self.obs_rl, self.info_rl = self.env_rl.reset(seed=seed)

        self.baseline_agent.reset()
        self.rl_agent.reset()

    def step(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Advance both baseline and RL environments synchronously by one decision step.

        Uses select_action(obs, info, eval_mode=True) adhering to Interface Contract.
        """
        # 1. Baseline Decision
        action_base = self.baseline_agent.select_action(
            self.obs_baseline,
            self.info_baseline,
            eval_mode=True,
        )
        self.obs_baseline, r_base, term_b, trunc_b, self.info_baseline = self.env_baseline.step(action_base)

        # 2. RL Agent Decision
        action_rl = self.rl_agent.select_action(
            self.obs_rl,
            self.info_rl,
            eval_mode=True,
        )
        self.obs_rl, r_rl, term_r, trunc_r, self.info_rl = self.env_rl.step(action_rl)

        return self.info_baseline, self.info_rl

    def handle_events(self) -> bool:
        """Handle Pygame user input and button clicks."""
        if self.headless or not self.pygame_module:
            return True

        pygame = self.pygame_module
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
                elif event.key == pygame.K_1:
                    self.setup_environments("SC1_LOW", self.seed)
                elif event.key == pygame.K_2:
                    self.setup_environments("SC2_HIGH", self.seed)
                elif event.key == pygame.K_3:
                    self.setup_environments("SC3_ASYMMETRIC", self.seed)
                elif event.key == pygame.K_4:
                    self.setup_environments("SC4_DYNAMIC_SURGE", self.seed)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos
                self._handle_click(mouse_x, mouse_y)

        return True

    def _handle_click(self, x: int, y: int) -> None:
        """Handle button click coordinates in Right Telemetry Panel."""
        x0 = LEFT_PANE_WIDTH + MID_PANE_WIDTH
        # Check scenario buttons
        if x0 + 15 <= x <= x0 + RIGHT_PANE_WIDTH - 15:
            sc_list = ["SC1_LOW", "SC2_HIGH", "SC3_ASYMMETRIC", "SC4_DYNAMIC_SURGE"]
            for idx, sc_id in enumerate(sc_list):
                btn_y = 520 + idx * 36
                if btn_y <= y <= btn_y + 30:
                    self.setup_environments(sc_id, self.seed)
                    return

            # Speed buttons
            if 670 <= y <= 700:
                btn_w = (RIGHT_PANE_WIDTH - 30) // 4
                for idx, spd in enumerate([1, 2, 5, 10]):
                    bx = x0 + 15 + idx * btn_w
                    if bx <= x <= bx + btn_w:
                        self.sim_speed = spd
                        return

    def render_intersection(self, surface: Any, offset_x: int, env: TrafficLightEnv, title: str, accent_color: tuple[int, int, int]) -> None:
        """Render 2D top-down view of a single intersection pane (480 x 720)."""
        # TODO [Cao Đức Đạo]: Hoàn thiện đồ họa ngã tư 2D, làn đường và đèn tín hiệu
        pygame = self.pygame_module
        if not pygame:
            return

        cx, cy = offset_x + LEFT_PANE_WIDTH // 2, WINDOW_HEIGHT // 2
        road_width = 120

        # Background pane
        pygame.draw.rect(surface, COLOR_BG, (offset_x, 0, LEFT_PANE_WIDTH, WINDOW_HEIGHT))

        # Vertical road (North - South)
        pygame.draw.rect(surface, COLOR_ROAD, (cx - road_width // 2, 0, road_width, WINDOW_HEIGHT))
        # Horizontal road (East - West)
        pygame.draw.rect(surface, COLOR_ROAD, (offset_x, cy - road_width // 2, LEFT_PANE_WIDTH, road_width))

        # Stop lines
        pygame.draw.line(surface, COLOR_ROAD_MARKING, (cx - road_width // 2, cy - road_width // 2), (cx + road_width // 2, cy - road_width // 2), 2)
        pygame.draw.line(surface, COLOR_ROAD_MARKING, (cx - road_width // 2, cy + road_width // 2), (cx + road_width // 2, cy + road_width // 2), 2)
        pygame.draw.line(surface, COLOR_ROAD_MARKING, (cx - road_width // 2, cy - road_width // 2), (cx - road_width // 2, cy + road_width // 2), 2)
        pygame.draw.line(surface, COLOR_ROAD_MARKING, (cx + road_width // 2, cy - road_width // 2), (cx + road_width // 2, cy + road_width // 2), 2)

        # Traffic Signal Indicators
        is_ns_green = (env.phase == PHASE_NS)
        color_ns = COLOR_GREEN_LIGHT if is_ns_green else COLOR_RED_LIGHT
        color_ew = COLOR_RED_LIGHT if is_ns_green else COLOR_GREEN_LIGHT

        # North & South signals
        pygame.draw.circle(surface, color_ns, (cx, cy - road_width // 2 - 15), 10)
        pygame.draw.circle(surface, color_ns, (cx, cy + road_width // 2 + 15), 10)
        # East & West signals
        pygame.draw.circle(surface, color_ew, (cx + road_width // 2 + 15, cy), 10)
        pygame.draw.circle(surface, color_ew, (cx - road_width // 2 - 15, cy), 10)

        # Draw Queue Bars & Car Counts
        for app in APPROACHES:
            q_len = len(env.queues[app])
            if app == NORTH:
                label = f"N: {q_len} cars"
                pos = (cx - 30, cy - road_width // 2 - 40)
            elif app == SOUTH:
                label = f"S: {q_len} cars"
                pos = (cx - 30, cy + road_width // 2 + 35)
            elif app == EAST:
                label = f"E: {q_len} cars"
                pos = (cx + road_width // 2 + 30, cy - 10)
            else:
                label = f"W: {q_len} cars"
                pos = (cx - road_width // 2 - 90, cy - 10)

            txt = self.fonts["small"].render(label, True, COLOR_TEXT_PRIMARY)
            surface.blit(txt, pos)

        # Pane Header Banner
        header_surf = pygame.Surface((LEFT_PANE_WIDTH - 20, 40))
        header_surf.fill((30, 36, 50))
        surface.blit(header_surf, (offset_x + 10, 10))
        pygame.draw.line(surface, accent_color, (offset_x + 10, 50), (offset_x + LEFT_PANE_WIDTH - 10, 50), 3)

        title_txt = self.fonts["title"].render(title, True, COLOR_TEXT_PRIMARY)
        surface.blit(title_txt, (offset_x + 20, 18))

        # Pane Border
        pygame.draw.line(surface, COLOR_PANE_BORDER, (offset_x + LEFT_PANE_WIDTH, 0), (offset_x + LEFT_PANE_WIDTH, WINDOW_HEIGHT), 2)

    def render_telemetry_panel(self, surface: Any) -> None:
        """Render Right Telemetry Dashboard (320 x 720)."""
        # TODO [Cao Đức Đạo]: Hoàn thiện bảng Live Delta và nút điều khiển tương tác
        pygame = self.pygame_module
        if not pygame:
            return

        x0 = LEFT_PANE_WIDTH + MID_PANE_WIDTH
        panel_rect = (x0, 0, RIGHT_PANE_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(surface, (24, 28, 42), panel_rect)

        # Header
        head_txt = self.fonts["title"].render("TELEMETRY & A/B DELTA", True, COLOR_TEXT_PRIMARY)
        surface.blit(head_txt, (x0 + 15, 18))
        pygame.draw.line(surface, (60, 70, 95), (x0 + 15, 45), (x0 + RIGHT_PANE_WIDTH - 15, 45), 2)

        # Elapsed Time & Scenario
        sim_time_sec = float(self.env_rl.current_time)
        mins = int(sim_time_sec // 60)
        secs = int(sim_time_sec % 60)
        time_str = f"Time: {mins:02d}:{secs:02d} / 30:00"
        time_txt = self.fonts["body"].render(time_str, True, COLOR_TEXT_MUTED)
        surface.blit(time_txt, (x0 + 15, 55))

        sc_name = SCENARIOS.get(self.scenario_id, None)
        sc_desc = sc_name.name if sc_name else self.scenario_id
        sc_txt = self.fonts["body"].render(f"Scenario: {self.scenario_id}", True, (244, 208, 63))
        surface.blit(sc_txt, (x0 + 15, 75))

        # Metrics: Average Wait Time & Delta
        wait_base = float(self.info_baseline.get("avg_wait_time", 0.0))
        wait_rl = float(self.info_rl.get("avg_wait_time", 0.0))
        delta_wait = wait_rl - wait_base

        # Passed Vehicles & Delta
        pass_base = int(self.info_baseline.get("total_passed", 0))
        pass_rl = int(self.info_rl.get("total_passed", 0))
        delta_pass = pass_rl - pass_base

        y_off = 110
        # Telemetry Card 1: Wait Time
        pygame.draw.rect(surface, (32, 38, 56), (x0 + 15, y_off, RIGHT_PANE_WIDTH - 30, 85), border_radius=6)
        surface.blit(self.fonts["small"].render("AVG VEHICLE WAIT TIME (s)", True, COLOR_TEXT_MUTED), (x0 + 25, y_off + 8))
        surface.blit(self.fonts["body"].render(f"Baseline: {wait_base:5.1f}s", True, COLOR_ACCENT_BASELINE), (x0 + 25, y_off + 28))
        surface.blit(self.fonts["body"].render(f"RL Agent: {wait_rl:5.1f}s", True, COLOR_ACCENT_RL), (x0 + 25, y_off + 48))

        delta_color = COLOR_GREEN_LIGHT if delta_wait <= 0 else COLOR_RED_LIGHT
        delta_wait_txt = self.fonts["metric"].render(f"Delta: {delta_wait:+4.1f}s", True, delta_color)
        surface.blit(delta_wait_txt, (x0 + 160, y_off + 35))

        y_off += 95
        # Telemetry Card 2: Total Passed Vehicles
        pygame.draw.rect(surface, (32, 38, 56), (x0 + 15, y_off, RIGHT_PANE_WIDTH - 30, 85), border_radius=6)
        surface.blit(self.fonts["small"].render("TOTAL DISCHARGED VEHICLES", True, COLOR_TEXT_MUTED), (x0 + 25, y_off + 8))
        surface.blit(self.fonts["body"].render(f"Baseline: {pass_base} veh", True, COLOR_ACCENT_BASELINE), (x0 + 25, y_off + 28))
        surface.blit(self.fonts["body"].render(f"RL Agent: {pass_rl} veh", True, COLOR_ACCENT_RL), (x0 + 25, y_off + 48))

        delta_p_color = COLOR_GREEN_LIGHT if delta_pass >= 0 else COLOR_RED_LIGHT
        delta_pass_txt = self.fonts["metric"].render(f"Delta: {delta_pass:+d}", True, delta_p_color)
        surface.blit(delta_pass_txt, (x0 + 160, y_off + 35))

        y_off += 95
        # Action Mask Status Card
        mask_base = self.info_baseline.get("action_mask", np.array([1, 1]))
        mask_rl = self.info_rl.get("action_mask", np.array([1, 1]))
        pygame.draw.rect(surface, (32, 38, 56), (x0 + 15, y_off, RIGHT_PANE_WIDTH - 30, 75), border_radius=6)
        surface.blit(self.fonts["small"].render("CURRENT ACTION MASKS [Keep, Switch]", True, COLOR_TEXT_MUTED), (x0 + 25, y_off + 8))
        surface.blit(self.fonts["body"].render(f"Baseline: {list(mask_base)}", True, COLOR_TEXT_PRIMARY), (x0 + 25, y_off + 28))
        surface.blit(self.fonts["body"].render(f"RL Agent: {list(mask_rl)}", True, COLOR_TEXT_PRIMARY), (x0 + 25, y_off + 48))

        # Scenario Selection Buttons
        surface.blit(self.fonts["small"].render("SELECT SCENARIO (CLICK OR KEYS 1-4)", True, COLOR_TEXT_MUTED), (x0 + 15, 500))
        sc_keys = ["SC1_LOW", "SC2_HIGH", "SC3_ASYMMETRIC", "SC4_DYNAMIC_SURGE"]
        for idx, sc_id in enumerate(sc_keys):
            btn_y = 520 + idx * 36
            btn_active = (self.scenario_id == sc_id)
            btn_bg = (59, 130, 246) if btn_active else (40, 48, 68)
            pygame.draw.rect(surface, btn_bg, (x0 + 15, btn_y, RIGHT_PANE_WIDTH - 30, 30), border_radius=4)
            label = f"{sc_id}: {SCENARIOS[sc_id].name}"
            txt = self.fonts["body"].render(label, True, (255, 255, 255))
            surface.blit(txt, (x0 + 25, btn_y + 6))

        # Speed Multiplier Buttons
        surface.blit(self.fonts["small"].render("SIMULATION SPEED (SPACE = PAUSE)", True, COLOR_TEXT_MUTED), (x0 + 15, 665))
        for i, spd in enumerate([1, 2, 5, 10]):
            btn_x = x0 + 15 + i * 72
            btn_active = (self.sim_speed == spd)
            btn_bg = (168, 85, 247) if btn_active else (40, 48, 68)
            pygame.draw.rect(surface, btn_bg, (btn_x, 685, 65, 25), border_radius=4)
            txt = self.fonts["body"].render(f"{spd}x", True, (255, 255, 255))
            surface.blit(txt, (btn_x + 20, 688))

    def render(self) -> None:
        """Render complete studio frame to screen."""
        if self.headless or not self.screen:
            return

        self.screen.fill(COLOR_BG)

        # 1. Left Intersection: Baseline
        self.render_intersection(
            self.screen,
            offset_x=0,
            env=self.env_baseline,
            title=f"BASELINE: {self.baseline_agent.name}",
            accent_color=COLOR_ACCENT_BASELINE,
        )

        # 2. Middle Intersection: RL Agent
        self.render_intersection(
            self.screen,
            offset_x=LEFT_PANE_WIDTH,
            env=self.env_rl,
            title=f"AI CONTROLLER: {self.rl_agent.name}",
            accent_color=COLOR_ACCENT_RL,
        )

        # 3. Right Telemetry & Control Panel
        self.render_telemetry_panel(self.screen)

        self.pygame_module.display.flip()

    def run(self, max_steps: Optional[int] = None) -> None:
        """Execute split-screen interactive visualization loop."""
        self.is_running = True
        step_idx = 0

        while self.is_running:
            if not self.handle_events():
                break

            if not self.is_paused:
                for _ in range(self.sim_speed):
                    self.step()
                    step_idx += 1
                    if max_steps and step_idx >= max_steps:
                        self.is_running = False
                        break

            self.render()

            if self.clock:
                self.clock.tick(FPS)

        self.close()

    def close(self) -> None:
        """Clean up Pygame and environment resources."""
        if self.pygame_module and not self.headless:
            try:
                self.pygame_module.quit()
            except Exception:
                pass
        self.env_baseline.close()
        self.env_rl.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SmartTraffic RL Split-Screen Studio")
    parser.add_argument(
        "--scenario",
        type=str,
        default="SC3_ASYMMETRIC",
        choices=["SC1_LOW", "SC2_HIGH", "SC3_ASYMMETRIC", "SC4_DYNAMIC_SURGE"],
        help="Initial traffic scenario",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master PRNG seed")
    parser.add_argument("--headless", action="store_true", help="Run without Pygame display window")
    parser.add_argument("--steps", type=int, default=None, help="Optional max simulation steps to run")
    args = parser.parse_args()

    print(f"Launching SmartTraffic RL Studio [Scenario: {args.scenario}, Seed: {args.seed}]...")
    studio = SplitScreenStudio(
        scenario_id=args.scenario,
        seed=args.seed,
        headless=args.headless,
    )
    studio.run(max_steps=args.steps)
