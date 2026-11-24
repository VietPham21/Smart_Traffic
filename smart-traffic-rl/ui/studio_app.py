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
from ui.traffic_renderer import TrafficRenderer


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
        self.renderer_baseline: Any = None
        self.renderer_rl: Any = None

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
            self.fonts["title"] = pygame.font.SysFont("segoeui", 18, bold=True)
            self.fonts["body"] = pygame.font.SysFont("segoeui", 14)
            self.fonts["metric"] = pygame.font.SysFont("segoeui", 22, bold=True)
            self.fonts["small"] = pygame.font.SysFont("segoeui", 12)
            self.fonts["bold"] = pygame.font.SysFont("segoeui", 12, bold=True)

            # Continuous Vehicle Physics Renderers
            self.renderer_baseline = TrafficRenderer(
                offset_x=0,
                offset_y=0,
                panel_w=LEFT_PANE_WIDTH,
                panel_h=WINDOW_HEIGHT,
                road_w=126,
            )
            self.renderer_baseline.reset(self.env_baseline)

            self.renderer_rl = TrafficRenderer(
                offset_x=LEFT_PANE_WIDTH,
                offset_y=0,
                panel_w=MID_PANE_WIDTH,
                panel_h=WINDOW_HEIGHT,
                road_w=126,
            )
            self.renderer_rl.reset(self.env_rl)
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

        if hasattr(self, "renderer_baseline") and self.renderer_baseline:
            self.renderer_baseline.reset(self.env_baseline)
        if hasattr(self, "renderer_rl") and self.renderer_rl:
            self.renderer_rl.reset(self.env_rl)

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
        if self.renderer_baseline:
            self.renderer_baseline.sync_with_env(self.env_baseline)

        # 2. RL Agent Decision
        action_rl = self.rl_agent.select_action(
            self.obs_rl,
            self.info_rl,
            eval_mode=True,
        )
        self.obs_rl, r_rl, term_r, trunc_r, self.info_rl = self.env_rl.step(action_rl)
        if self.renderer_rl:
            self.renderer_rl.sync_with_env(self.env_rl)

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
                btn_y = 440 + idx * 38
                if btn_y <= y <= btn_y + 32:
                    self.setup_environments(sc_id, self.seed)
                    return

            # Speed buttons
            if 634 <= y <= 671:
                btn_w = 54
                for idx, spd in enumerate([1, 2, 5, 10]):
                    bx = x0 + 15 + idx * (btn_w + 7)
                    if bx <= x <= bx + btn_w:
                        self.sim_speed = spd
                        return

    def render_intersection(self, surface: Any, offset_x: int, env: TrafficLightEnv, title: str, accent_color: tuple[int, int, int]) -> None:
        """Render one isometric intersection and its comparison header."""
        pygame = self.pygame_module
        if not pygame:
            return

        renderer = self.renderer_baseline if offset_x == 0 else self.renderer_rl
        if renderer:
            renderer.render(surface, env, self.fonts)

        header_rect = pygame.Rect(offset_x + 12, 10, LEFT_PANE_WIDTH - 24, 42)
        pygame.draw.rect(surface, (22, 34, 49), header_rect, border_radius=10)
        pygame.draw.rect(surface, accent_color, header_rect, 1, border_radius=10)
        pygame.draw.circle(surface, accent_color, (offset_x + 28, 31), 5)
        title_txt = self.fonts["title"].render(title, True, COLOR_TEXT_PRIMARY)
        surface.blit(title_txt, (offset_x + 42, 20))
        mode_txt = self.fonts["small"].render("LIVE · A/B TEST", True, COLOR_TEXT_MUTED)
        surface.blit(mode_txt, (offset_x + LEFT_PANE_WIDTH - 133, 24))

        # Pane Border
        pygame.draw.line(surface, COLOR_PANE_BORDER,
                         (offset_x + LEFT_PANE_WIDTH, 0),
                         (offset_x + LEFT_PANE_WIDTH, WINDOW_HEIGHT), 1)

    def render_telemetry_panel(self, surface: Any) -> None:
        """Draw compact live metrics and controls in the fixed 280px sidebar."""
        pygame = self.pygame_module
        if not pygame:
            return

        x0 = LEFT_PANE_WIDTH + MID_PANE_WIDTH
        pygame.draw.rect(surface, (15, 23, 36), (x0, 0, RIGHT_PANE_WIDTH, WINDOW_HEIGHT))
        pygame.draw.line(surface, (48, 66, 83), (x0, 0), (x0, WINDOW_HEIGHT), 1)

        def label(text: str, x: int, y: int, color: tuple[int, int, int], font: str = "small") -> None:
            surface.blit(self.fonts[font].render(text, True, color), (x, y))

        label("LIVE TELEMETRY", x0 + 15, 17, COLOR_TEXT_PRIMARY, "title")
        pygame.draw.line(surface, (48, 66, 83), (x0 + 15, 45), (x0 + RIGHT_PANE_WIDTH - 15, 45), 1)
        elapsed = int(self.env_rl.current_time)
        label(f"SIM TIME   {elapsed // 60:02d}:{elapsed % 60:02d} / 30:00", x0 + 15, 57, COLOR_TEXT_MUTED, "body")
        scenario = SCENARIOS.get(self.scenario_id)
        label(scenario.name if scenario else self.scenario_id, x0 + 15, 80, (244, 196, 104))

        def card(y: int, title: str) -> None:
            rect = pygame.Rect(x0 + 12, y, RIGHT_PANE_WIDTH - 24, 91)
            pygame.draw.rect(surface, (25, 37, 52), rect, border_radius=10)
            pygame.draw.rect(surface, (47, 66, 83), rect, 1, border_radius=10)
            label(title, rect.x + 11, rect.y + 9, COLOR_TEXT_MUTED)

        wait_base = float(self.info_baseline.get("avg_wait_time", 0.0))
        wait_rl = float(self.info_rl.get("avg_wait_time", 0.0))
        delta_wait = wait_rl - wait_base
        passed_base = int(self.info_baseline.get("total_passed", 0))
        passed_rl = int(self.info_rl.get("total_passed", 0))
        delta_passed = passed_rl - passed_base

        card(108, "AVG WAIT  ·  SECONDS")
        label("BASE", x0 + 23, 139, COLOR_ACCENT_BASELINE)
        label(f"{wait_base:.1f}s", x0 + 66, 137, COLOR_TEXT_PRIMARY, "body")
        label("RL", x0 + 23, 164, COLOR_ACCENT_RL)
        label(f"{wait_rl:.1f}s", x0 + 66, 162, COLOR_TEXT_PRIMARY, "body")
        label(f"Δ {delta_wait:+.1f}s", x0 + 157, 151,
              COLOR_GREEN_LIGHT if delta_wait <= 0 else COLOR_RED_LIGHT, "body")

        card(208, "VEHICLES DISCHARGED")
        label("BASE", x0 + 23, 239, COLOR_ACCENT_BASELINE)
        label(str(passed_base), x0 + 66, 237, COLOR_TEXT_PRIMARY, "body")
        label("RL", x0 + 23, 264, COLOR_ACCENT_RL)
        label(str(passed_rl), x0 + 66, 262, COLOR_TEXT_PRIMARY, "body")
        label(f"Δ {delta_passed:+d}", x0 + 157, 251,
              COLOR_GREEN_LIGHT if delta_passed >= 0 else COLOR_RED_LIGHT, "body")

        mask_base = self.info_baseline.get("action_mask", np.array([1, 1]))
        mask_rl = self.info_rl.get("action_mask", np.array([1, 1]))
        card(308, "PHASE  ·  MASK [KEEP, SWITCH]")
        phase_base = "N—S" if self.env_baseline.phase == PHASE_NS else "E—W"
        phase_rl = "N—S" if self.env_rl.phase == PHASE_NS else "E—W"
        label(f"BASE  {phase_base} {self.env_baseline.phase_duration:.0f}s  {list(mask_base)}",
              x0 + 22, 340, COLOR_ACCENT_BASELINE)
        label(f"RL      {phase_rl} {self.env_rl.phase_duration:.0f}s  {list(mask_rl)}",
              x0 + 22, 367, COLOR_ACCENT_RL)

        label("SCENARIO  ·  KEYS 1–4", x0 + 15, 415, COLOR_TEXT_MUTED)
        scenario_ids = ["SC1_LOW", "SC2_HIGH", "SC3_ASYMMETRIC", "SC4_DYNAMIC_SURGE"]
        for idx, scenario_id in enumerate(scenario_ids):
            y = 440 + idx * 38
            active = self.scenario_id == scenario_id
            rect = pygame.Rect(x0 + 15, y, RIGHT_PANE_WIDTH - 30, 32)
            pygame.draw.rect(surface, (36, 65, 86) if active else (25, 37, 52), rect, border_radius=8)
            pygame.draw.rect(surface, COLOR_ACCENT_RL if active else (47, 66, 83), rect, 1, border_radius=8)
            label(f"{idx + 1}   {SCENARIOS[scenario_id].name}", rect.x + 11, rect.y + 8,
                  COLOR_TEXT_PRIMARY, "body")

        label("SPEED  ·  SPACE TO PAUSE", x0 + 15, 605, COLOR_TEXT_MUTED)
        for idx, speed in enumerate([1, 2, 5, 10]):
            rect = pygame.Rect(x0 + 15 + idx * 61, 634, 54, 36)
            active = self.sim_speed == speed
            pygame.draw.rect(surface, (43, 77, 96) if active else (25, 37, 52), rect, border_radius=8)
            pygame.draw.rect(surface, COLOR_ACCENT_RL if active else (47, 66, 83), rect, 1, border_radius=8)
            txt = self.fonts["body"].render(f"{speed}x", True, COLOR_TEXT_PRIMARY)
            surface.blit(txt, (rect.centerx - txt.get_width() // 2,
                               rect.centery - txt.get_height() // 2))

        status = "PAUSED" if self.is_paused else "RUNNING"
        status_color = (246, 190, 92) if self.is_paused else (89, 210, 151)
        label(f"●  {status}   ·   SEED {self.seed}", x0 + 15, 689, status_color)

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
        """Execute split-screen interactive visualization loop with continuous vehicle motion."""
        self.is_running = True
        step_idx = 0
        last_step_ticks = self.pygame_module.time.get_ticks() if self.pygame_module and not self.headless else 0
        last_frame_ticks = last_step_ticks

        while self.is_running:
            if not self.handle_events():
                break

            now = self.pygame_module.time.get_ticks() if self.pygame_module and not self.headless else 0

            # 1. Update continuous physical animation at 60 FPS
            if self.pygame_module and not self.headless:
                dt = (now - last_frame_ticks) / 1000.0
                last_frame_ticks = now
                if dt > 0.08:
                    dt = 0.016
                if self.renderer_baseline:
                    self.renderer_baseline.update(dt)
                if self.renderer_rl:
                    self.renderer_rl.update(dt)

            # 2. Advance decision steps
            if not self.is_paused:
                if self.headless:
                    for _ in range(self.sim_speed):
                        self.step()
                        step_idx += 1
                        if max_steps and step_idx >= max_steps:
                            self.is_running = False
                            break
                else:
                    step_interval_ms = max(50, 400 // max(1, self.sim_speed))
                    if now - last_step_ticks >= step_interval_ms:
                        self.step()
                        last_step_ticks = now
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
