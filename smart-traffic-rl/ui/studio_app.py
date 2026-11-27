"""SmartTraffic RL Studio (Pygame 1360x760, 60 FPS Inference & Live Control UI).

Supports two interactive visualization modes:
1. Split-Screen A/B Testing (Default): Left (Baseline: Max-Pressure / Fixed-Time) vs
   Mid (AI: PPO / Double DQN / Q-Learning) receiving identical Poisson arrivals
   and synchronized by simulation clock `sim_time` (`|t_base - t_rl| <= 5s`).
2. Single Interactive View: Expanded 1020x760 center junction with direct hot-toggle
   checkbox `[x] Dùng AI` between AI and independent Baseline controller.

Features full interactive HUD (340x760):
- Checkbox `[x] Dùng AI` + AI selector (PPO / DDQN / Q-Learn) + Baseline selector (MP / FT)
- Inference Engine switcher: PyTorch (.pth) <-> ONNXRuntime (.onnx) at 60 FPS
- SpawnerCar: ON/OFF, ResetCar, Ẩn/Hiện TT, Scenario SC1-SC4, Speed 1x/2x/5x, Pause
- 4 Live Traffic Injection sliders (lambda_B, lambda_N, lambda_Đ, lambda_T in [0.00, 0.60])
- Sliding 1-minute metrics card (=== 1 phút gần nhất ===) & Delta % improvements
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    ACTION_KEEP,
    NORTH,
    SOUTH,
    EAST,
    WEST,
    PHASE_NS,
    SCENARIOS,
    COLOR_BG,
    COLOR_PANE_BORDER,
    COLOR_GREEN_LIGHT,
    COLOR_RED_LIGHT,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_MUTED,
    COLOR_ACCENT_RL,
    COLOR_ACCENT_BASELINE,
)
from core.env import TrafficLightEnv
from core.traffic_generator import TrafficGenerator
from agents.base_agent import BaseAgent
from agents.baselines import MaxPressureAgent, FixedTimeAgent
from agents.q_learning import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from ui.visual_entities import VisualTrafficScene


class OnnxRuntimeAgent(BaseAgent):
    """Lightweight 60 FPS ONNXRuntime inference wrapper adhering strictly to BaseAgent."""

    def __init__(self, onnx_path: str, name: str = "ONNX-Agent") -> None:
        super().__init__(name=name)
        import onnxruntime as ort

        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def select_action(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        eval_mode: bool = False,
    ) -> int:
        action_mask = np.asarray(
            info.get("action_mask", np.array([1, 1], dtype=np.int8)),
            dtype=np.float32,
        )
        valid_actions = np.where(action_mask > 0.5)[0]
        if len(valid_actions) == 0:
            return ACTION_KEEP

        obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        mask_batch = action_mask.reshape(1, -1)
        outputs = self.session.run(None, {"obs": obs_batch, "action_mask": mask_batch})
        chosen = int(np.asarray(outputs[1]).reshape(-1)[0])
        if action_mask[chosen] < 0.5:
            chosen = int(valid_actions[0])
        return chosen


def _format_compact_status(ai_key: str, filename: Optional[str], engine: str) -> str:
    """Format compact checkpoint badge (<= 22 chars) so it never clips on 1360px HUD."""
    prefix_map = {"ppo": "PPO", "ddqn": "DDQN", "q_learning": "QL"}
    prefix = prefix_map.get(ai_key, ai_key.upper())
    if not filename:
        return f"{prefix} [Chưa train]"

    stem = os.path.splitext(filename)[0]
    for tok in (f"{ai_key}_", "SC1_LOW", "SC2_HIGH", "SC3_ASYMMETRIC", "SC4_DYNAMIC_SURGE"):
        short = {
            f"{ai_key}_": "",
            "SC1_LOW": "SC1",
            "SC2_HIGH": "SC2",
            "SC3_ASYMMETRIC": "SC3",
            "SC4_DYNAMIC_SURGE": "SC4",
        }[tok]
        stem = stem.replace(tok, short)
    eng_suffix = ".onnx" if engine == "onnx" else ""
    label = f"{prefix} [{stem}{eng_suffix}]"
    return label[:22]


class SplitScreenStudio:
    """Interactive 1360x760 60 FPS Inference & Live Benchmarking Studio."""

    AI_KEYS = ("ppo", "ddqn", "q_learning")
    AI_LABELS = {"ppo": "PPO", "ddqn": "DDQN", "q_learning": "Q-Learn"}
    BASE_KEYS = ("max_pressure", "fixed_time")
    BASE_LABELS = {"max_pressure": "MaxPress", "fixed_time": "Fixed30s"}

    def __init__(
        self,
        baseline_agent: Optional[BaseAgent] = None,
        rl_agent: Optional[BaseAgent] = None,
        scenario_id: str = "SC3_ASYMMETRIC",
        seed: int = 42,
        headless: bool = False,
        view_mode: str = "split",
        ai_key: str = "ppo",
        inference_engine: str = "pytorch",
    ) -> None:
        self.scenario_id: str = scenario_id if scenario_id in SCENARIOS else "SC3_ASYMMETRIC"
        self.seed: int = int(seed)
        self.headless: bool = bool(headless)
        self.view_mode: str = "single" if view_mode == "single" else "split"

        self.is_running: bool = False
        self.is_paused: bool = False
        self.sim_speed: int = 1
        self.use_ai: bool = True
        self.show_hud_overlay: bool = True
        self.spawner_enabled: bool = True
        self.live_rates_active: bool = False
        self.dragging_slider_idx: Optional[int] = None

        self.ai_key: str = ai_key if ai_key in self.AI_KEYS else "ppo"
        self.base_key: str = "max_pressure"
        self.inference_engine: str = "onnx" if inference_engine == "onnx" else "pytorch"
        self.ai_checkpoint_status: str = ""
        self.active_engine_label: str = "PT"

        # Independent baseline instances for Left (baseline) & Right (when AI is unchecked)
        self._custom_baseline = baseline_agent is not None
        self._custom_rl = rl_agent is not None
        self.baseline_agent: BaseAgent = baseline_agent or self._build_baseline_agent(self.base_key)
        self.baseline_agent_right: BaseAgent = self._build_baseline_agent(self.base_key)
        self.rl_agent: BaseAgent = rl_agent or self._load_ai_agent(self.ai_key, self.scenario_id)

        # Synchronized Environments (strictly decoupled from Pygame)
        self.env_baseline: TrafficLightEnv
        self.env_rl: TrafficLightEnv
        self.obs_baseline: np.ndarray
        self.obs_rl: np.ndarray
        self.info_baseline: dict[str, Any] = {}
        self.info_rl: dict[str, Any] = {}
        self._base_finished: bool = False
        self._rl_finished: bool = False

        # Live Injection slider values (veh/s in [0.0, 0.60])
        self.slider_rates: list[float] = [0.42, 0.42, 0.08, 0.08]
        self.ui_rects: dict[str, Any] = {}

        # Pygame handles
        self.pygame_module: Any = None
        self.screen: Any = None
        self.clock: Any = None
        self.fonts: dict[str, Any] = {}
        self.scene_baseline: Optional[VisualTrafficScene] = None
        self.scene_rl: Optional[VisualTrafficScene] = None
        self.scene_single: Optional[VisualTrafficScene] = None

        self.setup_environments(self.scenario_id, self.seed)

        if not self.headless:
            self._init_pygame()

    def _build_baseline_agent(self, key: str) -> BaseAgent:
        if key == "fixed_time":
            return FixedTimeAgent(cycle_green_time=30.0, name="Fixed-Time")
        return MaxPressureAgent(theta_diff=3.0, name="Max-Pressure")

    def _load_ai_agent(self, key: str, scenario_id: str) -> BaseAgent:
        """Instantiate AI agent in PyTorch (.pth) or ONNXRuntime (.onnx) mode with graceful fallback."""
        models_dir = os.path.join(PROJECT_ROOT, "models")
        want_onnx = (self.inference_engine == "onnx") and (key in ("ppo", "ddqn"))

        if want_onnx:
            onnx_candidates = [
                os.path.join(models_dir, f"{key}_MIXED_best.onnx"),
                os.path.join(models_dir, f"{key}_{scenario_id}_best.onnx"),
                os.path.join(models_dir, f"{key}_MIXED.onnx"),
                os.path.join(models_dir, f"{key}_{scenario_id}.onnx"),
            ]
            for onnx_path in onnx_candidates:
                if os.path.exists(onnx_path):
                    try:
                        disp = "PPO-ONNX" if key == "ppo" else "DDQN-ONNX"
                        ag = OnnxRuntimeAgent(onnx_path=onnx_path, name=disp)
                        self.active_engine_label = "ONNX"
                        self.ai_checkpoint_status = _format_compact_status(key, os.path.basename(onnx_path), "onnx")
                        return ag
                    except Exception:
                        continue

        self.active_engine_label = "PT" if key in ("ppo", "ddqn") else "TBL"
        if key == "ppo":
            agent: BaseAgent = PPOAgent(device="cpu", name="PPO-Clip")
        elif key == "ddqn":
            agent = DQNAgent(use_double_dqn=True, device="cpu", name="Double-DQN")
        else:
            agent = QLearningAgent(name="Q-Learning")

        pth_candidates = [
            os.path.join(models_dir, f"{key}_MIXED_best.pth"),
            os.path.join(models_dir, f"{key}_{scenario_id}_best.pth"),
            os.path.join(models_dir, f"{key}_MIXED.pth"),
            os.path.join(models_dir, f"{key}_{scenario_id}.pth"),
            os.path.join(models_dir, f"{key}_SC3_ASYMMETRIC_best.pth"),
        ]
        loaded_from = None
        for path in pth_candidates:
            if os.path.exists(path):
                try:
                    agent.load(path)
                    loaded_from = os.path.basename(path)
                    break
                except Exception:
                    continue

        self.ai_checkpoint_status = _format_compact_status(key, loaded_from, "pytorch")
        return agent

    def _init_pygame(self) -> None:
        try:
            import pygame

            self.pygame_module = pygame
            pygame.init()
            pygame.display.set_caption("SmartTraffic RL Studio — 60 FPS Inference & A/B Lab (1360x760)")
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.clock = pygame.time.Clock()

            pygame.font.init()
            self.fonts = {
                "title": pygame.font.SysFont("segoeui", 16, bold=True),
                "section": pygame.font.SysFont("segoeui", 13, bold=True),
                "body": pygame.font.SysFont("segoeui", 13),
                "small": pygame.font.SysFont("segoeui", 11),
                "bold": pygame.font.SysFont("segoeui", 12, bold=True),
                "mono": pygame.font.SysFont("consolas", 12, bold=True),
            }

            self._init_scenes()
        except Exception as e:
            print(f"[SplitScreenStudio] Falling back to headless mode: {e}")
            self.headless = True

    def _init_scenes(self) -> None:
        if self.headless or not self.pygame_module:
            return
        self.scene_baseline = VisualTrafficScene(0, 0, LEFT_PANE_WIDTH, WINDOW_HEIGHT, road_w=120)
        self.scene_rl = VisualTrafficScene(LEFT_PANE_WIDTH, 0, MID_PANE_WIDTH, WINDOW_HEIGHT, road_w=120)
        self.scene_single = VisualTrafficScene(0, 0, LEFT_PANE_WIDTH + MID_PANE_WIDTH, WINDOW_HEIGHT, road_w=156)

        self.scene_baseline.reset(self.info_baseline)
        self.scene_rl.reset(self.info_rl)
        self.scene_single.reset(self.info_rl)

    def setup_environments(self, scenario_id: str, seed: int = 42) -> None:
        """Reset both Baseline and AI environments with identical synchronized vehicle seeds."""
        self.scenario_id = scenario_id
        self.seed = int(seed)
        self.live_rates_active = False
        self._base_finished = False
        self._rl_finished = False

        if not self._custom_rl:
            self.rl_agent = self._load_ai_agent(self.ai_key, self.scenario_id)

        gen_base = TrafficGenerator(scenario_id=scenario_id, seed=self.seed)
        gen_base.set_spawner_enabled(self.spawner_enabled)
        gen_rl = gen_base.clone_synchronized_generator()

        self.env_baseline = TrafficLightEnv(scenario_id=scenario_id, seed=self.seed, traffic_generator=gen_base)
        self.env_rl = TrafficLightEnv(scenario_id=scenario_id, seed=self.seed, traffic_generator=gen_rl)

        self.obs_baseline, self.info_baseline = self.env_baseline.reset(seed=self.seed)
        self.obs_rl, self.info_rl = self.env_rl.reset(seed=self.seed)

        self.baseline_agent.reset()
        self.baseline_agent_right.reset()
        self.rl_agent.reset()

        rates = self.info_rl.get("current_rates", {"N": 0.1, "S": 0.1, "E": 0.1, "W": 0.1})
        self.slider_rates = [rates["N"], rates["S"], rates["E"], rates["W"]]

        if self.scene_baseline:
            self.scene_baseline.reset(self.info_baseline)
        if self.scene_rl:
            self.scene_rl.reset(self.info_rl)
        if self.scene_single:
            self.scene_single.reset(self.info_rl)

    def toggle_spawner(self) -> None:
        self.spawner_enabled = not self.spawner_enabled
        t_sync = max(self.env_baseline.sim_time, self.env_rl.sim_time)
        self.env_baseline.traffic_generator.set_spawner_enabled(self.spawner_enabled, sync_from_time=t_sync)
        self.env_rl.traffic_generator.set_spawner_enabled(self.spawner_enabled, sync_from_time=t_sync)

    def apply_live_slider_rates(self) -> None:
        self.live_rates_active = True
        rate_dict = {
            NORTH: float(self.slider_rates[0]),
            SOUTH: float(self.slider_rates[1]),
            EAST: float(self.slider_rates[2]),
            WEST: float(self.slider_rates[3]),
        }
        t_sync = max(self.env_baseline.sim_time, self.env_rl.sim_time)
        self.env_baseline.traffic_generator.set_live_rates(rate_dict, sync_from_time=t_sync)
        self.env_rl.traffic_generator.set_live_rates(rate_dict, sync_from_time=t_sync)

    def toggle_inference_engine(self) -> None:
        """Hot-swap AI backend between PyTorch (.pth) and ONNXRuntime (.onnx)."""
        self.inference_engine = "onnx" if self.inference_engine == "pytorch" else "pytorch"
        if not self._custom_rl:
            self.rl_agent = self._load_ai_agent(self.ai_key, self.scenario_id)
            self.rl_agent.reset()

    def step(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Advance environments synchronized strictly so |env_baseline.sim_time - env_rl.sim_time| <= 5s."""
        t_b = self.env_baseline.sim_time
        t_r = self.env_rl.sim_time

        def _exec_base(preselected_action: Optional[int] = None) -> None:
            action_b = (
                preselected_action
                if preselected_action is not None
                else self.baseline_agent.select_action(self.obs_baseline, self.info_baseline, eval_mode=True)
            )
            self.obs_baseline, _, term_b, trunc_b, self.info_baseline = self.env_baseline.step(action_b)
            if term_b or trunc_b:
                self._base_finished = True
            if self.scene_baseline:
                self.scene_baseline.sync_from_info(self.info_baseline)

        def _exec_rl(preselected_action: Optional[int] = None) -> None:
            active_ctrl = self.rl_agent if self.use_ai else self.baseline_agent_right
            action_r = (
                preselected_action
                if preselected_action is not None
                else active_ctrl.select_action(self.obs_rl, self.info_rl, eval_mode=True)
            )
            self.obs_rl, _, term_r, trunc_r, self.info_rl = self.env_rl.step(action_r)
            if term_r or trunc_r:
                self._rl_finished = True
            if not self.live_rates_active:
                rates = self.info_rl.get("current_rates", {})
                if rates:
                    self.slider_rates = [
                        rates.get("N", 0.0),
                        rates.get("S", 0.0),
                        rates.get("E", 0.0),
                        rates.get("W", 0.0),
                    ]
            if self.scene_rl:
                self.scene_rl.sync_from_info(self.info_rl)
            if self.scene_single:
                self.scene_single.sync_from_info(self.info_rl)

        if self._base_finished and not self._rl_finished:
            _exec_rl()
        elif self._rl_finished and not self._base_finished:
            _exec_base()
        elif not self._base_finished and not self._rl_finished:
            if abs(t_b - t_r) <= 1e-6:
                _exec_base()
                _exec_rl()
            elif t_b < t_r:
                act_b = self.baseline_agent.select_action(self.obs_baseline, self.info_baseline, eval_mode=True)
                mask_b = self.info_baseline.get("action_mask", [1, 1])
                dt_b = 8.0 if (act_b == 1 and int(mask_b[1]) == 1) else 5.0
                _exec_base(preselected_action=act_b)
                if (t_b + dt_b) - t_r > 5.0:
                    _exec_rl()
            else:
                active_ctrl = self.rl_agent if self.use_ai else self.baseline_agent_right
                act_r = active_ctrl.select_action(self.obs_rl, self.info_rl, eval_mode=True)
                mask_r = self.info_rl.get("action_mask", [1, 1])
                dt_r = 8.0 if (act_r == 1 and int(mask_r[1]) == 1) else 5.0
                _exec_rl(preselected_action=act_r)
                if (t_r + dt_r) - t_b > 5.0:
                    _exec_base()

        if self._base_finished and self._rl_finished:
            self.setup_environments(self.scenario_id, self.seed + 1)

        return self.info_baseline, self.info_rl

    def handle_events(self) -> bool:
        if self.headless or not self.pygame_module:
            return True

        pygame = self.pygame_module
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
                elif event.key == pygame.K_v:
                    self.view_mode = "single" if self.view_mode == "split" else "split"
                elif event.key == pygame.K_i:
                    self.use_ai = not self.use_ai
                elif event.key == pygame.K_s:
                    self.toggle_spawner()
                elif event.key == pygame.K_h:
                    self.show_hud_overlay = not self.show_hud_overlay
                elif event.key == pygame.K_o:
                    self.toggle_inference_engine()
                elif event.key == pygame.K_r:
                    self.setup_environments(self.scenario_id, self.seed + 1)
                elif event.key == pygame.K_1:
                    self.setup_environments("SC1_LOW", self.seed)
                elif event.key == pygame.K_2:
                    self.setup_environments("SC2_HIGH", self.seed)
                elif event.key == pygame.K_3:
                    self.setup_environments("SC3_ASYMMETRIC", self.seed)
                elif event.key == pygame.K_4:
                    self.setup_environments("SC4_DYNAMIC_SURGE", self.seed)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_mouse_down(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging_slider_idx = None
            elif event.type == pygame.MOUSEMOTION and self.dragging_slider_idx is not None:
                self._update_slider_from_x(self.dragging_slider_idx, event.pos[0])

        return True

    def _update_slider_from_x(self, idx: int, mouse_x: int) -> None:
        slider_rect = self.ui_rects.get(f"slider_{idx}")
        if not slider_rect:
            return
        ratio = max(0.0, min(1.0, (mouse_x - slider_rect.x) / max(1, slider_rect.w)))
        self.slider_rates[idx] = round(ratio * 0.60, 2)
        self.apply_live_slider_rates()

    def _handle_mouse_down(self, pos: tuple[int, int]) -> None:
        for i in range(4):
            hit_rect = self.ui_rects.get(f"slider_hit_{i}")
            if hit_rect and hit_rect.collidepoint(pos):
                self.dragging_slider_idx = i
                self._update_slider_from_x(i, pos[0])
                return

        if "btn_view_mode" in self.ui_rects and self.ui_rects["btn_view_mode"].collidepoint(pos):
            self.view_mode = "single" if self.view_mode == "split" else "split"
        elif "chk_use_ai" in self.ui_rects and self.ui_rects["chk_use_ai"].collidepoint(pos):
            self.use_ai = not self.use_ai
        elif "btn_spawner" in self.ui_rects and self.ui_rects["btn_spawner"].collidepoint(pos):
            self.toggle_spawner()
        elif "btn_reset_car" in self.ui_rects and self.ui_rects["btn_reset_car"].collidepoint(pos):
            self.setup_environments(self.scenario_id, self.seed + 1)
        elif "btn_toggle_hud" in self.ui_rects and self.ui_rects["btn_toggle_hud"].collidepoint(pos):
            self.show_hud_overlay = not self.show_hud_overlay
        elif "btn_engine" in self.ui_rects and self.ui_rects["btn_engine"].collidepoint(pos):
            self.toggle_inference_engine()
        elif "btn_auto_rates" in self.ui_rects and self.ui_rects["btn_auto_rates"].collidepoint(pos):
            self.live_rates_active = False
            t_sync = max(self.env_baseline.sim_time, self.env_rl.sim_time)
            self.env_baseline.traffic_generator.set_live_rates(None, sync_from_time=t_sync)
            self.env_rl.traffic_generator.set_live_rates(None, sync_from_time=t_sync)
        elif "btn_pause" in self.ui_rects and self.ui_rects["btn_pause"].collidepoint(pos):
            self.is_paused = not self.is_paused
        else:
            for k in self.AI_KEYS:
                r = self.ui_rects.get(f"btn_ai_{k}")
                if r and r.collidepoint(pos):
                    self.ai_key = k
                    self._custom_rl = False
                    self.rl_agent = self._load_ai_agent(self.ai_key, self.scenario_id)
                    self.rl_agent.reset()
                    self.use_ai = True
                    return

            for k in self.BASE_KEYS:
                r = self.ui_rects.get(f"btn_base_{k}")
                if r and r.collidepoint(pos):
                    self.base_key = k
                    self._custom_baseline = False
                    self.baseline_agent = self._build_baseline_agent(self.base_key)
                    self.baseline_agent_right = self._build_baseline_agent(self.base_key)
                    self.baseline_agent.reset()
                    self.baseline_agent_right.reset()
                    return

            for sc in ("SC1_LOW", "SC2_HIGH", "SC3_ASYMMETRIC", "SC4_DYNAMIC_SURGE"):
                r = self.ui_rects.get(f"btn_sc_{sc}")
                if r and r.collidepoint(pos):
                    self.setup_environments(sc, self.seed)
                    return

            for spd in (1, 2, 5):
                r = self.ui_rects.get(f"btn_spd_{spd}")
                if r and r.collidepoint(pos):
                    self.sim_speed = spd
                    return

    def _draw_banner(
        self,
        surface: Any,
        offset_x: int,
        width: int,
        title: str,
        subtitle: str,
        accent: tuple[int, int, int],
    ) -> None:
        pygame = self.pygame_module
        rect = pygame.Rect(offset_x + 12, 10, width - 24, 42)
        pygame.draw.rect(surface, (15, 23, 36), rect, border_radius=8)
        pygame.draw.rect(surface, accent, rect, 1, border_radius=8)
        pygame.draw.circle(surface, accent, (rect.x + 16, rect.centery), 5)
        t_surf = self.fonts["title"].render(title, True, COLOR_TEXT_PRIMARY)
        s_surf = self.fonts["small"].render(subtitle, True, COLOR_TEXT_MUTED)
        surface.blit(t_surf, (rect.x + 30, rect.y + 5))
        surface.blit(s_surf, (rect.right - s_surf.get_width() - 12, rect.y + 14))

    def _draw_button(
        self,
        surface: Any,
        rect: Any,
        text: str,
        active: bool = False,
        active_color: tuple[int, int, int] = (59, 130, 246),
        idle_color: tuple[int, int, int] = (28, 38, 54),
        border_color: tuple[int, int, int] = (55, 72, 95),
    ) -> None:
        pygame = self.pygame_module
        bg = active_color if active else idle_color
        bc = active_color if active else border_color
        pygame.draw.rect(surface, bg, rect, border_radius=6)
        pygame.draw.rect(surface, bc, rect, 1, border_radius=6)
        txt = self.fonts["small"].render(text, True, COLOR_TEXT_PRIMARY)
        surface.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

    def render_hud_panel(self, surface: Any) -> None:
        """Render interactive Right HUD & 1-Minute Rolling Telemetry Dashboard (340 x 760)."""
        pygame = self.pygame_module
        x0 = LEFT_PANE_WIDTH + MID_PANE_WIDTH  # 1020
        pw = RIGHT_PANE_WIDTH                  # 340
        pygame.draw.rect(surface, (15, 23, 36), (x0, 0, pw, WINDOW_HEIGHT))
        pygame.draw.line(surface, (48, 66, 83), (x0, 0), (x0, WINDOW_HEIGHT), 2)

        self.ui_rects.clear()

        def lbl(text: str, x: int, y: int, color: tuple[int, int, int], font: str = "small") -> None:
            surface.blit(self.fonts[font].render(text, True, color), (x, y))

        # 1. Top Title & View Mode Toggle
        lbl("SMARTTRAFFIC HUD", x0 + 12, 13, COLOR_TEXT_PRIMARY, "title")
        mode_rect = pygame.Rect(x0 + 188, 10, 140, 26)
        self.ui_rects["btn_view_mode"] = mode_rect
        self._draw_button(
            surface,
            mode_rect,
            "Chế độ: Split A/B" if self.view_mode == "split" else "Chế độ: Single View",
            active=True,
            active_color=(37, 99, 235),
        )
        pygame.draw.line(surface, (42, 58, 78), (x0 + 12, 42), (x0 + pw - 12, 42), 1)

        # 2. Interactive Controls Card (y=48..224)
        ctrl_rect = pygame.Rect(x0 + 10, 48, pw - 20, 174)
        pygame.draw.rect(surface, (23, 33, 49), ctrl_rect, border_radius=8)
        pygame.draw.rect(surface, (45, 62, 85), ctrl_rect, 1, border_radius=8)

        # Row A: Checkbox [x] Dùng AI + Compact Checkpoint status (anti-overflow)
        chk_rect = pygame.Rect(x0 + 18, 56, 88, 24)
        self.ui_rects["chk_use_ai"] = chk_rect
        box_bg = (34, 197, 94) if self.use_ai else (35, 45, 62)
        pygame.draw.rect(surface, (30, 42, 60), chk_rect, border_radius=5)
        pygame.draw.rect(surface, box_bg, (chk_rect.x + 6, chk_rect.y + 5, 14, 14), border_radius=3)
        if self.use_ai:
            pygame.draw.line(surface, (15, 23, 36), (chk_rect.x + 9, chk_rect.y + 12), (chk_rect.x + 12, chk_rect.y + 16), 2)
            pygame.draw.line(surface, (15, 23, 36), (chk_rect.x + 12, chk_rect.y + 16), (chk_rect.x + 17, chk_rect.y + 8), 2)
        lbl("Dùng AI", chk_rect.x + 25, chk_rect.y + 4, COLOR_TEXT_PRIMARY, "bold")
        lbl(self.ai_checkpoint_status, x0 + 112, 60, (148, 163, 184), "small")

        # Row B: AI Selector (PPO | DDQN | Q-Learn) + Baseline (MaxPress | Fixed30s)
        bx = x0 + 18
        for k in self.AI_KEYS:
            r = pygame.Rect(bx, 86, 56, 22)
            self.ui_rects[f"btn_ai_{k}"] = r
            self._draw_button(surface, r, self.AI_LABELS[k], active=(self.ai_key == k), active_color=COLOR_ACCENT_RL)
            bx += 60
        for k in self.BASE_KEYS:
            r = pygame.Rect(bx + 6, 86, 62, 22)
            self.ui_rects[f"btn_base_{k}"] = r
            self._draw_button(surface, r, self.BASE_LABELS[k], active=(self.base_key == k), active_color=COLOR_ACCENT_BASELINE)
            bx += 66

        # Row C: SpawnerCar: ON/OFF | ResetCar | Ẩn/Hiện TT
        r_sp = pygame.Rect(x0 + 18, 114, 108, 26)
        r_rs = pygame.Rect(x0 + 131, 114, 86, 26)
        r_hd = pygame.Rect(x0 + 222, 114, 100, 26)
        self.ui_rects["btn_spawner"] = r_sp
        self.ui_rects["btn_reset_car"] = r_rs
        self.ui_rects["btn_toggle_hud"] = r_hd
        self._draw_button(
            surface,
            r_sp,
            f"SpawnerCar: {'ON' if self.spawner_enabled else 'OFF'}",
            active=self.spawner_enabled,
            active_color=(22, 163, 74),
            idle_color=(153, 27, 27),
        )
        self._draw_button(surface, r_rs, "ResetCar", active=False)
        self._draw_button(
            surface,
            r_hd,
            "Ẩn TT" if self.show_hud_overlay else "Hiện TT",
            active=self.show_hud_overlay,
            active_color=(14, 116, 144),
        )

        # Row D: Scenarios SC1 - SC4
        sc_items = [("SC1_LOW", "SC1 Low"), ("SC2_HIGH", "SC2 High"), ("SC3_ASYMMETRIC", "SC3 Lệch"), ("SC4_DYNAMIC_SURGE", "SC4 Surge")]
        for idx, (sc_id, sc_lbl) in enumerate(sc_items):
            r = pygame.Rect(x0 + 18 + idx * 77, 148, 73, 26)
            self.ui_rects[f"btn_sc_{sc_id}"] = r
            self._draw_button(surface, r, sc_lbl, active=(self.scenario_id == sc_id), active_color=(56, 189, 248))

        # Row E: Speed 1x | 2x | 5x + Engine (PT/ONNX) + Pause
        for idx, spd in enumerate((1, 2, 5)):
            r = pygame.Rect(x0 + 18 + idx * 44, 182, 40, 28)
            self.ui_rects[f"btn_spd_{spd}"] = r
            self._draw_button(surface, r, f"{spd}x", active=(self.sim_speed == spd), active_color=(147, 51, 234))

        r_eng = pygame.Rect(x0 + 154, 182, 82, 28)
        self.ui_rects["btn_engine"] = r_eng
        is_onnx_active = (self.active_engine_label == "ONNX")
        self._draw_button(
            surface,
            r_eng,
            f"Eng: {self.active_engine_label}",
            active=is_onnx_active,
            active_color=(16, 185, 129),
            idle_color=(30, 41, 59),
        )

        r_pause = pygame.Rect(x0 + 240, 182, 82, 28)
        self.ui_rects["btn_pause"] = r_pause
        self._draw_button(
            surface,
            r_pause,
            "▶ Tiếp" if self.is_paused else "⏸ Dừng",
            active=self.is_paused,
            active_color=(217, 119, 6),
        )

        # 3. Live Traffic Injection Sliders (y=230..374)
        sl_card = pygame.Rect(x0 + 10, 230, pw - 20, 142)
        pygame.draw.rect(surface, (23, 33, 49), sl_card, border_radius=8)
        pygame.draw.rect(surface, (45, 62, 85), sl_card, 1, border_radius=8)

        lbl("LIVE INJECTION (λ xe/s)", x0 + 18, 238, COLOR_TEXT_PRIMARY, "bold")
        r_auto = pygame.Rect(x0 + 252, 236, 70, 20)
        self.ui_rects["btn_auto_rates"] = r_auto
        self._draw_button(surface, r_auto, "Auto SC" if self.live_rates_active else "Theo KB", active=not self.live_rates_active, active_color=(30, 100, 130))

        dir_labels = ["Bắc (λ_B)", "Nam (λ_N)", "Đông (λ_Đ)", "Tây (λ_T)"]
        for idx, dname in enumerate(dir_labels):
            sy = 266 + idx * 26
            lbl(dname, x0 + 18, sy + 2, COLOR_TEXT_MUTED, "small")

            track = pygame.Rect(x0 + 92, sy + 7, 172, 6)
            hit = pygame.Rect(x0 + 88, sy, 180, 22)
            self.ui_rects[f"slider_{idx}"] = track
            self.ui_rects[f"slider_hit_{idx}"] = hit

            pygame.draw.rect(surface, (45, 58, 78), track, border_radius=3)
            val = float(self.slider_rates[idx])
            fill_w = int((min(0.60, max(0.0, val)) / 0.60) * track.w)
            pygame.draw.rect(surface, (56, 189, 248) if self.live_rates_active else (74, 222, 128), (track.x, track.y, fill_w, track.h), border_radius=3)
            knob_x = track.x + fill_w
            pygame.draw.circle(surface, COLOR_TEXT_PRIMARY, (knob_x, track.centery), 6)

            lbl(f"{val:.2f}", x0 + 274, sy + 1, COLOR_TEXT_PRIMARY, "mono")

        # 4. Telemetry Card: === 1 phút gần nhất === (y=380..748)
        tel_card = pygame.Rect(x0 + 10, 380, pw - 20, 368)
        pygame.draw.rect(surface, (23, 33, 49), tel_card, border_radius=8)
        pygame.draw.rect(surface, (45, 62, 85), tel_card, 1, border_radius=8)

        sim_t_b = int(self.info_baseline.get("sim_time", self.env_baseline.current_time))
        sim_t_r = int(self.info_rl.get("sim_time", self.env_rl.current_time))
        lbl("=== 1 phút gần nhất ===", x0 + 18, 388, (56, 189, 248), "section")
        if abs(sim_t_b - sim_t_r) >= 1 and self.view_mode == "split":
            clk_txt = f"B={sim_t_b // 60:02d}:{sim_t_b % 60:02d} | AI={sim_t_r // 60:02d}:{sim_t_r % 60:02d}"
        else:
            clk_txt = f"T={sim_t_r // 60:02d}:{sim_t_r % 60:02d}/30:00 | S{self.seed}"
        lbl(clk_txt, x0 + 196, 390, COLOR_TEXT_MUTED, "small")

        lbl("CHỈ SỐ", x0 + 18, 414, COLOR_TEXT_MUTED, "small")
        lbl("BASELINE", x0 + 146, 414, COLOR_ACCENT_BASELINE, "bold")
        lbl("AI / CTRL", x0 + 236, 414, COLOR_ACCENT_RL, "bold")
        pygame.draw.line(surface, (42, 58, 78), (x0 + 18, 432), (x0 + pw - 18, 432), 1)

        p60_b = int(self.info_baseline.get("passed_last_60s", 0))
        p60_ai = int(self.info_rl.get("passed_last_60s", 0))
        tot_b = int(self.info_baseline.get("total_passed", 0))
        tot_ai = int(self.info_rl.get("total_passed", 0))
        wait_b = float(self.info_baseline.get("avg_wait_time", 0.0))
        wait_ai = float(self.info_rl.get("avg_wait_time", 0.0))
        maxw_b = float(self.info_baseline.get("max_wait_recorded", 0.0))
        maxw_ai = float(self.info_rl.get("max_wait_recorded", 0.0))
        tau_b = float(self.info_baseline.get("phase_duration", 0.0))
        tau_ai = float(self.info_rl.get("phase_duration", 0.0))
        ph_b = "B-N" if self.info_baseline.get("phase", 0) == PHASE_NS else "Đ-T"
        ph_ai = "B-N" if self.info_rl.get("phase", 0) == PHASE_NS else "Đ-T"
        m_b = list(self.info_baseline.get("action_mask", [1, 1]))
        m_ai = list(self.info_rl.get("action_mask", [1, 1]))

        rows = [
            ("Xe qua (1p):", f"{p60_b} xe", f"{p60_ai} xe"),
            ("Tổng xe qua:", f"{tot_b} xe", f"{tot_ai} xe"),
            ("Chờ TB (s):", f"{wait_b:.1f}s", f"{wait_ai:.1f}s"),
            ("Chờ Max (s):", f"{maxw_b:.1f}s", f"{maxw_ai:.1f}s"),
            ("Đèn xanh:", f"{ph_b} {tau_b:.0f}s", f"{ph_ai} {tau_ai:.0f}s"),
            ("Action Mask:", f"{m_b}", f"{m_ai}"),
        ]

        ry = 442
        for k_txt, v_b, v_ai in rows:
            lbl(k_txt, x0 + 18, ry, COLOR_TEXT_MUTED, "body")
            lbl(v_b, x0 + 146, ry, COLOR_TEXT_PRIMARY, "mono")
            lbl(v_ai, x0 + 236, ry, COLOR_TEXT_PRIMARY, "mono")
            ry += 28

        pct_wait_reduce = ((wait_b - wait_ai) / max(0.1, wait_b)) * 100.0
        pct_tp_gain = ((tot_ai - tot_b) / max(1, tot_b)) * 100.0

        d_card1 = pygame.Rect(x0 + 18, 622, pw - 36, 48)
        w_win = pct_wait_reduce >= 0.0
        pygame.draw.rect(surface, (18, 52, 38) if w_win else (58, 24, 30), d_card1, border_radius=7)
        pygame.draw.rect(surface, COLOR_GREEN_LIGHT if w_win else COLOR_RED_LIGHT, d_card1, 1, border_radius=7)
        lbl("% Giảm Thời gian Chờ (AI vs Base)", d_card1.x + 10, d_card1.y + 6, COLOR_TEXT_MUTED, "small")
        lbl(f"{pct_wait_reduce:+.1f}% ({wait_ai - wait_b:+.1f}s)", d_card1.x + 10, d_card1.y + 23, COLOR_GREEN_LIGHT if w_win else COLOR_RED_LIGHT, "bold")

        d_card2 = pygame.Rect(x0 + 18, 678, pw - 36, 48)
        t_win = pct_tp_gain >= 0.0
        pygame.draw.rect(surface, (18, 52, 38) if t_win else (58, 24, 30), d_card2, border_radius=7)
        pygame.draw.rect(surface, COLOR_GREEN_LIGHT if t_win else COLOR_RED_LIGHT, d_card2, 1, border_radius=7)
        lbl("% Tăng Thông lượng (AI vs Base)", d_card2.x + 10, d_card2.y + 6, COLOR_TEXT_MUTED, "small")
        lbl(f"{pct_tp_gain:+.1f}% ({tot_ai - tot_b:+d} xe)", d_card2.x + 10, d_card2.y + 23, COLOR_GREEN_LIGHT if t_win else COLOR_RED_LIGHT, "bold")

    def render(self) -> None:
        if self.headless or not self.screen:
            return

        pygame = self.pygame_module
        self.screen.fill(COLOR_BG)
        t_b = int(self.info_baseline.get("sim_time", self.env_baseline.sim_time))
        t_r = int(self.info_rl.get("sim_time", self.env_rl.sim_time))

        if self.view_mode == "split":
            if self.scene_baseline:
                self.scene_baseline.draw(self.screen, self.info_baseline, self.fonts, show_overlay=self.show_hud_overlay)
                self._draw_banner(
                    self.screen,
                    offset_x=0,
                    width=LEFT_PANE_WIDTH,
                    title=f"BASELINE: {self.baseline_agent.name}",
                    subtitle=f"SYNCHRONIZED · t={t_b}s",
                    accent=COLOR_ACCENT_BASELINE,
                )
            pygame.draw.line(self.screen, COLOR_PANE_BORDER, (LEFT_PANE_WIDTH, 0), (LEFT_PANE_WIDTH, WINDOW_HEIGHT), 2)

            if self.scene_rl:
                self.scene_rl.draw(self.screen, self.info_rl, self.fonts, show_overlay=self.show_hud_overlay)
                ctrl_name = self.rl_agent.name if self.use_ai else f"{self.baseline_agent_right.name} (AI OFF)"
                eng_str = self.active_engine_label if self.use_ai else "RULE"
                self._draw_banner(
                    self.screen,
                    offset_x=LEFT_PANE_WIDTH,
                    width=MID_PANE_WIDTH,
                    title=f"AI CONTROLLER: {ctrl_name}",
                    subtitle=f"INFERENCE ({eng_str}) · t={t_r}s",
                    accent=COLOR_ACCENT_RL if self.use_ai else COLOR_ACCENT_BASELINE,
                )
        else:
            if self.scene_single:
                self.scene_single.draw(self.screen, self.info_rl, self.fonts, show_overlay=self.show_hud_overlay)
                ctrl_title = f"CHẾ ĐỘ ĐƠN — ĐANG DÙNG AI: {self.rl_agent.name}" if self.use_ai else f"CHẾ ĐỘ ĐƠN — CỐ ĐỊNH: {self.baseline_agent_right.name}"
                self._draw_banner(
                    self.screen,
                    offset_x=0,
                    width=LEFT_PANE_WIDTH + MID_PANE_WIDTH,
                    title=ctrl_title,
                    subtitle=f"Backend: {self.active_engine_label} | Nhấn [x] Dùng AI để chuyển nóng",
                    accent=COLOR_ACCENT_RL if self.use_ai else COLOR_ACCENT_BASELINE,
                )

        self.render_hud_panel(self.screen)
        pygame.display.flip()

    def run(self, max_steps: Optional[int] = None) -> None:
        self.is_running = True
        step_idx = 0
        last_step_ticks = self.pygame_module.time.get_ticks() if self.pygame_module and not self.headless else 0
        last_frame_ticks = last_step_ticks

        while self.is_running:
            if not self.handle_events():
                break

            now = self.pygame_module.time.get_ticks() if self.pygame_module and not self.headless else 0

            if self.pygame_module and not self.headless:
                dt = (now - last_frame_ticks) / 1000.0
                last_frame_ticks = now
                if dt > 0.08:
                    dt = 0.016
                if self.scene_baseline:
                    self.scene_baseline.update(dt, speed_mult=float(self.sim_speed))
                if self.scene_rl:
                    self.scene_rl.update(dt, speed_mult=float(self.sim_speed))
                if self.scene_single:
                    self.scene_single.update(dt, speed_mult=float(self.sim_speed))

            if not self.is_paused:
                if self.headless:
                    self.step()
                    step_idx += 1
                    if max_steps and step_idx >= max_steps:
                        self.is_running = False
                        break
                else:
                    step_interval_ms = max(80, 900 // max(1, self.sim_speed))
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
        if self.pygame_module and not self.headless:
            try:
                self.pygame_module.quit()
            except Exception:
                pass
        self.env_baseline.close()
        self.env_rl.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SmartTraffic RL Studio (1360x760 Inference & HUD)")
    parser.add_argument(
        "--scenario",
        type=str,
        default="SC3_ASYMMETRIC",
        choices=["SC1_LOW", "SC2_HIGH", "SC3_ASYMMETRIC", "SC4_DYNAMIC_SURGE"],
        help="Initial traffic scenario",
    )
    parser.add_argument("--seed", type=int, default=42, help="Synchronized PRNG seed")
    parser.add_argument("--view", type=str, default="split", choices=["split", "single"], help="Initial view layout")
    parser.add_argument("--ai", type=str, default="ppo", choices=["ppo", "ddqn", "q_learning"], help="Default AI agent")
    parser.add_argument("--engine", type=str, default="pytorch", choices=["pytorch", "onnx"], help="Inference backend")
    parser.add_argument("--headless", action="store_true", help="Run off-screen headless loop")
    parser.add_argument("--steps", type=int, default=None, help="Max decision steps")
    args = parser.parse_args()

    print(f"Launching SmartTraffic RL Studio [View: {args.view}, AI: {args.ai}, Engine: {args.engine}, Scenario: {args.scenario}, Seed: {args.seed}]...")
    studio = SplitScreenStudio(
        scenario_id=args.scenario,
        seed=args.seed,
        headless=args.headless,
        view_mode=args.view,
        ai_key=args.ai,
        inference_engine=args.engine,
    )
    studio.run(max_steps=args.steps)
