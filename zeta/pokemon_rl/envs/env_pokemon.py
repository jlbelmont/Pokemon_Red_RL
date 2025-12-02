import os
from typing import Dict, Iterable, Optional, Set, Tuple

# Ensure SDL never attempts to output audio (disables in-game music/SFX).
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from pyboy import PyBoy
    try:
        from pyboy.utils import WindowEvent  # type: ignore[attr-defined]
    except ImportError:
        WindowEvent = None  # type: ignore[assignment]
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "PyBoy is required to run PokemonRedEnv. Install it with `pip install pyboy`."
    ) from exc


# Core WRAM addresses documented in the pret/pokered project.
# These values let us peek at the player's state without pausing emulation.
MAP_ID_ADDR = 0xD35E
PLAYER_Y_ADDR = 0xD361
PLAYER_X_ADDR = 0xD362
BATTLE_TYPE_ADDR = 0xD057  # Non-zero while in battle
BATTLE_RESULT_ADDR = 0xD059  # Encodes outcome once a battle ends
ENCOUNTER_SPECIES_ADDR = 0xD0B5
# Event flag byte containing "Beat Champion Rival" (bit 1).
CHAMPION_FLAG_ADDR = 0xD867
CHAMPION_FLAG_BIT = 1
BATTLE_TYPE_DETAIL_ADDR = 0xD057
ENEMY_HP_CURRENT_ADDR = 0xCFE6  # little-endian (lo, hi)
ENEMY_HP_MAX_ADDR = 0xCFE8  # little-endian (lo, hi)

# Bag and Pokédex metadata.
BAG_ITEMS_ADDR = 0xD31D  # alternating item id / quantity, terminated by 0xFF
MAX_BAG_SLOTS = 20
POKEDEX_OWNED_ADDR = 0xD2F7  # 19 bytes support all 151 species
POKEDEX_OWNED_BYTES = 19

# Lead Pokémon status/HP.
LEAD_STATUS_ADDR = 0xD16A
LEAD_HP_CURRENT_ADDR = 0xD16B  # little-endian (lo, hi)
LEAD_HP_MAX_ADDR = 0xD16D  # little-endian (lo, hi)

# Known overworld map ids pulled from pret/pokered's constants for coverage.
MAP_NAMES: Dict[int, str] = {
    0x00: "pallet town",
    0x01: "viridian city",
    0x02: "pewter city",
    0x03: "cerulean city",
    0x04: "lavender town",
    0x05: "vermilion city",
    0x06: "celadon city",
    0x07: "fuchsia city",
    0x08: "cinnabar island",
    0x09: "indigo plateau",
    0x0A: "saffron city",
    0x0B: "unused map 0b",
    0x0C: "route 1",
    0x0D: "route 2",
    0x0E: "route 3",
    0x0F: "route 4",
    0x10: "route 5",
    0x11: "route 6",
    0x12: "route 7",
    0x13: "route 8",
    0x14: "route 9",
    0x15: "route 10",
    0x16: "route 11",
    0x17: "route 12",
    0x18: "route 13",
    0x19: "route 14",
    0x1A: "route 15",
    0x1B: "route 16",
    0x1C: "route 17",
    0x1D: "route 18",
    0x1E: "route 19",
    0x1F: "route 20",
    0x20: "route 21",
    0x21: "route 22",
    0x22: "route 23",
    0x23: "route 24",
    0x24: "route 25",
    0x25: "reds house 1f",
    0x26: "reds house 2f",
    0x27: "blues house",
    0x28: "oaks lab",
    0x29: "viridian pokecenter",
    0x2A: "viridian mart",
    0x2B: "viridian school house",
    0x2C: "viridian nickname house",
    0x2D: "viridian gym",
    0x2E: "digletts cave route 2",
    0x2F: "viridian forest north gate",
    0x30: "route 2 trade house",
    0x31: "route 2 gate",
    0x32: "viridian forest south gate",
    0x33: "viridian forest",
    0x34: "museum 1f",
    0x35: "museum 2f",
    0x36: "pewter gym",
    0x37: "pewter nidoran house",
    0x38: "pewter mart",
    0x39: "pewter speech house",
    0x3A: "pewter pokecenter",
    0x3B: "mt moon 1f",
    0x3C: "mt moon b1f",
    0x3D: "mt moon b2f",
    0x3E: "cerulean trashed house",
    0x3F: "cerulean trade house",
    0x40: "cerulean pokecenter",
    0x41: "cerulean gym",
    0x42: "bike shop",
    0x43: "cerulean mart",
    0x44: "mt moon pokecenter",
    0x45: "cerulean trashed house copy",
    0x46: "route 5 gate",
    0x47: "underground path route 5",
    0x48: "daycare",
    0x49: "route 6 gate",
    0x4A: "underground path route 6",
    0x4B: "underground path route 6 copy",
    0x4C: "route 7 gate",
    0x4D: "underground path route 7",
    0x4E: "underground path route 7 copy",
    0x4F: "route 8 gate",
    0x50: "underground path route 8",
    0x51: "rock tunnel pokecenter",
    0x52: "rock tunnel 1f",
    0x53: "power plant",
    0x54: "route 11 gate 1f",
    0x55: "digletts cave route 11",
    0x56: "route 11 gate 2f",
    0x57: "route 12 gate 1f",
    0x58: "bills house",
    0x59: "vermilion pokecenter",
    0x5A: "pokemon fan club",
    0x5B: "vermilion mart",
    0x5C: "vermilion gym",
    0x5D: "vermilion pidgey house",
    0x5E: "vermilion dock",
    0x5F: "ss anne 1f",
    0x60: "ss anne 2f",
    0x61: "ss anne 3f",
    0x62: "ss anne b1f",
    0x63: "ss anne bow",
    0x64: "ss anne kitchen",
    0x65: "ss anne captains room",
    0x66: "ss anne 1f rooms",
    0x67: "ss anne 2f rooms",
    0x68: "ss anne b1f rooms",
    0x69: "unused map 69",
    0x6A: "unused map 6a",
    0x6B: "unused map 6b",
    0x6C: "victory road 1f",
    0x6D: "unused map 6d",
    0x6E: "unused map 6e",
    0x6F: "unused map 6f",
    0x70: "unused map 70",
    0x71: "lances room",
    0x72: "unused map 72",
    0x73: "unused map 73",
    0x74: "unused map 74",
    0x75: "unused map 75",
    0x76: "hall of fame",
    0x77: "underground path north south",
    0x78: "champions room",
    0x79: "underground path west east",
    0x7A: "celadon mart 1f",
    0x7B: "celadon mart 2f",
    0x7C: "celadon mart 3f",
    0x7D: "celadon mart 4f",
    0x7E: "celadon mart roof",
    0x7F: "celadon mart elevator",
    0x80: "celadon mansion 1f",
    0x81: "celadon mansion 2f",
    0x82: "celadon mansion 3f",
    0x83: "celadon mansion roof",
    0x84: "celadon mansion roof house",
    0x85: "celadon pokecenter",
    0x86: "celadon gym",
    0x87: "game corner",
    0x88: "celadon mart 5f",
    0x89: "game corner prize room",
    0x8A: "celadon diner",
    0x8B: "celadon chief house",
    0x8C: "celadon hotel",
    0x8D: "lavender pokecenter",
    0x8E: "pokemon tower 1f",
    0x8F: "pokemon tower 2f",
    0x90: "pokemon tower 3f",
    0x91: "pokemon tower 4f",
    0x92: "pokemon tower 5f",
    0x93: "pokemon tower 6f",
    0x94: "pokemon tower 7f",
    0x95: "mr fujis house",
    0x96: "lavender mart",
    0x97: "lavender cubone house",
    0x98: "fuchsia mart",
    0x99: "fuchsia bills grandpas house",
    0x9A: "fuchsia pokecenter",
    0x9B: "wardens house",
    0x9C: "safari zone gate",
    0x9D: "fuchsia gym",
    0x9E: "fuchsia meeting room",
    0x9F: "seafoam islands b1f",
    0xA0: "seafoam islands b2f",
    0xA1: "seafoam islands b3f",
    0xA2: "seafoam islands b4f",
    0xA3: "vermilion old rod house",
    0xA4: "fuchsia good rod house",
    0xA5: "pokemon mansion 1f",
    0xA6: "cinnabar gym",
    0xA7: "cinnabar lab",
    0xA8: "cinnabar lab trade room",
    0xA9: "cinnabar lab metronome room",
    0xAA: "cinnabar lab fossil room",
    0xAB: "cinnabar pokecenter",
    0xAC: "cinnabar mart",
    0xAD: "cinnabar mart copy",
    0xAE: "indigo plateau lobby",
    0xAF: "copycats house 1f",
    0xB0: "copycats house 2f",
    0xB1: "fighting dojo",
    0xB2: "saffron gym",
    0xB3: "saffron pidgey house",
    0xB4: "saffron mart",
    0xB5: "silph co 1f",
    0xB6: "saffron pokecenter",
    0xB7: "mr psychics house",
    0xB8: "route 15 gate 1f",
    0xB9: "route 15 gate 2f",
    0xBA: "route 16 gate 1f",
    0xBB: "route 16 gate 2f",
    0xBC: "route 16 fly house",
    0xBD: "route 12 super rod house",
    0xBE: "route 18 gate 1f",
    0xBF: "route 18 gate 2f",
    0xC0: "seafoam islands 1f",
    0xC1: "route 22 gate",
    0xC2: "victory road 2f",
    0xC3: "route 12 gate 2f",
    0xC4: "vermilion trade house",
    0xC5: "digletts cave",
    0xC6: "victory road 3f",
    0xC7: "rocket hideout b1f",
    0xC8: "rocket hideout b2f",
    0xC9: "rocket hideout b3f",
    0xCA: "rocket hideout b4f",
    0xCB: "rocket hideout elevator",
    0xCC: "unused map cc",
    0xCD: "unused map cd",
    0xCE: "unused map ce",
    0xCF: "silph co 2f",
    0xD0: "silph co 3f",
    0xD1: "silph co 4f",
    0xD2: "silph co 5f",
    0xD3: "silph co 6f",
    0xD4: "silph co 7f",
    0xD5: "silph co 8f",
    0xD6: "pokemon mansion 2f",
    0xD7: "pokemon mansion 3f",
    0xD8: "pokemon mansion b1f",
    0xD9: "safari zone east",
    0xDA: "safari zone north",
    0xDB: "safari zone west",
    0xDC: "safari zone center",
    0xDD: "safari zone center rest house",
    0xDE: "safari zone secret house",
    0xDF: "safari zone west rest house",
    0xE0: "safari zone east rest house",
    0xE1: "safari zone north rest house",
    0xE2: "cerulean cave 2f",
    0xE3: "cerulean cave b1f",
    0xE4: "cerulean cave 1f",
    0xE5: "name raters house",
    0xE6: "cerulean badge house",
    0xE7: "unused map e7",
    0xE8: "rock tunnel b1f",
    0xE9: "silph co 9f",
    0xEA: "silph co 10f",
    0xEB: "silph co 11f",
    0xEC: "silph co elevator",
    0xED: "unused map ed",
    0xEE: "unused map ee",
    0xEF: "trade center",
    0xF0: "colosseum",
    0xF1: "unused map f1",
    0xF2: "unused map f2",
    0xF3: "unused map f3",
    0xF4: "unused map f4",
    0xF5: "loreleis room",
    0xF6: "brunos room",
    0xF7: "agathas room",
}

# Gym badge bitfield is stored at 0xD356 in Pokémon Red/Blue (wPlayerBadgeFlags).
BADGE_FLAGS_ADDR = 0xD356
BADGE_NAMES = (
    "boulder",
    "cascade",
    "thunder",
    "rainbow",
    "soul",
    "marsh",
    "volcano",
    "earth",
)

# Mapping battle result values → readable labels.
BATTLE_RESULT_LABELS: Dict[int, str] = {
    0x00: "ongoing",
    0x01: "won",
    0x02: "lost",
    0x03: "blackout",
    0x04: "escaped",
    0x07: "caught",
}

_BATTLE_DEBUG_ADDRS: Dict[str, int] = {
    "battle_type_addr": BATTLE_TYPE_ADDR,
    "battle_result_addr": BATTLE_RESULT_ADDR,
    "battle_type_detail_addr": BATTLE_TYPE_DETAIL_ADDR,
    "encounter_species_addr": ENCOUNTER_SPECIES_ADDR,
    "enemy_hp_lo_addr": ENEMY_HP_CURRENT_ADDR,
    "enemy_hp_hi_addr": ENEMY_HP_CURRENT_ADDR + 1,
    "enemy_hp_max_lo_addr": ENEMY_HP_MAX_ADDR,
    "enemy_hp_max_hi_addr": ENEMY_HP_MAX_ADDR + 1,
}

_BATTLE_DEBUG_WINDOWS: Tuple[Tuple[str, int, int], ...] = (
    ("battle_info_d04b", 0xD04B, 0x20),
    ("battle_state_d050", 0xD050, 0x20),
    ("enemy_hp_block_cfe0", 0xCFE0, 0x40),
)


def _downsample_frame(frame: np.ndarray) -> np.ndarray:
    """Reduce the native 160x144 RGB frame to 72x80 grayscale for lightweight networks."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    resized = cv2.resize(gray, (80, 72), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)[..., None]


class PokemonRedEnv(gym.Env):
    """Minimal PyBoy-powered environment focused on Route 1 catch attempts."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        rom_path: Optional[str] = None,
        *,
        show_display: bool = False,
        frame_skip: int = 4,
        boot_steps: int = 120,
        max_no_input_frames: int = 60,
        state_path: Optional[str] = None,
        story_flag_defs: Optional[Dict[str, Dict[str, object]]] = None,
        track_visit_stats: bool = True,
        delete_sav_on_reset: bool = True,
        input_spacing_frames: int = 0,
    ) -> None:
        super().__init__()
        self.rom_path = rom_path or os.path.join(os.path.dirname(__file__), "pokemon_red.gb")
        if not os.path.exists(self.rom_path):
            raise FileNotFoundError(
                f"Pokemon ROM not found at {self.rom_path}. "
                "Place pokemon_red.gb inside epsilon/pokemon_rl or pass rom_path explicitly."
            )

        self.frame_skip = max(1, int(frame_skip))
        self.show_display = bool(show_display)
        self.boot_steps = max(0, int(boot_steps))
        self.max_no_input_frames = max(1, int(max_no_input_frames))
        self.state_path = state_path
        self.story_flag_defs = self._normalise_story_defs(story_flag_defs)
        self.track_visit_stats = bool(track_visit_stats)
        self.delete_sav_on_reset = bool(delete_sav_on_reset)
        self.input_spacing_frames = max(0, int(input_spacing_frames))

        self._pyboy: Optional[PyBoy] = None
        self._screen = None
        self._steps = 0
        self._frames_since_input = 0
        self._prev_in_battle = False
        self._last_battle_result: Optional[str] = None
        self._caught_flag = False
        self._last_raw_frame: Optional[np.ndarray] = None
        self._story_cache = {}
        self._global_visit_counts: Dict[Tuple[int, int], int] = {}
        self._episode_visits: Set[Tuple[int, int]] = set()
        self._episode_revisit_steps = 0
        self._pokemon_caught_total = 0
        self._pokemon_defeated_total = 0
        self._pokemon_caught_episode = 0
        self._pokemon_defeated_episode = 0
        self._battle_losses_total = 0
        self._battle_losses_episode = 0
        self._battle_blackouts_total = 0
        self._battle_blackouts_episode = 0
        self._battle_flees_total = 0
        self._battle_flees_episode = 0
        self._battle_result_cache_code: Optional[int] = None
        self._battle_result_cache_label: Optional[str] = None
        self._battle_enemy_hp_snapshot: Dict[str, int] = {}
        self._visited_towns: Set[str] = set()

        self._button_groups: Tuple[Tuple[str, ...], ...] = (
            tuple(),  # NOOP
            ("a",),
            ("b",),
            ("start",),
            ("select",),
            ("up",),
            ("down",),
            ("left",),
            ("right",),
        )
        self._button_event_table = self._build_button_event_table()

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(72, 80, 1),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(self._button_groups))

        self._initialize_emulator()

    @property
    def pyboy(self) -> PyBoy:
        assert self._pyboy is not None, "PyBoy has not been initialised."
        return self._pyboy

    def _initialize_emulator(self) -> None:
        """Create the PyBoy instance, optionally load a state, and warm up the ROM."""
        if self.delete_sav_on_reset:
            self._purge_save_file()
        window = "SDL2" if self.show_display else "null"
        if not os.path.isfile(self.rom_path):
            raise FileNotFoundError(
                f"Pokemon ROM not found at {self.rom_path}. Update config or set POKEMON_ROM_PATH."
            )
        try:
            rom_size = os.path.getsize(self.rom_path)
        except OSError as exc:
            raise RuntimeError(f"Unable to stat ROM at {self.rom_path}: {exc}") from exc
        if rom_size <= 0:
            raise RuntimeError(f"Pokemon ROM at {self.rom_path} is empty (size={rom_size}).")
        try:
            with open(self.rom_path, "rb") as rom_file:
                # Read a byte to ensure network mounts / permissions allow data access.
                sample = rom_file.read(1)
        except OSError as exc:
            raise RuntimeError(f"Failed to read Pokemon ROM at {self.rom_path}: {exc}") from exc
        if not sample:
            raise RuntimeError(f"Pokemon ROM at {self.rom_path} returned no data on read().")
        print(f"[env] Loading Pokemon ROM: {self.rom_path} ({rom_size / 1024:.1f} KiB)")
        try:
            self._pyboy = PyBoy(self.rom_path, window=window, sound=False)
        except TypeError:
            # Older PyBoy builds use window_type.
            self._pyboy = PyBoy(self.rom_path, window_type=window, sound=False)  # type: ignore[arg-type]  # pragma: no cover
        self._pyboy.set_emulation_speed(0 if not self.show_display else 1)
        self._screen = getattr(self.pyboy, "screen", None)
        if self._screen is None:
            # Headless builds do not expose pyboy.screen; fall back to the botsupport manager.
            try:
                bot_manager = self.pyboy.botsupport_manager()
                if hasattr(bot_manager, "screen"):
                    self._screen = bot_manager.screen()
            except Exception:  # pragma: no cover - best-effort fallback
                self._screen = None
        if self._screen is None:
            raise RuntimeError(
                "PyBoy did not expose a screen surface. Install pyboy with SDL2 support "
                "or ensure botsupport screen plugins are available."
            )

        if self.state_path and os.path.exists(self.state_path):
            with open(self.state_path, "rb") as fh:
                self.pyboy.load_state(fh)
        else:
            # Fast-forward the intro sequence to land on the overworld.
            for _ in range(self.boot_steps):
                self.pyboy.tick()
            self._press_and_tick(("start",), hold_frames=30)

        self._steps = 0
        self._frames_since_input = 0
        self._prev_in_battle = bool(self._read_u8(BATTLE_TYPE_ADDR))
        self._last_battle_result = None
        self._caught_flag = False
        self._last_raw_frame = None
        self._last_bag_snapshot: list[tuple[int, int]] = []
        self._reset_episode_visit_stats()

    def _press_and_tick(
        self,
        buttons: Iterable[str],
        *,
        hold_frames: int = 1,
    ) -> None:
        """Helper to press a group of buttons, tick the emulator, then release."""
        pressed = tuple(buttons)
        if pressed:
            for button in pressed:
                self._press_button(button)
        for _ in range(max(1, hold_frames)):
            self.pyboy.tick()
        if pressed:
            for button in pressed[::-1]:
                self._release_button(button)

    def _press_button(self, button: str) -> None:
        button = button.lower()
        if hasattr(self.pyboy, "button_press"):
            self.pyboy.button_press(button)
            return
        event_pair = self._button_event_pair(button)
        self.pyboy.send_input(event_pair[0])

    def _release_button(self, button: str) -> None:
        button = button.lower()
        if hasattr(self.pyboy, "button_release"):
            self.pyboy.button_release(button)
            return
        event_pair = self._button_event_pair(button)
        self.pyboy.send_input(event_pair[1])

    def _button_event_pair(self, button: str):
        if WindowEvent is None:
            raise RuntimeError(
                "PyBoy build lacks button_press/button_release and WindowEvent definitions."
            )
        mapping = self._button_event_table.get(button)
        if mapping is None:
            raise ValueError(f"Unknown button '{button}' for PyBoy input.")
        press_name, release_name = mapping
        try:
            return getattr(WindowEvent, press_name), getattr(WindowEvent, release_name)
        except AttributeError as exc:  # pragma: no cover - safety guard
            raise RuntimeError(
                f"WindowEvent does not define {press_name}/{release_name}; update pyboy."
            ) from exc

    def _build_button_event_table(self) -> Dict[str, Tuple[str, str]]:
        if WindowEvent is None:
            return {}
        return {
            "a": ("PRESS_BUTTON_A", "RELEASE_BUTTON_A"),
            "b": ("PRESS_BUTTON_B", "RELEASE_BUTTON_B"),
            "start": ("PRESS_BUTTON_START", "RELEASE_BUTTON_START"),
            "select": ("PRESS_BUTTON_SELECT", "RELEASE_BUTTON_SELECT"),
            "up": ("PRESS_ARROW_UP", "RELEASE_ARROW_UP"),
            "down": ("PRESS_ARROW_DOWN", "RELEASE_ARROW_DOWN"),
            "left": ("PRESS_ARROW_LEFT", "RELEASE_ARROW_LEFT"),
            "right": ("PRESS_ARROW_RIGHT", "RELEASE_ARROW_RIGHT"),
        }

    def _read_u8(self, address: int) -> int:
        try:
            return int(self.pyboy.memory[address])
        except Exception:
            return 0

    @staticmethod
    def _normalise_story_defs(defs: Optional[Dict[str, Dict[str, object]]]) -> Dict[str, Dict[str, object]]:
        if not defs:
            return {}
        normalised: Dict[str, Dict[str, object]] = {}
        for name, cfg in defs.items():
            if not isinstance(cfg, dict):
                continue
            addr = cfg.get("address")
            bit = cfg.get("bit", 0)
            reward = cfg.get("reward") if isinstance(cfg.get("reward"), (int, float)) else None
            try:
                if isinstance(addr, str):
                    addr_int = int(addr, 16) if addr.lower().startswith("0x") else int(addr)
                elif isinstance(addr, (int, float)):
                    addr_int = int(addr)
                else:
                    continue
                bit_int = max(0, int(bit))
            except Exception:
                continue
            normalised[name] = {
                "address": addr_int,
                "bit": bit_int,
            }
            if reward is not None:
                normalised[name]["reward"] = float(reward)
        return normalised

    def _purge_save_file(self) -> None:
        sav_path = os.path.splitext(self.rom_path)[0] + ".sav"
        if os.path.exists(sav_path):
            try:
                os.remove(sav_path)
            except OSError:
                pass

    def _reset_episode_visit_stats(self) -> None:
        self._episode_visits.clear()
        self._episode_revisit_steps = 0
        if not self.track_visit_stats:
            self._global_visit_counts.clear()

    def _record_visit_info(self, coord: Tuple[int, int]) -> Dict[str, object]:
        if not self.track_visit_stats:
            return {
                "episode_unique_tiles": 0,
                "total_unique_tiles": 0,
                "tile_visit_count": 0,
                "episode_revisit_ratio": 0.0,
            }

        was_seen = coord in self._episode_visits
        if not was_seen:
            self._episode_visits.add(coord)
        else:
            self._episode_revisit_steps += 1

        visit_total = self._global_visit_counts.get(coord, 0) + 1
        self._global_visit_counts[coord] = visit_total

        episode_unique = len(self._episode_visits)
        total_unique = len(self._global_visit_counts)
        revisit_ratio = self._episode_revisit_steps / max(1, self._steps)
        return {
            "episode_unique_tiles": episode_unique,
            "total_unique_tiles": total_unique,
            "total_locations_visited": total_unique,
            "tile_visit_count": visit_total,
            "episode_revisit_ratio": revisit_ratio,
        }

    @staticmethod
    def _normalise_town_label(map_name: str) -> Optional[str]:
        lower = map_name.lower()
        if "town" in lower or "city" in lower:
            return lower
        return None

    def _gather_info(self) -> Dict[str, object]:
        map_id = self._read_u8(MAP_ID_ADDR)
        map_name = MAP_NAMES.get(map_id)
        if map_name is None:
            map_name = f"map_{map_id:02X}"

        info: Dict[str, object] = {
            "map_id": map_id,
            "map_name": map_name,
            "agent_coords": (
                self._read_u8(PLAYER_X_ADDR),
                self._read_u8(PLAYER_Y_ADDR),
            ),
            "steps": self._steps,
        }

        badge_mask = self._read_u8(BADGE_FLAGS_ADDR)
        badges = {name: bool(badge_mask & (1 << idx)) for idx, name in enumerate(BADGE_NAMES)}
        info["badge_bits"] = badge_mask
        info["badge_count"] = sum(badges.values())
        info["badges"] = badges

        champion_byte = self._read_u8(CHAMPION_FLAG_ADDR)
        champion_flag_raw = bool((champion_byte >> CHAMPION_FLAG_BIT) & 1)
        champion_defeated = champion_flag_raw and info["badge_count"] >= 8
        info["champion_flag_raw"] = champion_flag_raw
        info["champion_defeated"] = champion_defeated

        town_label = self._normalise_town_label(map_name)
        new_town_flag = False
        if town_label:
            if town_label not in self._visited_towns:
                self._visited_towns.add(town_label)
                new_town_flag = True
        info["unique_towns_visited"] = len(self._visited_towns)
        info["new_town_visited"] = town_label if new_town_flag else None

        if self.story_flag_defs:
            story_flags = {}
            for name, cfg in self.story_flag_defs.items():
                addr = cfg["address"]
                bit = cfg["bit"]
                value = self._read_u8(addr)
                story_flags[name] = bool((value >> bit) & 1)
            info["story_flags"] = story_flags

        in_battle = self._read_u8(BATTLE_TYPE_ADDR) != 0
        info["in_battle"] = in_battle
        if in_battle and not self._prev_in_battle:
            self._last_battle_result = None

        battle_code = self._read_u8(BATTLE_RESULT_ADDR)
        info["battle_result_code"] = battle_code
        battle_label = BATTLE_RESULT_LABELS.get(battle_code)
        info["battle_result"] = battle_label
        info["encounter_species"] = self._read_u8(ENCOUNTER_SPECIES_ADDR)
        info["battle_type"] = self._read_u8(BATTLE_TYPE_DETAIL_ADDR)
        enemy_hp_info = self._read_enemy_hp()
        if in_battle:
            self._battle_result_cache_code = battle_code
            if battle_label and battle_label != "ongoing":
                self._battle_result_cache_label = battle_label
            self._battle_enemy_hp_snapshot = enemy_hp_info.copy()
        info["battle_enemy_hp_snapshot"] = self._battle_enemy_hp_snapshot.copy()
        info["battle_result_cached"] = self._battle_result_cache_label
        info["battle_debug_bytes"] = self._read_battle_debug_bytes()
        info["battle_debug_windows"] = self._read_battle_debug_windows()

        resolved_label = self._last_battle_result
        if not resolved_label and self._battle_result_cache_label:
            resolved_label = self._battle_result_cache_label
        enemy_hp_zero = enemy_hp_info.get("max", 0) > 0 and (enemy_hp_info.get("current", 1) <= 0)
        if battle_label and battle_label != "ongoing":
            resolved_label = battle_label
        elif enemy_hp_zero:
            resolved_label = "won"
        if resolved_label and resolved_label != self._last_battle_result:
            self._last_battle_result = resolved_label

        caught_now = False
        battle_transition = self._prev_in_battle and not in_battle
        if battle_transition:
            if resolved_label == "caught" or battle_label == "caught":
                caught_now = True
                self._caught_flag = True
                self._pokemon_caught_total += 1
                self._pokemon_caught_episode += 1
            cached_label = self._battle_result_cache_label
            if not resolved_label and cached_label:
                resolved_label = cached_label
            outcome = resolved_label or battle_label or cached_label or "unknown"
            defeated_flag = outcome == "won" or enemy_hp_zero
            lost_flag = outcome == "lost"
            blackout_flag = outcome == "blackout"
            flee_flag = outcome == "escaped"
            if defeated_flag and not caught_now:
                self._pokemon_defeated_total += 1
                self._pokemon_defeated_episode += 1
            if lost_flag:
                self._battle_losses_total += 1
                self._battle_losses_episode += 1
            if blackout_flag:
                self._battle_blackouts_total += 1
                self._battle_blackouts_episode += 1
            if flee_flag:
                self._battle_flees_total += 1
                self._battle_flees_episode += 1
            self._last_battle_result = outcome
            info["last_battle_outcome"] = outcome
            info["last_battle_won"] = defeated_flag and not (lost_flag or blackout_flag or flee_flag)
            info["last_battle_lost"] = lost_flag
            info["last_battle_blackout"] = blackout_flag
            info["last_battle_fled"] = flee_flag
            self._battle_result_cache_code = None
            self._battle_result_cache_label = None
            self._battle_enemy_hp_snapshot = {}
        else:
            # Reset catch flag once we leave the battle resolution screen.
            if not in_battle and self._caught_flag:
                self._caught_flag = False
        info["caught_pokemon"] = caught_now
        info["last_battle_result"] = self._last_battle_result
        info["pokemon_caught_total"] = self._pokemon_caught_total
        info["pokemon_caught_episode"] = self._pokemon_caught_episode
        info["pokemon_defeated_total"] = self._pokemon_defeated_total
        info["pokemon_defeated_episode"] = self._pokemon_defeated_episode
        info["battle_losses_total"] = self._battle_losses_total
        info["battle_losses_episode"] = self._battle_losses_episode
        info["battle_blackouts_total"] = self._battle_blackouts_total
        info["battle_blackouts_episode"] = self._battle_blackouts_episode
        info["battle_flees_total"] = self._battle_flees_total
        info["battle_flees_episode"] = self._battle_flees_episode
        info["raw_frame"] = (
            np.array(self._last_raw_frame, copy=True) if self._last_raw_frame is not None else None
        )
        bag_items = self._read_bag_items()
        info["bag_items"] = bag_items
        info["bag_item_ids"] = [item_id for item_id, _ in bag_items]
        info["key_item_ids"] = self._infer_key_items(bag_items)
        pokedex_bits = self._read_pokedex_owned_bits()
        info["pokedex_owned_bits"] = pokedex_bits
        info["pokedex_owned_count"] = self._count_bits(pokedex_bits)
        info["first_pokemon_hp"] = self._read_lead_pokemon_hp()
        info["enemy_hp"] = enemy_hp_info
        coords_int = (int(info["agent_coords"][0]), int(info["agent_coords"][1]))
        info.update(self._record_visit_info(coords_int))
        self._prev_in_battle = in_battle
        return info

    def _obs(self) -> np.ndarray:
        frame = self._capture_frame()
        self._last_raw_frame = np.array(frame, copy=True)
        return _downsample_frame(self._last_raw_frame)

    def _capture_frame(self) -> np.ndarray:
        assert self._screen is not None
        if hasattr(self._screen, "ndarray"):
            return self._screen.ndarray
        grab = getattr(self._screen, "screen_image", None)
        if callable(grab):
            image = grab()
            return np.asarray(image)
        raise RuntimeError("Unable to capture frame from PyBoy screen backend.")

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        # Always perform a hard reset since reset_game() does not exist in PyBoy
        self.close()
        self._initialize_emulator()

        self._steps = 0
        self._frames_since_input = 0
        self._prev_in_battle = False
        self._last_battle_result = None
        self._caught_flag = False
        self._last_raw_frame = None
        self._pokemon_caught_episode = 0
        self._pokemon_defeated_episode = 0
        self._battle_losses_episode = 0
        self._battle_blackouts_episode = 0
        self._battle_flees_episode = 0
        self._battle_result_cache_code = None
        self._battle_result_cache_label = None
        self._battle_enemy_hp_snapshot = {}
        obs = self._obs()
        info = self._gather_info()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action index {action}"
        buttons = self._button_groups[action]
        hold_frames = self.frame_skip
        self._press_and_tick(buttons, hold_frames=hold_frames)

        self._steps += 1
        self._frames_since_input = 0 if buttons else self._frames_since_input + hold_frames

        if self.input_spacing_frames > 0:
            self._press_and_tick(tuple(), hold_frames=self.input_spacing_frames)
            self._frames_since_input += self.input_spacing_frames

        obs = self._obs()
        info = self._gather_info()

        # End the episode only when the first Pokemon is caught,
        # as requested. Do not terminate on escape/loss.
        terminated = bool(info.get("caught_pokemon"))

        truncated = self._frames_since_input >= self.max_no_input_frames

        # Environment reward is always neutral; auxiliary modules supply shaped reward.
        reward = 0.0
        return obs, reward, terminated, truncated, info

    def render(self):
        # PyBoy handles window updates internally when show_display=True.
        return None

    def close(self):
        if self._pyboy is not None:
            self._pyboy.stop()
            self._pyboy = None
        self._screen = None
        self._last_bag_snapshot = []

    def save_state(self, path: str) -> bool:
        """Persist the current emulator state to disk."""
        if not self._pyboy:
            return False
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except (TypeError, OSError):
            pass
        try:
            with open(path, "wb") as fh:
                self._pyboy.save_state(fh)
            return True
        except Exception as exc:
            print(f"[env] Warning: failed to save savestate to {path}: {exc}")
            return False

    def _read_bag_items(self) -> list[tuple[int, int]]:
        items: list[tuple[int, int]] = []
        try:
            for slot in range(MAX_BAG_SLOTS):
                item_addr = BAG_ITEMS_ADDR + slot * 2
                item_id = self._read_u8(item_addr)
                if item_id == 0xFF:
                    break
                quantity = self._read_u8(item_addr + 1)
                if item_id == 0:
                    continue
                items.append((item_id, quantity))
        except Exception:
            pass
        return items

    @staticmethod
    def _infer_key_items(bag_items: list[tuple[int, int]]) -> list[int]:
        key_items: list[int] = []
        for item_id, quantity in bag_items:
            if quantity == 1 and item_id not in key_items:
                key_items.append(item_id)
        return key_items

    def _read_pokedex_owned_bits(self) -> list[int]:
        data: list[int] = []
        try:
            for offset in range(POKEDEX_OWNED_BYTES):
                data.append(self._read_u8(POKEDEX_OWNED_ADDR + offset))
        except Exception:
            pass
        return data

    @staticmethod
    def _count_bits(bytes_list: list[int]) -> int:
        return sum(bin(byte & 0xFF).count("1") for byte in bytes_list)

    def _read_lead_pokemon_hp(self) -> Dict[str, int]:
        try:
            current = self._read_u8(LEAD_HP_CURRENT_ADDR) | (self._read_u8(LEAD_HP_CURRENT_ADDR + 1) << 8)
            maximum = self._read_u8(LEAD_HP_MAX_ADDR) | (self._read_u8(LEAD_HP_MAX_ADDR + 1) << 8)
            status = self._read_u8(LEAD_STATUS_ADDR)
            return {"current": current, "max": maximum, "status": status}
        except Exception:
            return {"current": 0, "max": 0, "status": 0}
        self._last_raw_frame = None

    def _read_enemy_hp(self) -> Dict[str, int]:
        try:
            current = self._read_u8(ENEMY_HP_CURRENT_ADDR) | (self._read_u8(ENEMY_HP_CURRENT_ADDR + 1) << 8)
            maximum = self._read_u8(ENEMY_HP_MAX_ADDR) | (self._read_u8(ENEMY_HP_MAX_ADDR + 1) << 8)
            return {"current": current, "max": maximum}
        except Exception:
            return {"current": 0, "max": 0}

    def _read_battle_debug_bytes(self) -> Dict[str, int]:
        debug: Dict[str, int] = {}
        for name, addr in _BATTLE_DEBUG_ADDRS.items():
            try:
                debug[name] = self._read_u8(addr)
            except Exception:
                debug[name] = -1
        return debug

    def _read_battle_debug_windows(self) -> Dict[str, list[int]]:
        windows: Dict[str, list[int]] = {}
        for name, start, length in _BATTLE_DEBUG_WINDOWS:
            chunk: list[int] = []
            for offset in range(length):
                try:
                    chunk.append(self._read_u8(start + offset))
                except Exception:
                    chunk.append(-1)
            windows[name] = chunk
        return windows
