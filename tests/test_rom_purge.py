import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epsilon.pokemon_rl.utils import purge_rom_save


def test_purge_rom_save_removes_file(tmp_path: Path) -> None:
    rom = tmp_path / "pokemon_red.gb"
    sav = tmp_path / "pokemon_red.sav"
    rom.write_bytes(b"rom")
    sav.write_text("old save")

    purge_rom_save(str(rom))
    assert not sav.exists()


def test_purge_rom_save_missing_file_does_not_raise(tmp_path: Path) -> None:
    rom = tmp_path / "pokemon_red.gb"
    rom.write_bytes(b"rom")
    purge_rom_save(str(rom))
    assert not (tmp_path / "pokemon_red.sav").exists()
