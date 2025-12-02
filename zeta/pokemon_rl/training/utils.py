from __future__ import annotations

import os


def purge_rom_save(rom_path: str) -> None:
    """Remove the .sav file associated with the ROM to guarantee clean flags."""
    base, _ = os.path.splitext(os.path.expanduser(rom_path))
    sav_path = f"{base}.sav"
    try:
        os.remove(sav_path)
        print(f"[progress] Removed stale save file {sav_path}")
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"[progress] Warning: unable to delete {sav_path}: {exc}")
