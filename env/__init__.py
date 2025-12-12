import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent / "pokemonred_puffer"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
