import os
from pathlib import Path
from mytorch.utils.goodies import FancyDict
from typing import Dict

# Random seeds
_SEED_ = 42
NPRSEED: int = _SEED_
PYTSEED: int = _SEED_

ROOT_LOC: Path = Path("..") if str(Path().cwd()).split("/")[-1] == "src" else Path(".")
LOCATIONS: Dict[str, Path] = FancyDict(
    **{
        "root": ROOT_LOC,
        "ontonotes_raw": ROOT_LOC
                         / "data"
                         / "raw"
                         / "ontonotes"
                         / "ontonotes-release-5.0",
        "ontonotes_conll": ROOT_LOC
                           / "data"
                           / "raw"
                           / "ontonotes"
                           / "conll-2012"
                           / "v5"
                           / "data",
        "raw": ROOT_LOC / "data" / "raw",
        "runs": ROOT_LOC / "data" / "runs",
        "parsed": ROOT_LOC / "data" / "parsed",
        "word2vec": ROOT_LOC / "models" / "word2vec",
        "glove": ROOT_LOC / "models" / "glove",
        "manual": ROOT_LOC / "data" / "manual",
    }
)
