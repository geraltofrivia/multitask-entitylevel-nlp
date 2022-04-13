import os
import numpy as np
from pathlib import Path
from typing import Dict
from mytorch.utils.goodies import FancyDict

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
        "models": ROOT_LOC / "models" / "trained"
    }
)

# LOSS_RATIO_CNP = [1.0 / 20000, 1.0 / 2.5, 1.0]  # Loss ratio to use to train coref, ner and pruner
# changed CNP ratio now that coref is normalised to 1
LOSS_RATIO_CNP = [1.0, 1.0 / 2.5, 1.0]  # Loss ratio to use to train coref, ner and pruner
LOSS_RATIO_CP = [1.0 / 20000, 1.0]  # Loss ratio to use to train coref, and pruner
LOSS_RATIO_CN = [1.0 / 20000, 1.0 / 2.5]  # Loss ratio to use to train coref, and pruner
CONFIG: dict = {
    'loss_scales_coref_ner_pruner': np.exp(LOSS_RATIO_CNP) / np.sum(np.exp(LOSS_RATIO_CNP)),
    'loss_scales_coref_pruner': np.exp(LOSS_RATIO_CP) / np.sum(np.exp(LOSS_RATIO_CP)),
    'loss_scales_coref_ner': np.exp(LOSS_RATIO_CN) / np.sum(np.exp(LOSS_RATIO_CN)),
    'loss_scales_coref': [1.0, ],
    'loss_scales_ner': [1.0, ],
    'filter_candidates_pos_threshold': 2000
}
