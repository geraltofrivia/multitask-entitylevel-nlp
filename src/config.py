import numpy as np
from pathlib import Path
from typing import Dict
from mytorch.utils.goodies import FancyDict
from utils.exceptions import UnknownDataSplitException

# No local imports (is a good idea)

# Random seeds
_SEED_ = 42
NPRSEED: int = _SEED_
PYTSEED: int = _SEED_

ROOT_LOC: Path = Path("..") if str(Path().cwd()).split("/")[-1] == "src" else Path(".")
LOCATIONS: Dict[str, Path] = FancyDict(
    **{
        "root": ROOT_LOC,
        "ontonotes_raw": ROOT_LOC / "data" / "raw" / "ontonotes" / "ontonotes-release-5.0",
        "ontonotes_conll": ROOT_LOC / "data" / "raw" / "ontonotes" / "conll-2012" / "v5" / "data",
        "scierc": ROOT_LOC / "data" / "raw" / "scierc",
        "raw": ROOT_LOC / "data" / "raw",
        "runs": ROOT_LOC / "data" / "runs",
        "parsed": ROOT_LOC / "data" / "parsed",
        "word2vec": ROOT_LOC / "models" / "word2vec",
        "glove": ROOT_LOC / "models" / "glove",
        "manual": ROOT_LOC / "data" / "manual",
        "models": ROOT_LOC / "models" / "trained"
    }
)

KNOWN_TASKS = ["coref", "ner", "ner_spacy", "pruner", "rel"]
KNOWN_SPLITS = FancyDict(**{
    'scierc': FancyDict(**{
        'train': 'train',
        'dev': 'dev',
        'test': 'test'
    }),
    'ontonotes': FancyDict(**{
        'train': 'train',
        'dev': 'development',
        'test': 'test'
    })
})


def unalias_split(split: str) -> str:
    for ds, ds_splits in KNOWN_SPLITS.items():
        for split_alias, split_vl in ds_splits.items():
            if split_vl == split:
                return split_alias

    raise UnknownDataSplitException(f"The split: {split} is unknown.")


# LOSS_RATIO_CNP = [1.0 / 20000, 1.0 / 2.5, 1.0]  # Loss ratio to use to train coref, ner and pruner
# changed CNP ratio now that coref is normalised to 1
LOSS_RATIO_CNP = [1.0, 1.0 / 2.5, 1.0]  # Loss ratio to use to train coref, ner and pruner
LOSS_RATIO_CP = [0.1, 1.0]  # Loss ratio to use to train coref, and pruner
LOSS_RATIO_CN = [1.0 / 20000, 1.0 / 2.5]  # Loss ratio to use to train coref, and pruner
CONFIG: dict = {
    'filter_candidates_pos_threshold': 2000,
    'max_span_width': 5,  # we need to push this to 30 somehow :shrug:
    'learning_rate': 0.005,
    'coref_metadata_feature_size': 20,  # self explanatory
    'coref_max_training_segments': 5  # used to determine max in segment distance part of coref
}
LOSS_SCALES = {
    'loss_scales_coref_ner_pruner': np.exp(LOSS_RATIO_CNP) / np.sum(np.exp(LOSS_RATIO_CNP)),
    'loss_scales_coref_pruner': np.exp(LOSS_RATIO_CP) / np.sum(np.exp(LOSS_RATIO_CP)),
    'loss_scales_coref_ner': np.exp(LOSS_RATIO_CN) / np.sum(np.exp(LOSS_RATIO_CN)),
    'loss_scales_coref': [1.0, ],
    'loss_scales_ner': [1.0, ],
    'loss_scales_pruner': [1.0, ],
}
