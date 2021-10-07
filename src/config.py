import os
from pathlib import Path
from mytorch.utils.goodies import FancyDict

ROOT_LOC: Path = Path('..') if str(Path().cwd()).split('/')[-1] == 'src' else Path('.')
LOCATIONS: dict = FancyDict(**{
    'root': ROOT_LOC,
    'ontonotes_raw': ROOT_LOC / 'data' / 'raw' / 'ontonotes' / 'ontonotes-release-5.0',
    'ontonotes_conll': ROOT_LOC / 'data' / 'raw' / 'ontonotes' / 'conll-2012' / 'v5' / 'data',
    'raw': ROOT_LOC / 'data' / 'raw',
    'parsed': ROOT_LOC / 'data' / 'parsed'
})
