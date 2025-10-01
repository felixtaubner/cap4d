import pickle
from pathlib import Path
import argparse
import numpy as np
import inspect
# define getargspec as getfullargspec
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

# Monkey-patch missing aliases (no-ops in Python 3)
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'str'):
    np.str = str
if not hasattr(np, 'unicode'):
    np.unicode = str  # Python 3 uses str for unicode
import chumpy as ch


def ch_to_numpy(obj):
    if isinstance(obj, ch.Ch):
        return obj.r  # Extract the numpy array
    elif isinstance(obj, dict):
        return {k: ch_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ch_to_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(ch_to_numpy(v) for v in obj)
    else:
        return obj


def main(args):
    pickle_path = Path(args.pickle_path)

    # Monkey-patch: make loading the flame pkl compatible with new Python and numpy versions
    # this is necessary because the FLAME model is saved in a deprecated numpy and chumpy versions
    np.bool = bool  
    np.int = int
    np.float = float
    np.complex = complex
    np.object = object
    np.unicode = str  # `str` now includes Unicode in Python 3
    np.str = str
    np.nan = np.nan
    np.inf = np.inf

    pickle_file = pickle.load(
        open(pickle_path, "rb"), encoding="latin1"
    )

    pickle_file = ch_to_numpy(pickle_file)

    # import pdb; pdb.set_trace()

    with open(pickle_path, "wb") as f:
        pickle.dump(pickle_file, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pickle_path",
        type=str,
        required=True,
        help="path to FLAME pickle file to fix",
    )
    args = parser.parse_args()
    main(args)
