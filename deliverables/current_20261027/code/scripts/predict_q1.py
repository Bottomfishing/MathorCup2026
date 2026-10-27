"""Compatibility Q1 CLI; the current workflow is scripts/current.py."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from threadpoolctl import threadpool_limits
from beer_models.inference import run
from beer_models.current_runtime import MODEL_DIR


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=MODEL_DIR/'q1.joblib')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--experimental-tail', action='store_true', help='Historical unadopted candidate; requires an explicit experimental bundle')
    args = p.parse_args()
    with threadpool_limits(limits=1):
        run(args.model, args.input, args.output, args.experimental_tail)
