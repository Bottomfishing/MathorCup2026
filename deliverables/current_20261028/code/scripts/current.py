"""Current delivery: fixed training, daily inference, export and verification.

``--version current`` uses the 20261028 result set (Q1 decision layer, Q2 full
Top-3, Q3 copula generation and 33/33 migration). ``--version 20261027`` still
exports and verifies the frozen historical package.
"""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from threadpoolctl import threadpool_limits
from beer_models.current_runtime import MODEL_DIR
from beer_models.panel_data import RAW_MANIFEST

VERSIONS = ['current', '20261027']


def delivery_module(version):
    if version == 'current':
        from beer_models import delivery_current
        return delivery_current
    from beer_models import delivery
    return delivery


def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('status', help='Show the adopted methods and unresolved issues; no models needed')
    p.add_argument('--version', choices=VERSIONS, default='current')
    p = sub.add_parser('train', help='Rebuild only the adopted full-data deployment models')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    p.add_argument('--output', type=Path, default=MODEL_DIR)
    p = sub.add_parser('predict', help='Official Q1 or separately labelled Q3 auxiliary inference')
    p.add_argument('--task', choices=['q1', 'q3'], default='q1')
    p.add_argument('--models', type=Path, default=MODEL_DIR)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('render', help='Refresh 300 adopted generated spectra and the beer119 sensory tables')
    p.add_argument('--models', type=Path, default=MODEL_DIR)
    p.add_argument('--generated', type=Path, help='Generated-profile directory; defaults to the verified current set')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('package', help='Export the current verified tables, figures and runnable source')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rendered', type=Path, help='Use a numerically verified render from rebuilt models')
    p.add_argument('--version', choices=VERSIONS, default='current')
    p = sub.add_parser('verify', help='Check current scientific tables; optionally replay rebuilt models')
    p.add_argument('--models', type=Path)
    p.add_argument('--source', type=Path)
    p.add_argument('--package', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--version', choices=VERSIONS, default='current')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command == 'train':
            from beer_models.current_runtime import train
            train(args.source, args.raw, args.output)
        elif args.command == 'predict':
            if args.task == 'q1':
                from beer_models.inference import run
                run(args.models/'q1.joblib', args.input, args.output)
            else:
                import joblib
                import pandas as pd
                from beer_audit.audit import save_csv
                from beer_models.current_runtime import auxiliary_defects
                official = joblib.load(args.models/'q1.joblib')
                q3 = joblib.load(args.models/'q3.joblib')
                risks, styles = auxiliary_defects(pd.read_csv(args.input), q3, official['targets'])
                args.output.mkdir(parents=True, exist_ok=False)
                save_csv(args.output, 'auxiliary_defect_risks.csv', risks)
                save_csv(args.output, 'style_predictions.csv', styles)
        elif args.command == 'render':
            from beer_models.current_runtime import render
            render(args.models, args.output, args.generated)
        else:
            module = delivery_module(args.version)
            if args.command == 'status':
                module.status()
            elif args.command == 'package':
                module.build(args.output, args.rendered)
            else:
                module.verify(args.output, args.models, args.source, args.package)


if __name__ == '__main__':
    main()
