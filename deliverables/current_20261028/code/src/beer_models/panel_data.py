"""Pinned raw-panel data alignment shared by training and historical experiments."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from beer_audit.raw_sensory import check_file, load_raw, normalize_panel, link_beers
from .baseline import load_data
from .raw_panel import rating_weights, rating_distribution
ROOT = Path(__file__).resolve().parents[2]
RAW_MANIFEST = ROOT/'data/reference/zenodo23624915/download_manifest.json'


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode('utf-8')).hexdigest()


def load_inputs(source, raw_path, protocol):
    ids, xf, yf, primary_names = load_data(source, ROOT / 'outputs/p0_audit_20261023_final')
    ids = ids.copy()
    ids['beer'] = ids.beer.astype(int)
    manifest = json.loads(RAW_MANIFEST.read_text(encoding='utf-8'))
    check_file(raw_path, manifest)
    raw = load_raw(raw_path, list(yf))
    z, valid, _ = normalize_panel(raw, list(yf))
    panel = pd.concat([ids[['beer']].reset_index(drop=True), yf.reset_index(drop=True)], axis=1)
    mapping = link_beers(z.groupby(raw.beer_name).mean(), panel, valid, 1e-7)
    prior = pd.read_csv(ROOT / 'outputs/raw_sensory_audit_20261025/beer_identity_mapping.csv')
    pd.testing.assert_frame_equal(mapping[['raw_beer_name', 'beer']], prior[['raw_beer_name', 'beer']])
    raw['beer'] = raw.beer_name.map(mapping.set_index('raw_beer_name').beer)
    positions = raw.beer.map({beer: i for i, beer in enumerate(ids.beer)}).to_numpy(int)
    weights = rating_weights(raw)
    scores = raw[list(yf)].to_numpy(int)
    distribution = rating_distribution(positions, scores, weights, len(ids))
    features = [f for f in xf if f not in protocol['excluded_features']]
    scope = pd.read_csv(ROOT / 'outputs/task1_acceptance_20261025/scorecard_scope.csv').fillna('')
    parents = []
    for row in scope.loc[scope.template_rule.eq('checkbox_zero_or_parent')].itertuples():
        parent = '_'.join(row.target_id.split('_')[:2]) + '_all'
        parents.append((list(yf).index(parent), list(yf).index(row.target_id)))
    return dict(ids=ids, x=xf[features].to_numpy(float), features=features, targets=list(yf),
        primary=np.array([t in primary_names for t in yf]), parents=parents, scope=scope,
        raw=raw, positions=positions, scores=scores, weights=weights, distribution=distribution,
        manifest=manifest)
