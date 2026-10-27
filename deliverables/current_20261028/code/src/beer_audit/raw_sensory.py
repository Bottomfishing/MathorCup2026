"""Audit external raw panel scores and link anonymized beers by numeric identity."""
import hashlib

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


ALIASES = {'A_ester_fruity': 'A_esters_fruity', 'F_ester_fruity': 'F_esters_fruity'}
EVENT_KEY = ['date', 'Beer', 'taster']


def check_file(path, manifest):
    content = path.read_bytes()
    method, expected = manifest['published_checksum'].split(':', 1)
    if (len(content) != manifest['bytes'] or
            hashlib.new(method, content).hexdigest() != expected or
            hashlib.sha256(content).hexdigest() != manifest['sha256']):
        raise ValueError('Raw sensory source does not match the pinned release')


def load_raw(path, targets):
    raw = pd.read_csv(path, encoding='cp1252').rename(columns=ALIASES)
    if raw.columns.duplicated().any() or not set(targets) <= set(raw):
        raise ValueError('Missing or duplicated sensory targets')
    validate_scores(raw[targets])
    if raw[EVENT_KEY + ['beer_name']].isna().any().any():
        raise ValueError('Missing panel event identifier')
    if raw.duplicated(EVENT_KEY).any():
        raise ValueError('Duplicate session/position/taster event')
    return raw


def validate_scores(scores):
    values = scores.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any() or (values > 6).any()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError('Expected observed integer scores from 0 to 6')


def normalize_panel(raw, targets, ddof=1):
    grouped = raw.groupby('taster')[targets]
    means, standard_deviations = grouped.mean(), grouped.std(ddof=ddof)
    zero = standard_deviations.eq(0)
    # Undefined standardized values remain missing; raw zero scores remain intact.
    row_means = means.loc[raw.taster].set_axis(raw.index)
    row_sds = standard_deviations.loc[raw.taster].set_axis(raw.index)
    normalized = (raw[targets] - row_means) / row_sds.replace(0, np.nan)
    groups = []
    for taster in zero.index:
        for target in zero.columns[zero.loc[taster]]:
            values = raw.loc[raw.taster.eq(taster), target]
            groups.append(dict(taster=int(taster), target=target, n_ratings=len(values),
                               constant_raw_score=int(values.iloc[0]), sample_sd=0.0))
    valid_targets = [target for target in targets if standard_deviations[target].gt(0).all()]
    return normalized, valid_targets, pd.DataFrame(groups)


def link_beers(normalized_means, panel, valid_targets, tolerance):
    """Require a unique match in both directions; row positions are never a join key."""
    if not valid_targets:
        raise ValueError('No nonconstant targets available for identity verification')
    left = normalized_means[valid_targets].to_numpy(float)
    right = panel[valid_targets].to_numpy(float)
    if len(left) != len(right) or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Incomplete beer fingerprint matrices')
    maximum_errors = cdist(left, right, metric='chebyshev')
    matches = maximum_errors <= tolerance
    if not (matches.sum(axis=1) == 1).all() or not (matches.sum(axis=0) == 1).all():
        raise ValueError('Beer identity is unmatched or ambiguous; do not join by row or ID')
    positions = matches.argmax(axis=1)
    distances = cdist(left, right) / np.sqrt(len(valid_targets))
    records = []
    for i, j in enumerate(positions):
        other = np.delete(distances[i], j)
        records.append(dict(raw_beer_name=int(normalized_means.index[i]),
            beer=int(panel.beer.iloc[j]), panel_row_1based=int(j + 1),
            verified_dimensions=len(valid_targets),
            max_abs_difference=float(maximum_errors[i, j]),
            rmse=float(distances[i, j]), second_best_rmse=float(other.min()) if len(other) else np.nan,
            matches_within_tolerance=int(matches[i].sum()),
            mapping_method='unique_standardized_score_fingerprint',
            official_crosswalk_supplied=False))
    return pd.DataFrame(records)


def group_fold_assignments(raw, mapping, beer_folds):
    mapped = raw[['beer_name']].rename(columns={'beer_name': 'raw_beer_name'}).copy()
    mapped.insert(0, 'raw_record_1based', np.arange(1, len(raw) + 1))
    mapped = mapped.merge(mapping[['raw_beer_name', 'beer']], on='raw_beer_name',
                          how='left', validate='many_to_one', sort=False)
    mapped = mapped.merge(beer_folds, on='beer', how='left', validate='many_to_one', sort=False)
    if len(mapped) != len(raw) or mapped.isna().any().any():
        raise ValueError('Incomplete beer-level fold assignment')
    if not mapped.groupby('beer').outer_fold.nunique().eq(1).all():
        raise ValueError('Ratings of the same beer cross validation folds')
    return mapped.sort_values('raw_record_1based').reset_index(drop=True)
