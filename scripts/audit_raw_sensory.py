"""Audit external 0-6 panel records, identity, repeats and label definitions; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from beer_audit.audit import sha256
from beer_audit.raw_sensory import (ALIASES, check_file, load_raw, normalize_panel,
                                   link_beers, group_fold_assignments)
from beer_models.baseline import make_splits

CONFIG = ROOT / 'configs/raw_sensory_audit_20261025.json'
MANIFEST = ROOT / 'data/reference/zenodo23624915/download_manifest.json'


def content_hash(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode('utf-8')).hexdigest()


def assemble(source, raw_path):
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    check_file(raw_path, manifest)
    inventory = pd.read_csv(ROOT / 'outputs/p0_audit_20261023_final/data_inventory.csv').set_index('file')
    source_hashes = {name: sha256(source / name) for name in ['1.csv', '3.xlsx', '4.csv']}
    for name, value in source_hashes.items():
        if value != inventory.loc[name, 'sha256']:
            raise ValueError('Contest source changed: ' + name)
    x = pd.read_csv(source / '1.csv')
    panel = pd.read_csv(source / '4.csv')
    if panel.beer.duplicated().any() or x.beer.duplicated().any() or set(panel.beer) != set(x.beer):
        raise ValueError('Contest beer IDs are not a one-to-one correspondence')
    panel = panel.set_index('beer').loc[x.beer].reset_index()
    if not np.array_equal(panel.tasting_category_fine, x.tasting_category_fine):
        raise ValueError('Contest feature/label styles disagree')
    targets = list(panel.columns[3:])
    raw = load_raw(raw_path, targets)
    normalized, valid_targets, constant_groups = normalize_panel(raw, targets)
    reconstructed = normalized.groupby(raw.beer_name).mean()
    mapping = link_beers(reconstructed, panel, valid_targets, config['matching_atol'])
    mapping['row_order_agrees'] = mapping.raw_beer_name.eq(mapping.panel_row_1based)
    mapping['raw_id_equals_contest_id'] = mapping.raw_beer_name.eq(mapping.beer)
    mapped_panel = panel.set_index('beer').loc[mapping.beer, targets].set_axis(mapping.raw_beer_name)
    mapped_panel = mapped_panel.loc[reconstructed.index]
    difference = reconstructed - mapped_panel
    diag = []
    for target in targets:
        errors = difference[target].to_numpy()
        diag.append(dict(target=target, identity_verification_dimension=target in valid_targets,
            constant_taster_groups=int(constant_groups.target.eq(target).sum()),
            undefined_standardized_ratings=int(normalized[target].isna().sum()),
            reconstructed_beers=int(np.isfinite(errors).sum()),
            max_abs_difference=float(np.nanmax(abs(errors))),
            rmse=float(np.sqrt(np.nanmean(errors ** 2))),
            mismatched_beers=int((abs(errors) > config['matching_atol']).sum()),
            zero_variance_policy='omit_undefined_for_diagnostic_only'))
    methods = []
    for name, ddof, fill_zero, equal_taster in config['normalization_diagnostics']:
        z, _, _ = normalize_panel(raw, targets, ddof=ddof)
        if fill_zero:
            z = z.fillna(0)
        if equal_taster:
            z = z.assign(beer_name=raw.beer_name, taster=raw.taster)
            z = z.groupby(['beer_name', 'taster'], as_index=False)[targets].mean()
            means = z.groupby('beer_name')[targets].mean()
        else:
            means = z.groupby(raw.beer_name).mean()
        err = means - mapped_panel
        methods.append(dict(method=name, ddof=ddof, zero_fill=fill_zero, equal_taster=equal_taster,
            matched_cells=int((err.abs() <= config['matching_atol']).sum().sum()),
            all_50_matched_beers=int(err.abs().le(config['matching_atol']).all(axis=1).sum()),
            matched_nonconstant_cells=int((err[valid_targets].abs() <= config['matching_atol']).sum().sum()),
            nonconstant_cells=len(reconstructed) * len(valid_targets),
            max_abs_difference=float(err.abs().max().max()),
            scope='provenance_reconstruction_not_a_model_or_independent_validation'))

    group = raw.groupby('beer_name', sort=True)
    counts = group.agg(n_ratings=('taster', 'size'), n_tasters=('taster', 'nunique'),
                       n_sessions=('date', 'nunique'))
    presentations = raw[['date', 'beer_name', 'Beer']].drop_duplicates()
    counts['n_presentations'] = presentations.groupby('beer_name').size()
    counts = counts.reset_index().rename(columns={'beer_name': 'raw_beer_name'})
    counts = counts.merge(mapping[['raw_beer_name', 'beer']], on='raw_beer_name', validate='one_to_one')
    counts = counts.merge(x[['beer', 'tasting_category_fine']], on='beer', validate='one_to_one')
    repeated = presentations.groupby(['date', 'beer_name']).Beer.agg(
        n_presentations='size', positions=lambda s: ';'.join(map(str, sorted(s))))
    repeated = repeated.loc[repeated.n_presentations > 1].reset_index()
    repeated = repeated.rename(columns={'beer_name': 'raw_beer_name'}).merge(
        mapping[['raw_beer_name', 'beer']], on='raw_beer_name', validate='many_to_one')

    raw_mean = group[targets].mean()
    raw_median = group[targets].median()
    balanced_mean = raw.groupby(['beer_name', 'taster'])[targets].mean().groupby('beer_name').mean()
    label_rows = []
    count_index = counts.set_index('raw_beer_name')
    for raw_id in raw_mean.index:
        rr = raw.loc[raw.beer_name.eq(raw_id)]
        meta = count_index.loc[raw_id]
        for target in targets:
            label_rows.append(dict(raw_beer_name=int(raw_id), beer=int(meta.beer), target=target,
                n_ratings=int(meta.n_ratings), n_tasters=int(meta.n_tasters), n_sessions=int(meta.n_sessions),
                raw_record_mean=float(raw_mean.loc[raw_id, target]),
                raw_record_median=float(raw_median.loc[raw_id, target]),
                raw_equal_taster_mean=float(balanced_mean.loc[raw_id, target]),
                **{f'raw_record_count_grade_{k}': int(rr[target].eq(k).sum()) for k in range(7)},
                scope='diagnostic_label_alternatives_not_selected_training_target'))
    label_alternatives = pd.DataFrame(label_rows)
    target_rows = []
    for target in targets:
        per_beer = group[target]
        target_rows.append(dict(target=target, n_ratings=len(raw), n_beers=raw.beer_name.nunique(),
            minimum=int(raw[target].min()), maximum=int(raw[target].max()),
            zero_fraction=float(raw[target].eq(0).mean()),
            all_zero_beers=int(per_beer.max().eq(0).sum()),
            mean_within_beer_sample_sd=float(per_beer.std().mean()),
            **{f'count_grade_{k}': int(raw[target].eq(k).sum()) for k in range(7)}))

    scope_path = ROOT / 'outputs/task1_acceptance_20261025/scorecard_scope.csv'
    scope = pd.read_csv(scope_path, keep_default_na=False)
    if len(scope) != 52 or scope.target_id.duplicated().any() or not set(targets) <= set(scope.target_id):
        raise ValueError('The audited 52-item scorecard scope no longer covers all raw targets')
    coverage = scope[['target_id', 'template_section', 'acceptance_scope', 'template_rule']].copy()
    coverage['raw_source_field'] = coverage.target_id.map(
        {target: next((old for old, new in ALIASES.items() if new == target), target) for target in targets}).fillna('')
    coverage['raw_ratings_available'] = coverage.target_id.isin(targets)
    coverage['n_observed_ratings'] = coverage.raw_ratings_available.astype(int) * len(raw)
    coverage['evidence_status'] = np.where(coverage.raw_ratings_available,
        'observed_original_integer_0_6', 'no_raw_panel_label_do_not_fill')

    hierarchy = []
    for row in scope.loc[scope.template_rule.eq('checkbox_zero_or_parent')].itertuples():
        target = row.target_id
        parent = '_'.join(target.split('_')[:2]) + '_all'
        bad = raw[target].ne(0) & raw[target].ne(raw[parent])
        hierarchy.append(dict(target=target, parent=parent, checked_ratings=len(raw),
            violations=int(bad.sum()), rule='individual_score_is_zero_or_parent'))

    folds = pd.DataFrame({'beer': x.beer, 'outer_fold': -1})
    splits = make_splits(x.tasting_category_fine, config['outer_folds'], config['fold_seed'])
    for fold, (_, test) in enumerate(splits):
        folds.loc[test, 'outer_fold'] = fold
    rating_folds = group_fold_assignments(raw, mapping, folds)
    # Cross-check the split by set intersection, independently of the group count guard.
    for fold in range(config['outer_folds']):
        train = rating_folds.loc[rating_folds.outer_fold.ne(fold), 'beer']
        test = rating_folds.loc[rating_folds.outer_fold.eq(fold), 'beer']
        assert not set(train) & set(test)

    tables = {'beer_identity_mapping.csv': mapping, 'target_coverage52.csv': coverage,
        'zero_variance_groups.csv': constant_groups, 'normalization_by_target.csv': pd.DataFrame(diag),
        'normalization_methods.csv': pd.DataFrame(methods), 'beer_record_counts.csv': counts,
        'repeat_presentations.csv': repeated, 'candidate_label_alternatives.csv': label_alternatives,
        'target_score_distributions.csv': pd.DataFrame(target_rows),
        'individual_hierarchy_checks.csv': pd.DataFrame(hierarchy),
        'beer_group_folds.csv': folds, 'rating_group_folds.csv': rating_folds}
    balance_diff = (raw_mean - balanced_mean).abs()
    summary = dict(status='audit_completed', baseline_commit=config['baseline_commit'],
        config_content_sha256=content_hash(config), source_sha256=source_hashes,
        raw_source_sha256=manifest['sha256'], provenance_manifest_content_sha256=content_hash(manifest),
        scorecard_scope_content_sha256=content_hash(scope.to_dict('records')),
        raw_release_doi=manifest['doi'], raw_release_date=manifest['publication_date'],
        records=len(raw), beers=int(raw.beer_name.nunique()), tasters=int(raw.taster.nunique()),
        sessions=int(raw.date.nunique()), raw_session_style_labels=int(raw.beer_category.nunique()),
        contest_fine_styles=int(x.tasting_category_fine.nunique()),
        observed_targets=len(targets), observed_rating_cells=len(raw) * len(targets),
        missing_card_targets=coverage.loc[~coverage.raw_ratings_available, 'target_id'].tolist(),
        original_integer_score_range=[int(raw[targets].min().min()), int(raw[targets].max().max())],
        identity_dimensions=len(valid_targets), identity_linked_beers=len(mapping),
        identity_max_abs_difference=float(mapping.max_abs_difference.max()),
        identity_min_second_best_rmse=float(mapping.second_best_rmse.min()),
        mapping_is_official_crosswalk=False,
        all_row_positions_agree=bool(mapping.row_order_agrees.all()),
        raw_id_direct_join_matches=int(mapping.raw_id_equals_contest_id.sum()),
        normalization_zero_variance_groups=len(constant_groups),
        normalization_affected_targets=len(targets) - len(valid_targets),
        normalization_all_50_recipe_reproduced=False,
        full_record_duplicates=int(raw.duplicated().sum()),
        duplicate_session_beer_taster_excess_rows=int(raw.duplicated(['date', 'beer_name', 'taster']).sum()),
        duplicate_session_position_taster_events=int(raw.duplicated(['date', 'Beer', 'taster']).sum()),
        beers_in_multiple_sessions=int(counts.n_sessions.gt(1).sum()),
        repeated_within_session_beer_groups=len(repeated),
        beers_with_repeat_presentations=int(counts.n_presentations.gt(1).sum()),
        raw_ratings_per_beer_range=[int(counts.n_ratings.min()), int(counts.n_ratings.max())],
        tasters_per_beer_range=[int(counts.n_tasters.min()), int(counts.n_tasters.max())],
        aggregation_weighting_changes_beers=int(balance_diff.gt(1e-12).any(axis=1).sum()),
        aggregation_weighting_max_score_difference=float(balance_diff.max().max()),
        hierarchy_pairs=len(hierarchy), hierarchy_checked_cells=sum(h['checked_ratings'] for h in hierarchy),
        hierarchy_violations=sum(h['violations'] for h in hierarchy),
        grouped_outer_folds=config['outer_folds'], grouped_fold_seed=config['fold_seed'],
        split_scope='registered_grouping_for_later_experiment_no_prediction_performance',
        decision='raw_0_6_records_usable_for_50_targets_with_verified_numeric_identity',
        next_task='define_panel_target_and_run_limited_Q1_comparison_only_after_user_instruction',
        label_aggregation_selected=False, models_trained=0, official_model_replaced=False)
    return tables, summary


def run(args):
    tables, summary = assemble(args.source, args.raw)
    if not args.verify_only:
        source = args.source.resolve()
        output = args.output.resolve()
        if output == source or source in output.parents:
            raise ValueError('Output must not be inside original contest source')
        args.output.mkdir(parents=True, exist_ok=False)
        for name, frame in tables.items():
            frame.to_csv(args.output / name, index=False, encoding='utf-8-sig')
        (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False,
            indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, frame in tables.items():
        saved = pd.read_csv(args.output / name, dtype=str, keep_default_na=False)
        expected = frame.fillna('').astype(str)
        pd.testing.assert_frame_equal(saved, expected, obj=name)
    assert json.loads((args.output / 'summary.json').read_text(encoding='utf-8')) == summary
    verification = dict(status='passed', tables_checked=len(tables), identity_beers=summary['beers'],
        identity_dimensions=summary['identity_dimensions'],
        original_score_cells_checked=summary['observed_rating_cells'],
        candidate_label_rows=len(tables['candidate_label_alternatives.csv']),
        grouped_rating_records=len(tables['rating_group_folds.csv']),
        checks=['pinned_raw_MD5_SHA256_size', 'frozen_contest_source_hashes',
            'integer_0_6_without_missing_scores', 'unique_bidirectional_numeric_identity',
            'all_52_card_targets_accounted_for', 'all_raw_events_retained',
            'one_fold_per_beer_and_all_its_ratings', 'saved_tables_recomputed_from_sources'],
        normalization_all_50_recipe_reproduced=summary['normalization_all_50_recipe_reproduced'],
        models_trained=0)
    path = args.output / 'verification.json'
    if args.verify_only:
        assert json.loads(path.read_text(encoding='utf-8')) == verification
    else:
        path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Original contest attachment folder')
    parser.add_argument('--raw', type=Path, default=MANIFEST.parent / 'Raw sensory data anonymized.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/raw_sensory_audit_20261025')
    parser.add_argument('--verify-only', action='store_true')
    run(parser.parse_args())
