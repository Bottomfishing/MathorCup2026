"""Finite nested Stale-risk comparison with immutable adoption rules."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json, sha256
from beer_models.baseline import make_splits
from beer_models.stale_risk import StaleNonzeroModel
from beer_models.tail_risk import base_probabilities, binary_loss
from beer_models.explain import stratified_draw
from r2_raw_q1 import load_inputs, RAW_MANIFEST, PROTOCOL as BASE, content_hash
from q1_target_experiment import fit_shared

PROTOCOL = ROOT/'configs/q3_stale_missing_protocol_20261027.json'


def run(args):
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    bp = json.loads(BASE.read_text(encoding='utf-8'))
    d = load_inputs(args.source, args.raw, bp)
    out = args.output; out.mkdir(parents=True, exist_ok=False); (out/'models').mkdir()
    write_json(out/'protocol.json', p)
    dictionary = pd.read_csv(ROOT/'outputs/p0_audit_20261023_final/feature_dictionary.csv').set_index('field_name_R').loc[d['features']]
    aging = np.flatnonzero(dictionary.aging.notna().to_numpy())
    assert len(aging) == 24
    save_csv(out, 'aging_features.csv', dictionary.iloc[aging].reset_index())
    j = d['targets'].index('stale_hops'); q = d['distribution'][:, j]; y = 1-q[:, 0]
    x, styles = d['x'], d['ids'].tasting_category_fine.to_numpy()
    save_csv(out, 'label_diagnostics.csv', [dict(grade=k, original_record_count=int((d['scores'][:, j] == k).sum()),
        beer_balanced_mass=float(q[:, k].sum()), beers_with_grade=int((q[:, k] > 0).sum())) for k in range(7)])
    methods = p['stale_candidates']
    prediction = {name: np.empty((len(p['outer_seeds']), len(y))) for name in methods+['inner_selected']}
    split_rows, inner_rows, fits, choices = [], [], [], []
    def fit(train, seed, context):
        models = {}
        for name in methods:
            model = fit_shared(d, train, bp, seed) if name == 'shared_et' else StaleNonzeroModel(name, aging, seed).fit(x[train], y[train])
            if name != 'training_prior':
                relevant = np.arange(x.shape[1]) if name in ['shared_et', 'all_et_leaf10'] else aging
                np.testing.assert_allclose(model.imputer.statistics_, np.nanmedian(x[train][:, relevant], axis=0), atol=0, rtol=0)
            models[name] = model
            fits.append(dict(context=context, method=name, seed=seed, training_beers=len(train)))
        return models
    def predict(models, rows):
        return {name: 1-base_probabilities(model, x[rows])[:, j, 0] if name == 'shared_et' else model.predict(x[rows]) for name, model in models.items()}
    def choose(train, repeat, outer_fold):
        probs = {name: np.empty(len(train)) for name in methods}
        for fold, (it, iv) in enumerate(make_splits(styles[train], p['inner_folds'], p['inner_seed']+repeat*10+outer_fold)):
            models = fit(train[it], p['model_seed']+repeat*1000+outer_fold*100+fold, f'r{repeat}_outer{outer_fold}_inner{fold}')
            values = predict(models, train[iv])
            for name in methods: probs[name][iv] = values[name]
            split_rows.extend(dict(repeat=repeat, outer_fold=outer_fold, inner_fold=fold, role=role, beer=int(d['ids'].beer.iloc[pos]))
                for role, positions in [('train', train[it]), ('validation', train[iv])] for pos in positions)
        losses = {name: float(binary_loss(prob, y[train]).mean()) for name, prob in probs.items()}
        best = min(losses.values()); chosen = next(name for name in methods if losses[name] <= best+1e-12)
        inner_rows.extend(dict(repeat=repeat, outer_fold=outer_fold, method=name, brier=loss, selected=name == chosen) for name, loss in losses.items())
        return chosen
    for repeat, seed in enumerate(p['outer_seeds']):
        for fold, (train, test) in enumerate(make_splits(styles, p['outer_folds'], seed)):
            chosen = choose(train, repeat, fold)
            models = fit(train, p['model_seed']+repeat*1000+fold*100+99, f'r{repeat}_outer{fold}')
            values = predict(models, test)
            for name in methods: prediction[name][repeat, test] = values[name]
            prediction['inner_selected'][repeat, test] = values[chosen]
            choices.append(dict(repeat=repeat, outer_fold=fold, selected=chosen))
            joblib.dump(dict(models=models, train=train, test=test, selected=chosen), out/'models'/f'r{repeat}_fold{fold}.joblib', compress=3)
            split_rows.extend(dict(repeat=repeat, outer_fold=fold, inner_fold=-1, role=role, beer=int(d['ids'].beer.iloc[pos])) for role, positions in [('outer_train', train), ('outer_test', test)] for pos in positions)
            print(f'Stale outer {repeat}/{fold}: {chosen}', flush=True)
    rows, metrics, losses = [], [], {}
    for name, values in prediction.items():
        loss = binary_loss(values, y[None]); losses[name] = loss
        for repeat, prob in enumerate(values):
            auc = roc_auc_score(np.r_[np.zeros(len(y)), np.ones(len(y))], np.r_[prob, prob], sample_weight=np.r_[1-y, y])
            ap = average_precision_score(np.r_[np.zeros(len(y)), np.ones(len(y))], np.r_[prob, prob], sample_weight=np.r_[1-y, y])
            metrics.append(dict(method=name, repeat=repeat, expected_brier=float(loss[repeat].mean()), fraction_MSE=float(np.square(prob-y).mean()),
                soft_label_AUC=float(auc), soft_label_AP=float(ap), prediction_sd=float(prob.std()), observed_prevalence=float(y.mean()), mean_prediction=float(prob.mean())))
            rows.extend(dict(method=name, repeat=repeat, beer=int(beer), observed_fraction=float(actual), prediction=float(pred), expected_brier=float(err))
                for beer, actual, pred, err in zip(d['ids'].beer, y, prob, loss[repeat]))
    rng = np.random.default_rng(p['bootstrap']['seed'])
    intervals = []
    for baseline, quantiles in [('training_prior', [.005, .995]), ('shared_et', [.025, .975])]:
        delta = (losses[baseline]-losses['inner_selected']).mean(0)
        samples = [delta[stratified_draw(styles, rng)].mean() for _ in range(p['bootstrap']['repeats'])]
        intervals.append(dict(baseline=baseline, mean_gain=float(delta.mean()), lower=float(np.quantile(samples, quantiles[0])), upper=float(np.quantile(samples, quantiles[1])), interval_level=.99 if baseline == 'training_prior' else .95))
    frame = pd.DataFrame(metrics); selected = frame.loc[frame.method.eq('inner_selected')]
    prior_gain = frame.loc[frame.method.eq('training_prior'), 'expected_brier'].to_numpy()-selected.expected_brier.to_numpy()
    gates = dict(mean_gain=float(prior_gain.mean()) >= p['adoption']['minimum_mean_prior_Brier_gain'],
        prior_interval=intervals[0]['lower'] > 0, shared_interval=intervals[1]['lower'] > 0,
        every_repeat=bool((prior_gain > 0).all()), AUC=float(selected.soft_label_AUC.mean()) >= p['adoption']['minimum_mean_soft_label_AUC'],
        variation=float(selected.prediction_sd.mean()) >= p['adoption']['minimum_selected_prediction_sd'])
    adopted = all(gates.values())
    for name, data in [('oof_predictions.csv', rows), ('metrics.csv', metrics), ('paired_intervals.csv', intervals), ('inner_selection.csv', inner_rows), ('outer_choices.csv', choices), ('split_membership.csv', split_rows), ('fit_log.csv', fits)]: save_csv(out, name, data)
    # Deployment fitting is conditional on predictive evidence, not merely a finite probability output.
    if adopted:
        chosen = choose(np.arange(len(y)), len(p['outer_seeds']), 0)
        full = fit(np.arange(len(y)), p['model_seed']+99999, 'full_training')
        joblib.dump(dict(model=full[chosen], method=chosen, aging_indices=aging, features=d['features'], target='stale_hops_nonzero', protocol=p), out/'model.joblib', compress=3)
    write_json(out/'summary.json', dict(status='completed', adopted=adopted, gates=gates, primary='individual_nonzero_Brier',
        selected_mean_brier=float(selected.expected_brier.mean()), selected_mean_AUC=float(selected.soft_label_AUC.mean()),
        mean_prior_gain=float(prior_gain.mean()), protocol_sha256=content_hash(p), raw_sha256=sha256(args.raw),
        source_sha256={f: sha256(args.source/f) for f in ['1.csv','3.xlsx','4.csv','8.xlsx']}, independent_beers=250,
        repeated_predictions=750, new_external_data=False, q1_integer_model_changed=False, true_intervention_safety_certified=False))
    print(frame.groupby('method')[['expected_brier','soft_label_AUC','prediction_sd']].mean().to_string(), flush=True)
    print(f'ADOPTED={adopted}; gates={gates}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/q3_stale_risk_20261027')
    with threadpool_limits(limits=1): run(parser.parse_args())
