"""Rebuild the adopted deployment models without rerunning model-selection history."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from beer_audit.audit import sha256, write_json, save_csv
from beer_audit.checks import numeric_frame
from .panel_data import load_inputs
from .raw_panel import fit_shared, portrait
from .selective_tail import SelectiveTailPoint
from .tail_risk import RawOrdinalModel, fit_risk, base_probabilities, survival
from .style_models import make_style_model

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT/'outputs/current_models'
Q3 = ROOT/'outputs/q3_missing_aware_20261027'
DEFECTS = ['dms', 'diacetyl', 'X4vg', 't2n', 'stale_hops']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def train(source, raw, output):
    """Fixed full-data fits only; historical OOF probabilities fit risk calibration."""
    bp = read_json(ROOT/'configs/r2_raw_panel_protocol_20261026.json')
    rp = read_json(ROOT/'configs/r2b_tail_risk_protocol_20261026.json')
    d = load_inputs(source, raw, bp)
    calibration = ROOT/'data/current/risk_full_training.csv.gz'
    if sha256(calibration) != read_json(ROOT/'data/current/risk_provenance.json')['sha256']:
        raise ValueError('Historical inner-OOF calibration input changed')
    output.mkdir(parents=True, exist_ok=False)
    positions = np.arange(len(d['x']))
    print('Training fixed official Q1 ET + retained ordinal risk source', flush=True)
    shared = fit_shared(d, positions, bp, bp['model_seed']+999)
    ordinal = RawOrdinalModel(rp).fit(d['x'], d['distribution'])
    inner = pd.read_csv(calibration, float_precision='round_trip')
    probabilities = {name: np.stack([part.pivot(index='beer', columns='target', values=f'p{k}')
        .loc[d['ids'].beer, d['targets']].to_numpy() for k in range(7)], axis=2)
        for name, part in inner.groupby('source')}
    risk, _ = fit_risk(dict(extra_trees=shared, raw_ordinal=ordinal), probabilities, d['distribution'], rp)
    status = ['exploratory_not_safety_validation']*len(d['targets'])
    for row in pd.read_csv(ROOT/'outputs/r2b_tail_risk_20261026/defect_risk_validation.csv').itertuples():
        status[d['targets'].index(row.target)] = row.status
    bundle = dict(bundle_kind='r2b_raw_panel', selected_model='extra_trees__median',
        raw_panel_model=SelectiveTailPoint(shared, d['distribution'].mean((0, 1)), None, d['parents']),
        risk_panel_model=risk, risk_status=status, features=d['features'], targets=d['targets'],
        primary=d['primary'], scorecard_scope=d['scope'].to_dict('records'), protocol=bp, r2b_protocol=rp,
        point_semantics='integer_balanced_panel_conditional_median_with_parent_cap',
        score_status='raw_panel_balanced_median_prediction', grade_method='r2_raw_panel_median',
        raw_source_sha256=d['manifest']['sha256'], qualified_on_historical_grouped_folds=True,
        tail_point_qualified_on_historical_folds=False,
        point_deployment_status='provisional_baseline_tail_not_qualified',
        risk_provenance='Retained historical training-inner-OOF selection and calibration; no new evaluation')
    joblib.dump(bundle, output/'q1.joblib', compress=3)
    print('Training fixed Q3 auxiliary ET223 + two style classifiers', flush=True)
    omitted = 'X2.3.butanedione'
    keep = np.array([j for j, f in enumerate(d['features']) if f != omitted])
    small = dict(d, x=d['x'][:, keep], features=[d['features'][j] for j in keep])
    sc = read_json(ROOT/'configs/q3_style_protocol.json')
    panel = fit_shared(small, positions, bp, 202612099)
    classifiers = {name: make_style_model(name, sc, sc['seed']).fit(small['x'], d['ids'].tasting_category_fine)
                   for name in ['logistic', 'extra_trees']}
    joblib.dump(dict(panel=panel, classifiers=classifiers, keep_indices=keep,
        features=small['features'], omitted_input=omitted), output/'q3.joblib', compress=3)
    write_json(output/'training.json', dict(status='completed', deployment_fits_only=True,
        new_external_validation=False, q1_high_tail_resolved=False, Stale_resolved=False,
        raw_sha256=sha256(raw), calibration_sha256=sha256(calibration),
        source_sha256={name: sha256(source/name) for name in ['1.csv', '4.csv', '8.xlsx']},
        model_sha256={name: sha256(output/name) for name in ['q1.joblib', 'q3.joblib']}))
    print('Deployment models ready: '+str(output), flush=True)


def auxiliary_defects(frame, q3, targets):
    """Q3 auxiliary risks; never substitute these for official Q1 scores."""
    x = numeric_frame(frame.reindex(columns=q3['features'])).to_numpy(float)
    if frame.empty or (~np.isfinite(x).any(axis=1)).any():
        raise ValueError('Q3 inputs require observed model features')
    grade = q3['panel'].predict(x)['grade']
    risk = survival(base_probabilities(q3['panel'], x))
    beers = frame['beer'].astype('string') if 'beer' in frame else pd.Series([f'row_{i+1:06d}' for i in range(len(x))])
    if beers.isna().any() or beers.str.strip().eq('').any() or beers.duplicated().any():
        raise ValueError('beer must be nonempty and unique')
    rows = []
    for i, beer in enumerate(beers):
        for name in DEFECTS:
            j = targets.index(name)
            rows.append(dict(beer=beer, target=name, auxiliary_grade=int(grade[i, j]),
                probability_ge1=float(risk[i, j, 0]), probability_ge5=float(risk[i, j, 4]),
                probability_6=float(risk[i, j, 5]),
                nonzero_validation_supported=name != 'stale_hops', tail_validation_established=False,
                probability_source='Q3_missing_aware_ET223_uncalibrated', safety_certified=False))
    styles = []
    for name, model in q3['classifiers'].items():
        probabilities = model.predict_proba(x)
        target = list(model.classes_).index('Tripel')
        for beer, guess, prob in zip(beers, model.predict(x), probabilities[:, target]):
            styles.append(dict(beer=beer, classifier=name, predicted_style=guess,
                               Tripel_probability=float(prob), calibration_established=False))
    return pd.DataFrame(rows), pd.DataFrame(styles)


GENERATED_CANDIDATES = [ROOT/'outputs/q3_copula_v3_20261028',
                        ROOT/'outputs/q3_copula_v2_20261028',
                        ROOT/'outputs/q3_copula_20261028',
                        ROOT/'outputs/task4_raw_q3_20261026']


def _adoption_ok(folder):
    """A generated spectrum set is current only if its own gate says adopted."""
    check = folder/'verification.json'
    if not check.exists():
        return False
    data = read_json(check)
    if 'replacement_gate_passed' in data:
        return bool(data.get('integrity_passed')) and bool(data.get('replacement_gate_passed'))
    return data.get('status') == 'passed' or bool(data.get('all_pass'))


def generated_source(preferred=None):
    """Directory holding the officially adopted generated spectra.

    A 20261028 copula output replaces the retained 20261027 interpolation only
    after its pre-registered replacement gate passes; otherwise the retained
    output stays current. ``preferred`` bypasses the rule for explicit diagnostics.
    """
    if preferred is not None:
        folder = Path(preferred)
        if not (folder/'generated_profiles.csv').exists():
            raise FileNotFoundError(str(folder/'generated_profiles.csv'))
        return folder
    for folder in GENERATED_CANDIDATES:
        if (folder/'generated_profiles.csv').exists() and _adoption_ok(folder):
            return folder
    raise FileNotFoundError('No verified generated profiles; pass --generated explicitly')


def render(models, output, generated=None):
    """Refresh adopted fixed profiles; no search, calibration or fitting."""
    official, auxiliary = [joblib.load(models/name) for name in ['q1.joblib', 'q3.joblib']]
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.concat([pd.read_csv(Q3/'source119_profile.csv'), pd.read_csv(Q3/'transfer119_profile.csv')], ignore_index=True)
    scores, _, _, _ = portrait(frame, official)
    save_csv(output, 'transfer_official_Q1_scores52.csv', scores)
    defects, styles = auxiliary_defects(frame, auxiliary, official['targets'])
    save_csv(output, 'transfer_auxiliary_Q3_risks.csv', defects)
    save_csv(output, 'transfer_styles.csv', styles)
    source = generated_source(generated)
    generated = pd.read_csv(source/'generated_profiles.csv').rename(columns={'generated_id':'beer'})
    scores, _, _, _ = portrait(generated, official)
    scores.to_csv(output/'generated_official_Q1_scores52.csv.gz', index=False, compression={'method':'gzip','mtime':0})
    write_json(output/'render.json', dict(status='completed', training_required=False, search_required=False,
        generated_profiles=len(generated), transfer_profiles=len(frame), actual_after_measured=False,
        generated_source=source.relative_to(ROOT).as_posix(),
        generated_sha256=sha256(source/'generated_profiles.csv'),
        q1_model_sha256=sha256(models/'q1.joblib'), q3_model_sha256=sha256(models/'q3.joblib')))
