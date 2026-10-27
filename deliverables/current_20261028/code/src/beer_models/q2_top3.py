"""Q2 compound-flavour Top-3 library.

Pure functions, no file IO and no absolute paths. Everything is parameterised by
arrays and field names so the official script and the tests share one code path:

  * co-expression graph from a supplied Spearman matrix (|rho| >= threshold);
  * Louvain communities via ``networkx.algorithms.community.louvain_communities``
    (explicit seed, no global state);
  * graph-neighbourhood importance ``score = |beta| + weight * neighbour_mean|beta|``;
  * per-target Top-k fields;
  * beer-level bootstrap Top-k stability;
  * multi-method Top-k Jaccard;
  * A/B/C chemical-evidence grading.

AI-assisted: OpenAI Codex, 2026-10-28.
"""
import re
import unicodedata

import numpy as np
import networkx as nx
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

META_COLUMNS = ('beer', 'beer_id', 'tasting_category_fine')

#: Whole-beer chemistry / derived aggregates are excluded from the *compound* ranking.
DEFAULT_EXCLUDED = ('pH', 'color.EBC.', 'bitterness', 'GFS', 'unfermentables', 'kcalperc',
                    'esters_sum', 'aroma_hops_sum', 'acids_sum', 'sulfur_sum',
                    'protein.g.L.', 'betaglucan.mg.L.')

DEFAULT_ALPHA = 1000.0
DEFAULT_GRAPH_THRESHOLD = 0.5
DEFAULT_NEIGHBOUR_WEIGHT = 0.5
DEFAULT_BOOTSTRAP = 200
DEFAULT_SEED = 20261027
DEFAULT_LOUVAIN_SEED = 42
DEFAULT_EXTRA_TREES = {'n_estimators': 300, 'max_features': 0.33,
                       'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1}

#: Direct identity markers: sensory target -> field names in 1.csv.
IDENTITY_MARKERS = {
    'X4vg': ['X4VG'], 'diacetyl': ['X2.3.butanedione'], 'dms': ['DMS'],
    'metallic': ['iron'], 't2n': ['X.E..2.nonenal'],
    'lactic': ['lactic_acid.mg.L.'], 'acetic': ['acetic_acid.g.L.'],
    'acidity': ['lactic_acid.mg.L.', 'acetic_acid.g.L.'],
    'alcohol': ['ethanol..v.v.'], 'co2': ['CO2.PSI.'],
}

#: class_key -> (name fragments, class key, Chinese class name, mechanism, sensory families).
CLASS_RULES = (
    (('diacetyl', 'butanedione'), 'diacetyl', '双乙酰(2,3-丁二酮)',
     '奶油/黄油香，啤酒中典型发酵副产物，阈值约0.1 mg/L量级；6.csv Diacetyl 行可作经验佐证',
     {'diacetyl'}),
    (('dimethylsulfide', 'dms', 'diethylsulfide', 'dimethyldisulfide', 'methanethiol', 'hydrogensulfide',
      'h2s', 'sulfurdioxide', 'so2', 'cs2', 'carbondisulfide'),
     'sulfur', '硫化物类', '还原味/熟蔬菜/硫味；DMS 阈值约17-25 μg/L，off_flavors 标注 cooked vegetables/corn/onion',
     {'dms', 'stale', 'sulfur'}),
    (('vinylguaiacol', 'guaiacol', 'ethylguaiacol', 'ethylphenol', 'cresol', 'eugenol', 'vanillin',
      'isoeugenol', 'methoxycinnamaldehyde', 'cinnamaldehyde', 'salicylaldehyde'),
     'phenol', '酚类/香料', '丁香/烟熏/香料；4VG 为阿魏酸脱羧产物，对应 X4vg 与 clove 维度；酚类阈值低、感知强',
     {'phenol', 'clove', 'coriander', 'malt'}),
    (('furfural', 'furfuryl', 'acetylfuran', 'furanone', 'furaneol', 'maltol', 'isomaltol', 'furoate',
      'methylfurfural', 'hydroxymethylfurfural', 'furan'),
     'maillard', '呋喃/美拉德产物', '焦糖/面包/烘烤香；美拉德反应与麦芽焙烤产物，是麦芽香与焦糖维度的直接化学来源',
     {'malt', 'sweet', 'body', 'overall', 'aftertaste'}),
    (('pyrazine',), 'pyrazine', '吡嗪类', '烘烤/坚果/可可香；美拉德反应产物，常见于深色麦芽',
     {'malt', 'overall', 'aftertaste'}),
    (('linalool', 'geraniol', 'citronellol', 'nerol', 'nerolidol', 'terpineol', 'terpinen', 'terpinolene',
      'ocimene', 'ocimenone', 'phellandrene', 'limonene', 'eucalyptol', 'camphor', 'isoborneol',
      'caryophyllene', 'humulene', 'farnesene', 'farnesol', 'farnesal', 'calacorene', 'muurolol',
      'eudesmol', 'cubenol', 'cadinene', 'ionone', 'damascenone', 'cyclocitral', 'cyclohomocitral',
      'terpinyl', 'cymenene', 'cymene', 'manool', 'kongol', 'eoeniculin', 'tobacarol', 'epicubenol',
      'dihydroapofarnesal', 'linalooloxide', 'neroloxide'),
     'terpene', '酒花萜类/倍半萜', '花香/柑橘/草本/木质与酒花香气；酒花来源，萜烯在发酵与老化中转化，是 hops/orange 维度的核心化学基础',
     {'hops', 'orange', 'coriander', 'esters', 'overall'}),
    (('acetate', 'butyrate', 'butanoate', 'hexanoate', 'octanoate', 'decanoate', 'propionate', 'isovalerate',
      'isobutyrate', 'tiglate', 'benzoate', 'cinnamate', 'phenylacetate', 'succinate', 'caprylate',
      'lactate', 'formate', 'heptanoate', 'nonanoate', 'pentanoate'),
     'ester', '酯类', '果香/香蕉/花香/nail polish；酵母乙酰/酰基转移酶合成，是 esters 与 fruity 维度的直接来源',
     {'esters', 'hops', 'sweet', 'overall', 'alcohol'}),
    (('acetaldehyde', 'nonanal', 'nonenal', 'decadienal', 'benzaldehyde', 'phenylacetaldehyde',
      'methylbutanal', 'methyl2butenal', 'methylbutyraldehyde', 'isovaleraldehyde', 'hexanal',
      'heptanal', 'octanal', 'methional'),
     'aldehyde', '醛类', '青苹果/生青/纸板/坚果；乙醛为乙醇前体，壬烯醛为脂肪氧化陈腐标志物，醛类是 acetaldehyde/t2n 维度的直接来源',
     {'acetaldehyde', 'stale', 'malt', 'overall'}),
    (('ethanol', 'propanol', 'butanol', 'pentanol', 'hexanol', 'heptanol', 'octanol', 'nonanol', 'decanol',
      'isoamylalcohol', 'isobutanol', 'phenethylalcohol', 'benzylalcohol', 'methylbutanol', 'hexenol',
      'octenol', 'methylhexanol', 'methyloctanol', 'methionol'),
     'alcohol', '醇类', '酒精/溶剂/玫瑰香；高级醇由酵母氨基酸代谢生成，是 alcohol 与 body 维度的重要贡献者',
     {'alcohol', 'body', 'overall', 'esters'}),
    (('lactic', 'acetic', 'butyric', 'isovaleric', 'hexanoic', 'octanoic', 'decanoic', 'succinic', 'propionic'),
     'acid', '有机酸', '酸味/酸奶/醋味；乳酸菌与醋酸菌代谢产物，是 acidity/lactic/acetic 维度的直接来源',
     {'acid', 'lactic', 'acetic', 'overall'}),
    (('glycerol',), 'polyol', '甘油', '甜润/酒体厚重感；发酵副产物，主要影响 body 与甜感',
     {'body', 'sweet', 'aftertaste', 'overall'}),
    (('iron',), 'iron', '铁离子', '金属味；铁离子催化氧化并直接贡献 metallic 味', {'iron'}),
    (('isoalphaacids', 'bitterness', 'alphaacid', 'betaacid'), 'bitter', '异α-酸/苦味物质',
     '苦味；酒花α-酸异构化产物，是 bitterness 维度的直接化学基础', {'bitter'}),
    (('carbondioxide', 'co2'), 'gas', '二氧化碳', '碳酸感/杀口感；直接对应 co2 维度', {'gas'}),
    (('sugars', 'gfs', 'glucose', 'fructose', 'maltose', 'unfermentables', 'kcalperc'),
     'sugar', '糖类/热量', '甜味/麦芽香/酒体；可发酵糖与残糖直接贡献 sweetness/body',
     {'sweet', 'body', 'malt', 'overall'}),
    (('protein', 'betaglucan', 'glycerin'), 'protein', '蛋白/β-葡聚糖', '酒体/口感厚度；大分子物质影响 body 与 aftertaste',
     {'body', 'aftertaste'}),
    (('color', 'ebc'), 'color', '色度', '焦香/烘烤外观与麦芽香；色度与深色麦芽焙烤产物共变，可作 malt 维度的外观佐证',
     {'malt', 'body', 'overall'}),
    (('ammonia', 'methylthio', 'dimethyltrisulfide', 'hydrogensulfide'), 'off', '含硫/含氮异味物',
     '还原味/异味；off_flavors 标注的异味化合物，与 stale/硫味维度相关', {'stale', 'dms', 'sulfur'}),
    (('undecanone', 'nonanone', 'heptanone', 'pentanone', 'methylheptenone', 'acetophenone', 'methylketone',
      'butanone', 'methylbutanone'), 'ketone', '酮类', '果香/奶油/青草；部分甲基酮是发酵与氧化产物',
     {'esters', 'malt', 'overall'}),
)

DIRECT_MECHANISM_CLASSES = frozenset({'diacetyl', 'sulfur', 'phenol', 'acid', 'iron', 'bitter', 'gas', 'sugar'})

ORIGIN_CN = {'Yeast': '酵母代谢', 'Hops': '酒花来源', 'Malt': '麦芽来源', 'Spices': '香料添加',
             'Others': '其他来源', 'Wild flora': '野生菌群', 'Internal standard solution': '内标'}

_TARGET_FAMILY_OVERRIDES = {
    'acidity': {'acid'}, 'bitternes': {'bitter'}, 'sweetness': {'sweet'},
    'x4vg': {'phenol', 'clove'}, 'diacetyl': {'diacetyl'}, 'dms': {'dms', 'sulfur'},
    'metallic': {'iron'}, 'stale_hops': {'stale', 'sulfur'}, 't2n': {'stale'},
    'orange': {'hops', 'terpene'}, 'coriander': {'terpene', 'phenol'},
    'clove': {'phenol', 'clove'}, 'lactic': {'acid', 'lactic'}, 'acetic': {'acid', 'acetic'},
    'barnyard': {'other'}, 'alcohol': {'alcohol'},
    'aftertaste': {'body', 'aftertaste', 'overall'},
    'body': {'body', 'protein', 'sugar'}, 'co2': {'gas'},
    'overall': {'malt', 'hops', 'esters', 'alcohol', 'body', 'sweet'},
}

_GREEK = (('α', 'alpha'), ('β', 'beta'), ('γ', 'gamma'), ('δ', 'delta'))


def canon_key(name):
    """Fold Greek letters and strip non-alphanumerics so 2.csv/6.csv names join safely."""
    text = unicodedata.normalize('NFKC', str(name))
    for greek, latin in _GREEK:
        text = text.replace(greek, latin)
    return re.sub(r'[^a-z0-9]', '', text.lower())


def candidate_features(fields, excluded=DEFAULT_EXCLUDED):
    """Feature names eligible for the compound ranking, preserving input order."""
    blocked = set(excluded)
    return [f for f in fields if f not in blocked]


def adjacency_from_corr(rho, threshold=DEFAULT_GRAPH_THRESHOLD):
    """Binarised co-expression graph; returns (float adjacency, edge records).

    ``rho`` is the square Spearman matrix (node order equals feature order). The
    diagonal is zeroed so a compound is never its own neighbour.
    """
    corr = np.asarray(rho, dtype=float).copy()
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError('rho must be square')
    np.fill_diagonal(corr, 0.0)
    weights = np.abs(corr)
    adjacency = (weights >= threshold).astype(float)
    edges = [{'source': int(i), 'target': int(j), 'weight': float(weights[i, j])}
             for i, j in zip(*np.triu_indices_from(adjacency, k=1)) if adjacency[i, j] > 0]
    return adjacency, edges


def louvain_labels(adjacency, seed=DEFAULT_LOUVAIN_SEED):
    """Louvain community id per node via networkx, deterministic for a fixed seed.

    ``adjacency`` holds non-negative edge weights: a 0 entry means "no edge", any
    positive value is used as the edge weight (pass |rho| for the weighted graph,
    or a 0/1 matrix for an unweighted one). Communities are relabelled by
    decreasing size (stable for equal sizes), so community id 0 is the largest.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(adjacency.shape[0]))
    upper = np.triu_indices_from(adjacency, k=1)
    for i, j in zip(*upper):
        if adjacency[i, j] > 0:
            graph.add_edge(int(i), int(j), weight=float(adjacency[i, j]))
    communities = nx.algorithms.community.louvain_communities(graph, seed=seed)
    communities = sorted(communities, key=len, reverse=True)
    labels = np.full(adjacency.shape[0], -1, dtype=int)
    for cid, members in enumerate(communities):
        for node in members:
            labels[node] = cid
    return labels, [len(c) for c in communities]


def adjacency_subset(adjacency, keep):
    """Sub-matrix of the co-expression graph restricted to candidate node indices."""
    keep = np.asarray(keep, dtype=int)
    return adjacency[np.ix_(keep, keep)]


def preprocess(X, Y):
    """Median-impute columns, asinh-compress and standardise X; centre Y.

    Imputation medians are computed on the given rows only, so bootstrap
    replicates refit the whole preprocessing chain (no leakage from the full data).
    """
    values = np.array(X, dtype=float, copy=True)
    medians = np.nanmedian(values, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    missing = np.isnan(values)
    if missing.any():
        values[missing] = np.take(medians, np.where(missing)[1])
    compressed = np.arcsinh(values)
    mean = compressed.mean(axis=0)
    scale = compressed.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardised = (compressed - mean) / scale
    return standardised, np.asarray(Y, dtype=float) - np.asarray(Y, dtype=float).mean(axis=0, keepdims=True)


def ridge_coefficients(X, Y, alpha=DEFAULT_ALPHA):
    """Multi-output ridge on already preprocessed X; returns (n_features, n_targets)."""
    gram = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(gram, X.T @ Y)


def neighbourhood_scores(beta, adjacency, weight=DEFAULT_NEIGHBOUR_WEIGHT):
    """``score_j = |beta_j| + weight * neighbour_mean|beta|`` for every target column.

    The self term keeps direct identity markers on top; the neighbour term uses
    the 2.csv co-expression graph as context (no centrality multiplier on the self term).
    """
    degree = adjacency.sum(axis=1)
    safe = np.where(degree > 0, degree, 1.0)
    scores = np.empty_like(beta)
    for j in range(beta.shape[1]):
        magnitude = np.abs(beta[:, j])
        scores[:, j] = magnitude + weight * (adjacency @ magnitude) / safe
    return scores


def top_k_fields(scores, fields, k=3):
    """Top-k field names per target column by absolute score; ties keep input order."""
    fields = list(fields)
    return [[fields[i] for i in np.argsort(-np.abs(scores[:, j]), kind='stable')[:k]]
            for j in range(scores.shape[1])]


def extra_trees_importance(X, Y, params=None):
    """Per-target ExtraTrees impurity importance on preprocessed X (one forest per target)."""
    config = dict(DEFAULT_EXTRA_TREES if params is None else params)
    importance = np.zeros((X.shape[1], Y.shape[1]))
    for j in range(Y.shape[1]):
        forest = ExtraTreesRegressor(**config)
        forest.fit(X, Y[:, j])
        importance[:, j] = forest.feature_importances_
    return importance


def marginal_spearman(X, Y):
    """Per-feature marginal Spearman correlation against each target column."""
    corr = np.zeros((X.shape[1], Y.shape[1]))
    for j in range(Y.shape[1]):
        for i in range(X.shape[1]):
            corr[i, j] = spearmanr(X[:, i], Y[:, j]).statistic
    return corr


def bootstrap_indices(n_samples, n_bootstrap=DEFAULT_BOOTSTRAP, seed=DEFAULT_SEED):
    """Sequential beer-level resample indices from one explicitly seeded generator."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_samples, n_samples) for _ in range(n_bootstrap)]


def bootstrap_top_k(X, Y, fields, adjacency, alpha=DEFAULT_ALPHA, n_bootstrap=DEFAULT_BOOTSTRAP,
                    seed=DEFAULT_SEED, weight=DEFAULT_NEIGHBOUR_WEIGHT):
    """Beer-level bootstrap of the full chain; returns per-target replicate tuples of field names.

    X must already be restricted to the candidate columns and adjacency to those
    same columns (use ``adjacency_subset``); Y is the raw target matrix.
    """
    fields = list(fields)
    replicates = [[] for _ in range(Y.shape[1])]
    for rs in bootstrap_indices(X.shape[0], n_bootstrap, seed):
        Xs, Yc = preprocess(X[rs], Y[rs])
        beta = ridge_coefficients(Xs, Yc, alpha)
        scores = neighbourhood_scores(beta, adjacency, weight)
        for j, selection in enumerate(top_k_fields(scores, fields, 3)):
            replicates[j].append(tuple(selection))
    return replicates


def selection_frequencies(replicates, selected, n_bootstrap=DEFAULT_BOOTSTRAP):
    """Bootstrap selection frequency for each full-data Top-k field."""
    counts = {}
    for replicate in replicates:
        for field in replicate:
            counts[field] = counts.get(field, 0) + 1
    return {field: counts.get(field, 0) / n_bootstrap for field in selected}


def jaccard(left, right):
    """Jaccard index of two field collections."""
    left, right = set(left), set(right)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def bootstrap_jaccard(replicates, selected):
    """Per-replicate Top-k Jaccard against the full-data selection."""
    values = [jaccard(replicate, selected) for replicate in replicates]
    return {'mean': float(np.mean(values)), 'min': float(np.min(values)),
            'p10': float(np.percentile(values, 10))}


def method_jaccard_row(target, sets_by_method, pairs):
    """One per-target row of pairwise Top-k Jaccard and shared-count values."""
    row = {'target': target}
    values = []
    for left, right in pairs:
        shared = len(set(sets_by_method[left][target]) & set(sets_by_method[right][target]))
        row[f'jaccard_{left}_vs_{right}'] = jaccard(sets_by_method[left][target], sets_by_method[right][target])
        row[f'overlap_n_{left}_vs_{right}'] = shared
        values.append(row[f'jaccard_{left}_vs_{right}'])
    row['jaccard_mean'] = float(np.mean(values))
    row['jaccard_min'] = float(np.min(values))
    row['n_pairs_ge_0.5'] = int(sum(v >= 0.5 for v in values))
    return row


def classify_compound(name):
    """Mechanism class for a compound display name: (class_key, class_name, mechanism, families)."""
    key = canon_key(name)
    for fragments, class_key, class_name, mechanism, families in CLASS_RULES:
        if any(fragment in key for fragment in fragments):
            return class_key, class_name, mechanism, families
    return ('other', '其他化合物',
            '通过共表达图邻域与1.csv丰度关联进入Top-3；机理直接证据不足，列为探索性关联', {'other'})


def target_families(targets):
    """Map each sensory target to the mechanism families it is expected to reflect."""
    mapping = {}
    for target in targets:
        text = str(target).lower()
        if text.startswith(('a_malt', 'f_malt')):
            family = {'malt'}
        elif text.startswith(('a_hops', 'f_hops')):
            family = {'hops'}
        elif text.startswith(('a_esters', 'f_esters')):
            family = {'esters'}
        else:
            family = set(_TARGET_FAMILY_OVERRIDES.get(text, {'overall'}))
        mapping[target] = family
    return mapping


def evidence_grade(target, field, coefficient, cross_corr, frequency, identity_markers=IDENTITY_MARKERS,
                   display_name=None):
    """A/B/C evidence tier, human-readable basis and confidence for one Top-3 entry.

    ``display_name`` is the human-readable compound name (e.g. ``4-vinylguaiacol``)
    used for mechanism classification; it defaults to the raw field name.

    The tiers reproduce the delivered Q2 prototype exactly, so the A tier stays a
    small, defensible set and 6.csv cross-correlations are supporting annotations
    rather than tier upgrades:

    A: identity marker of the target, or a direct mechanism class (acid, iron,
       gas, bitter, ...) with bootstrap frequency >= 0.5.
    B: mechanism family matches the target's sensory family (mechanism argument).
    C: exploratory association only (no direct mechanistic support).
    """
    name = str(display_name or field)
    class_key, class_name, mechanism, families = classify_compound(name)
    affinity = bool(families & target_families([target])[target])
    has_corr = cross_corr is not None and not np.isnan(cross_corr)
    correlation_basis = f'6.csv 交叉相关 r={cross_corr:.2f}' if has_corr else ''
    if field in identity_markers.get(target, []):
        tier, basis = 'A', f'身份标记物：{name} 为 {target} 维度同名/直系化合物'
        if has_corr:
            basis += f'，{correlation_basis}'
    elif affinity and class_key in DIRECT_MECHANISM_CLASSES and frequency >= 0.5:
        tier, basis = 'A', f'直接机理物质({class_name}) 且 bootstrap 入选率 {frequency:.2f}≥0.5'
    elif affinity:
        tier, basis = 'B', f'化学机理同族({class_name})，与目标感官语义一致'
    else:
        tier, basis = 'C', '图邻域+1.csv 关联证据；机理仅间接，列为探索性候选'
    if tier == 'A' and frequency >= 0.5:
        confidence = 'high'
    elif tier in ('A', 'B') and frequency >= 0.4:
        confidence = 'medium'
    else:
        confidence = 'low'
    return {'class_key': class_key, 'mechanism_class': class_name, 'mechanism_note': mechanism,
            'target_semantic_match': affinity, 'chemical_evidence': tier,
            'evidence_basis': basis, 'confidence': confidence}


def as_compound_label(field):
    """Readable compound name fallback when no feature dictionary is available."""
    return str(field).replace('X', '').replace('.', ' ').strip()
