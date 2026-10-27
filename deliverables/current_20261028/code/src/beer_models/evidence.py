"""Explicit name resolution and conservative sensory-evidence annotations."""
import re
import unicodedata
import pandas as pd
from beer_audit.checks import repair_utf8_mojibake


def normalize_name(value):
    value = unicodedata.normalize('NFKC', repair_utf8_mojibake(str(value))).casefold()
    return re.sub(r'[^\w]+', '', value)


def resolve_compound(field, name, reference, aliases):
    alias = aliases.get(field)
    if alias:
        if normalize_name(name) != normalize_name(alias['expected_dataset_name']):
            raise ValueError('Dataset name changed for explicit alias: ' + field)
        key = normalize_name(alias['reference_compound'])
    else:
        key = normalize_name(name)
    hits = reference.loc[reference.Compound.map(normalize_name).eq(key)]
    if len(hits) != 1:
        raise ValueError(f'Expected one exact reference match for {field}, found {len(hits)}')
    return hits.iloc[0], 'explicit_alias' if alias else 'normalized_exact'


def annotate_relation(target, field, flavor, origin, signed_coefficient, config):
    descriptor = target[2:] if target.startswith(('A_', 'F_')) else target
    terms = config['descriptor_terms'][descriptor]
    text = '' if pd.isna(flavor) else str(flavor).casefold()
    matched = [term for term in terms if re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', text)]
    identity = field in config['identity_markers'].get(target, [])
    if identity:
        relation = 'named_marker_matches_target'
        reason = '该指标对应目标命名的物质或直接理化指标，支持化学相关性。'
    elif matched:
        relation = 'reference_descriptor_overlap'
        reason = '原研究参考描述与目标词义有交集：' + ', '.join(matched) + '。该判定只核查描述一致性。'
    else:
        relation = 'target_specific_evidence_insufficient'
        reason = '当前参考描述未提供该目标的明确直接支持；可能反映风格共变或相关物质替代，不能据此确定直接作用。'
    origin_text = '' if pd.isna(origin) else str(origin).strip()
    reason += ' 参考来源类别：' + (origin_text or '未说明') + '。'
    if signed_coefficient < 0:
        reason += ' Ridge条件系数为负；即使描述相符，也不能解释成增浓后增强该风味。'
    reason += ' 系数不等于物理浓度效应，未证明本数据中的因果效应。'
    return relation, ';'.join(matched), reason
