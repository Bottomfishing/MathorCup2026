"""Helpers for held-out, outcome-blind matching. AI-assisted: Codex, 2026-10-24."""
import numpy as np


def nearest_reference(query, references, columns):
    """Select by chemical distance only. Outcomes deliberately are not arguments."""
    query,references=np.asarray(query,float),np.asarray(references,float)
    columns=np.asarray(columns,int)
    if query.ndim!=1 or references.ndim!=2 or references.shape[1]!=len(query):
        raise ValueError('Reference dimensions differ')
    if not len(references) or not len(columns) or not np.isfinite(query).all() or not np.isfinite(references).all():
        raise ValueError('Finite, nonempty matching inputs required')
    distance=np.sqrt(np.mean((references[:,columns]-query[columns])**2,axis=1))
    index=int(np.argmin(distance))
    return index,float(distance[index])


def anchored_prediction(observed_reference, predicted_query, predicted_reference):
    values=[np.asarray(x,float) for x in [observed_reference,predicted_query,predicted_reference]]
    if any(not np.isfinite(x).all() for x in values):raise ValueError('Nonfinite outcome')
    return values[0]+values[1]-values[2]
