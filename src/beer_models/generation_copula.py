"""Style-conditional copula generator with empirical marginals and defect projection.

Why this module exists
----------------------
``beer_models.generation.StyleGenerator`` builds synthetic spectra as convex
combinations of two style donors (``method='interpolate'``).  Convex combinations
mechanically shrink variance and under-disperse the style: measured on the Q3
validation folds its ``variance_ratio`` is 0.55-0.86 and Kriek/Lambic show 34-42%
internal near duplicates.  This module implements the alternative that was
diagnosed as the best candidate in the Q3 review (``copula_quantile_projected``):

1. a Gaussian copula in a low dimensional principal-score space (style specific
   score mean and empirical covariance, plus independent residual noise so the
   generator is not restricted to the style's leading subspace);
2. an *empirical quantile map* per coordinate, so every marginal is exactly the
   style's own empirical marginal (this is what removes variance shrinkage
   without inventing values outside the observed support);
3. a data-coordinate defect projection that caps the off-flavour coordinates at
   ``cap_fraction * threshold``.

Coordinate and threshold conventions (REUSED, not redefined)
------------------------------------------------------------
The 224 columns of ``load_inputs()['x']`` are the *attachment-1 coordinates*.
The existing project convention (``scripts/threshold_scenarios.py``) is:

    field                      unit      transform into attachment-1 coordinate
    DMS                        ug/L      log10
    X2.3.butanedione           mg/L      identity (already mg/L)
    X4VG                       mg/L      log10
    X.E..2.nonenal             mg/L      identity   (no sensory threshold)
    stale hops                 -         no chemical coordinate at all

``8.xlsx`` sheet ``Sensory thresholds`` is keyed by ``field_name_R`` and exposes
``threshold_median`` and ``threshold_mean``; the project default is the median
scenario (``threshold_scenarios.py``).  Therefore a caller reads the thresholds
from the workbook and passes e.g. ``{'DMS': 19.0, 'X2.3.butanedione': 0.09,
'X4VG': 0.20}`` in **threshold units**.  This module converts each one into a
coordinate-space cap:

    log10 field : cap = log10(cap_fraction * threshold)
    identity    : cap = cap_fraction * threshold

Capping is monotone non-increasing, so it can never increase a defect; it is a
*data-coordinate* statement, not a brewing certificate.  Generated spectra are
model/stylistic references, not measured brews.

The module never reads files and contains no absolute path: the caller supplies
``features`` and thresholds.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


def _pairwise(left, right):
    """Euclidean pairwise distances (thin wrapper so the generator never imports scipy)."""
    return cdist(np.asarray(left, float), np.asarray(right, float), "euclidean")

DEFECT_TRANSFORMS = {"DMS": "log10", "X2.3.butanedione": "identity", "X4VG": "log10"}
DEFECT_UNITS = {"DMS": "ug/L", "X2.3.butanedione": "mg/L", "X4VG": "mg/L"}

PC_RESIDUAL = "pc_residual"
DONOR_QUANTILE = "donor_quantile"
DONOR_RESIDUAL = "donor_residual"
PROPOSALS = (PC_RESIDUAL, DONOR_QUANTILE, DONOR_RESIDUAL)
METHOD_NAMES = {PC_RESIDUAL: "copula_quantile_projected", DONOR_QUANTILE: "interp_quantile_projected",
                DONOR_RESIDUAL: "donor_residual_projected"}


def physical_values(coordinate, field):
    """Map an attachment-1 coordinate back to the threshold unit of ``field``.

    Fields outside :data:`DEFECT_TRANSFORMS` are treated as identity (raw
    concentration) columns, which is the project convention for
    ``X.E..2.nonenal``.
    """
    coordinate = np.asarray(coordinate, float)
    if DEFECT_TRANSFORMS.get(field, "identity") == "log10":
        return np.power(10.0, coordinate)
    return coordinate


def defect_caps(thresholds, cap_fraction=0.9, transforms=None, units=None):
    """Convert threshold-unit caps into coordinate-space caps.

    ``thresholds`` maps field name -> threshold in that field's unit.  Fields are
    restricted to :data:`DEFECT_TRANSFORMS` unless ``transforms`` is given.
    """
    if not 0.0 < float(cap_fraction) <= 1.0:
        raise ValueError("cap_fraction must be in (0, 1]")
    transforms = DEFECT_TRANSFORMS if transforms is None else transforms
    caps = {}
    for field, threshold in dict(thresholds).items():
        if field not in transforms:
            raise ValueError("No transform convention for field %r" % (field,))
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("Invalid threshold for %r" % (field,))
        cap = float(cap_fraction) * threshold
        caps[field] = np.log10(cap) if transforms[field] == "log10" else cap
    return caps


def _empirical_quantile_map(values, marginal):
    """Monotone map of ``values`` onto the empirical ``marginal`` (sorted)."""
    values = np.asarray(values, float)
    marginal = np.asarray(marginal, float)
    n, m = values.size, marginal.size
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, float)
    ranks[order] = np.arange(n)
    u = (ranks + 0.5) / n
    grid = (np.arange(m) + 0.5) / m
    return np.interp(u, grid, marginal)


class CopulaQuantileGenerator:
    """Gaussian-copula style generator with empirical marginals + defect projection.

    Parameters
    ----------
    features : sequence of str, optional
        Column names of ``x``; required to locate the defect coordinates.
    components : int
        Number of principal scores used for the copula (default 8).
    defect_thresholds : dict, optional
        field -> threshold in threshold units (see module docstring).  When
        omitted the projection step is a no-op and ``metadata['projection']``
        records that no cap was applied.
    cap_fraction : float
        Projection cap as a fraction of the threshold (default 0.9).
    min_style_samples : int
        Minimum number of training rows per style (default 2, because a stratified
        fold can leave the rarest style with only two training rows).
    """

    def __init__(self, features=None, components=8, defect_thresholds=None, cap_fraction=0.9,
                 min_style_samples=2):
        self.features = None if features is None else [str(f) for f in features]
        self.components = int(components)
        self.defect_thresholds = None if defect_thresholds is None else dict(defect_thresholds)
        self.cap_fraction = float(cap_fraction)
        self.min_style_samples = int(min_style_samples)

    # ---------------------------------------------------------------- fit
    def fit(self, x, styles):
        x = np.asarray(x, float)
        styles = np.asarray(styles).astype(str)
        if x.ndim != 2:
            raise ValueError("x must be a 2-D array")
        if len(x) != len(styles):
            raise ValueError("x and styles must have the same length")
        if len(x) < 3:
            raise ValueError("At least three training rows are required")
        if np.isnan(x).all(axis=0).any():
            raise ValueError("Every coordinate needs at least one finite observation")
        if self.components < 1:
            raise ValueError("components must be positive")

        self.n_features_ = x.shape[1]
        if self.features is not None and len(self.features) != self.n_features_:
            raise ValueError("features must describe all columns of x")
        self.styles_ = np.array(sorted(set(styles.tolist())))

        self.impute_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(x)
        filled = self.impute_.transform(x)
        self.x_ = filled
        self.styles_train_ = styles

        self.center_ = np.median(filled, axis=0)
        q75, q25 = np.percentile(filled, [75, 25], axis=0)
        scale = q75 - q25
        fallback = np.std(filled, axis=0)
        scale = np.where(scale > 1e-12, scale, np.where(fallback > 1e-12, fallback, 1.0))
        self.scale_ = scale
        z = (filled - self.center_) / self.scale_

        n_components = int(min(self.components, len(z) - 1, self.n_features_))
        self.n_components_ = n_components
        self.pca_ = PCA(n_components=n_components, svd_solver="full").fit(z)
        scores = self.pca_.transform(z)
        reconstruction = self.pca_.inverse_transform(scores)
        residual = z - reconstruction
        self.residual_variance_ = np.mean(residual ** 2, axis=0)

        self.style_params_ = {}
        for style in self.styles_:
            local = scores[styles == style]
            if len(local) < self.min_style_samples:
                raise ValueError("Style %r has fewer than min_style_samples rows" % (style,))
            mean_score = local.mean(axis=0)
            centered = local - mean_score
            covariance = centered.T @ centered / max(len(local) - 1, 1)
            covariance = covariance + np.eye(n_components) * 1e-6 * max(np.trace(covariance) / n_components, 1e-12)
            marginals = []
            for j in range(self.n_features_):
                column = filled[styles == style, j]
                column = column[np.isfinite(column)]
                marginals.append(np.sort(column))
            self.style_params_[style] = {
                "n": int(len(local)),
                "mean_score": mean_score,
                "covariance": covariance,
                "marginals": marginals,
            }
        self._interp = None
        self._caps = defect_caps(self.defect_thresholds, self.cap_fraction) if self.defect_thresholds else {}
        return self

    # ------------------------------------------------------------- sample
    def sample(self, style, n, seed, proposal=PC_RESIDUAL):
        if not hasattr(self, "style_params_"):
            raise ValueError("Call fit before sample")
        style = str(style)
        if style not in self.style_params_:
            raise ValueError("Unknown style %r" % (style,))
        if proposal not in PROPOSALS:
            raise ValueError("Unknown proposal %r" % (proposal,))
        if proposal == DONOR_RESIDUAL:
            raise ValueError("%r is implemented by DonorResidualGenerator" % (DONOR_RESIDUAL,))
        n = int(n)
        if n < 1:
            raise ValueError("Sample count must be positive")
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an int or None")
        rng = np.random.default_rng(seed)
        params = self.style_params_[style]

        if proposal == PC_RESIDUAL:
            scores = rng.multivariate_normal(params["mean_score"], params["covariance"], size=n,
                                             check_valid="ignore")
            z = scores @ self.pca_.components_ + rng.normal(
                scale=np.sqrt(self.residual_variance_), size=(n, self.n_features_))
        else:  # donor_quantile: interpolation proposal, marginals restored below
            generators = self._donor_generator()
            z = (generators.sample(style, n, "interpolate", seed)[0] - self.center_) / self.scale_

        coordinates = self.center_ + z * self.scale_
        for j in range(self.n_features_):
            coordinates[:, j] = _empirical_quantile_map(coordinates[:, j], params["marginals"][j])

        projected = {}
        for field, cap in self._caps.items():
            j = self._feature_index(field)
            if j is None:
                continue
            before = coordinates[:, j].copy()
            coordinates[:, j] = np.minimum(coordinates[:, j], cap)
            projected[field] = int(np.count_nonzero(before > cap))

        metadata = {
            "style": style,
            "n": n,
            "seed": None if seed is None else int(seed),
            "proposal": proposal,
            "method": METHOD_NAMES[proposal],
            "components": int(self.n_components_),
            "cap_fraction": self.cap_fraction,
            "projection": projected,
            "caps": {field: float(cap) for field, cap in self._caps.items()},
            "defect_physical_range": {},
        }
        for field, cap in self._caps.items():
            j = self._feature_index(field)
            if j is None:
                continue
            physical = physical_values(coordinates[:, j], field)
            metadata["defect_physical_range"][field] = {
                "min": float(physical.min()),
                "median": float(np.median(physical)),
                "max": float(physical.max()),
                "cap_physical": float(physical_values(cap, field)),
            }
        return coordinates, metadata

    # ------------------------------------------------------------ helpers
    def _feature_index(self, field):
        if self.features is None:
            return None
        try:
            return self.features.index(field)
        except ValueError:
            return None

    def _donor_generator(self):
        if self._interp is None:
            from beer_models.generation import StyleGenerator

            self._interp = StyleGenerator().fit(self.x_, self.styles_train_)
        return self._interp

    def protocol(self):
        """JSON-safe description of the fitted generator (for protocol.json)."""
        return {
            "component": "CopulaQuantileGenerator",
            "components": int(self.n_components_),
            "cap_fraction": self.cap_fraction,
            "defect_transforms": dict(DEFECT_TRANSFORMS),
            "defect_units": dict(DEFECT_UNITS),
            "defect_thresholds": None if self.defect_thresholds is None else dict(self.defect_thresholds),
            "caps": {field: float(cap) for field, cap in self._caps.items()},
            "styles": {style: int(params["n"]) for style, params in self.style_params_.items()},
            "method_names": dict(METHOD_NAMES),
        }


def _empirical_quantile_map_jittered(values, marginal, rng):
    """Like :func:`_empirical_quantile_map` but randomises the within-rank offset.

    Two draws that reuse the same donor row get different values in every column, so the
    generated spectra are distinct even when the donor pool of a rare style is tiny.  The
    random offset is uniform inside one rank step, so the marginal distribution is
    unchanged up to that finite-sample smoothing.
    """
    values = np.asarray(values, float)
    marginal = np.asarray(marginal, float)
    n, m = values.size, marginal.size
    if n < 1:
        raise ValueError("need at least one value")
    offset = rng.random(n)
    order = np.lexsort((offset, values))
    u = np.empty(n, float)
    u[order] = (np.arange(n) + offset[order]) / n
    grid = (np.arange(m) + 0.5) / m
    return np.interp(u, grid, marginal)


class DonorResidualGenerator(CopulaQuantileGenerator):
    """v2 style generator: whole donor residuals + support projection + defect projection.

    Why v2 exists
    -------------
    The v1 generator (:class:`CopulaQuantileGenerator`, proposal ``pc_residual``) samples
    the 8 principal scores from a per-style Gaussian and then adds **independent** noise on
    the remaining 216 coordinates.  That destroys the cross-coordinate residual correlation
    of the real spectra, and the failure shows up in the project's own diagnostic: the
    rank-correlation RMSE of the generated spectra against the training spectra is 0.39
    instead of the 0.15 the donor interpolation baseline achieves.  Resampling the
    *whole* residual vector of a same-style donor keeps that correlation structure intact.

    How v2 differs from v1
    ----------------------
    1. residual reuse (bootstrap): a generated spectrum starts from the standardised
       coordinate vector of an actual same-style training beer, i.e. the whole 224-D
       deviation from the style mean is reused instead of independent per-coordinate noise;
    2. empirical quantile mapping with a randomised within-rank offset, so the style
       marginal is still reproduced exactly, but repeated donors become distinct spectra;
    3. support projection: after the quantile map, any spectrum further than the style's
       own ``support_quantile`` leave-one-out nearest-neighbour radius is shrunk onto that
       radius toward its nearest training neighbour (this keeps generated spectra inside
       the observed support, which is what held-out coverage measures);
    4. the defect caps are applied **last** and are identical to v1 (``cap_fraction *
       threshold`` in coordinate space), so the projection can never undo a cap.

    This is still a statistical style reference: not a recipe, not a measured beer, and not
    a brewability or sensory-acceptance certificate.
    """

    def __init__(self, features=None, components=8, defect_thresholds=None, cap_fraction=0.9,
                 min_style_samples=2, support_quantile=0.95, support_projection=True,
                 jitter=True):
        super().__init__(features=features, components=components,
                         defect_thresholds=defect_thresholds, cap_fraction=cap_fraction,
                         min_style_samples=min_style_samples)
        self.support_quantile = float(support_quantile)
        self.support_projection = bool(support_projection)
        self.jitter = bool(jitter)

    # ---------------------------------------------------------------- fit
    def fit(self, x, styles):
        super().fit(x, styles)
        z = (self.x_ - self.center_) / self.scale_
        self.z_ = z
        self.scores_ = self.pca_.transform(z)
        # Support radius is measured in a robust per-channel standardized space whose scale
        # never falls back to the raw unit (the median/IQR scale used by the copula collapses
        # on nearly constant channels inside a 10-beer style, which made the leave-one-out
        # p95 radius meaningless: 6322 for Kriek versus 1.9 for Lambic).  Here the scale is
        # the channel standard deviation with a floor at 1% of the median channel standard
        # deviation, the same convention as the quality diagnostics.
        standard_deviation = self.x_.std(axis=0)
        positive = standard_deviation[standard_deviation > 0]
        self.support_center_ = np.median(self.x_, axis=0)
        if positive.size:
            self.support_scale_ = np.maximum(standard_deviation, 0.01 * float(np.median(positive)))
        else:
            self.support_scale_ = np.where(self.scale_ > 1e-12, self.scale_, 1.0)
        support_z = (self.x_ - self.support_center_) / self.support_scale_
        self.support_z_ = support_z
        self.donor_pools_ = {}
        for style in self.styles_:
            index = np.flatnonzero(self.styles_train_ == style)
            local = support_z[index]
            distance = _pairwise(local, local) / np.sqrt(self.n_features_)
            np.fill_diagonal(distance, np.inf)
            nearest = distance.min(axis=1)
            self.donor_pools_[style] = {
                "index": index,
                "mean_score": self.scores_[index].mean(axis=0),
                "radius": float(np.quantile(nearest, self.support_quantile)),
            }
        return self

    # ------------------------------------------------------------- sample
    def sample(self, style, n, seed, proposal=DONOR_RESIDUAL):
        if not hasattr(self, "donor_pools_"):
            raise ValueError("Call fit before sample")
        style = str(style)
        if style not in self.donor_pools_:
            raise ValueError("Unknown style %r" % (style,))
        if proposal not in (DONOR_RESIDUAL, PC_RESIDUAL):
            raise ValueError("DonorResidualGenerator supports %r and %r"
                             % (DONOR_RESIDUAL, PC_RESIDUAL))
        n = int(n)
        if n < 1:
            raise ValueError("Sample count must be positive")
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an int or None")
        if proposal == PC_RESIDUAL:
            return super().sample(style, n, seed, proposal=PC_RESIDUAL)

        rng = np.random.default_rng(seed)
        params = self.style_params_[style]
        pool = self.donor_pools_[style]
        donor_rows = pool["index"][rng.integers(0, len(pool["index"]), size=n)]
        z = self.z_[donor_rows]                       # whole residual vector reused
        coordinates = self.center_ + z * self.scale_

        for j in range(self.n_features_):
            column = coordinates[:, j]
            coordinates[:, j] = (_empirical_quantile_map_jittered(column, params["marginals"][j], rng)
                                 if self.jitter else
                                 _empirical_quantile_map(column, params["marginals"][j]))

        shrunk = 0
        if self.support_projection:
            local = self.support_z_[pool["index"]]
            projected_z = (coordinates - self.support_center_) / self.support_scale_
            distance = _pairwise(projected_z, local) / np.sqrt(self.n_features_)
            nearest = distance.argmin(axis=1)
            nearest_distance = distance[np.arange(n), nearest]
            outside = nearest_distance > pool["radius"]
            if outside.any():
                anchor = local[nearest[outside]]
                factor = (pool["radius"] / nearest_distance[outside])[:, None]
                projected_z[outside] = anchor + (projected_z[outside] - anchor) * factor
                coordinates = self.support_center_ + projected_z * self.support_scale_
                shrunk = int(np.count_nonzero(outside))

        projected = {}
        for field, cap in self._caps.items():
            j = self._feature_index(field)
            if j is None:
                continue
            before = coordinates[:, j].copy()
            coordinates[:, j] = np.minimum(coordinates[:, j], cap)
            projected[field] = int(np.count_nonzero(before > cap))

        metadata = {
            "style": style,
            "n": n,
            "seed": None if seed is None else int(seed),
            "proposal": proposal,
            "method": METHOD_NAMES[proposal],
            "components": int(self.n_components_),
            "cap_fraction": self.cap_fraction,
            "projection": projected,
            "caps": {field: float(cap) for field, cap in self._caps.items()},
            "support_quantile": self.support_quantile,
            "support_radius": float(pool["radius"]),
            "n_support_projected": shrunk,
            "jitter": bool(self.jitter),
            "donor_pool_size": int(len(pool["index"])),
            "defect_physical_range": {},
        }
        for field, cap in self._caps.items():
            j = self._feature_index(field)
            if j is None:
                continue
            physical = physical_values(coordinates[:, j], field)
            metadata["defect_physical_range"][field] = {
                "min": float(physical.min()),
                "median": float(np.median(physical)),
                "max": float(physical.max()),
                "cap_physical": float(physical_values(cap, field)),
            }
        return coordinates, metadata

    # ------------------------------------------------------------ helpers
    def protocol(self):
        described = super().protocol()
        described.update({
            "component": "DonorResidualGenerator",
            "proposal": DONOR_RESIDUAL,
            "method": METHOD_NAMES[DONOR_RESIDUAL],
            "support_quantile": self.support_quantile,
            "support_projection": self.support_projection,
            "jitter": self.jitter,
            "support_radius": {style: float(pool["radius"])
                               for style, pool in self.donor_pools_.items()},
            "donor_pool_size": {style: int(len(pool["index"]))
                                for style, pool in self.donor_pools_.items()},
        })
        return described
