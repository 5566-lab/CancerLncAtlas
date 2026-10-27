Single-cell frozen geometry reconstruction
=========================================

The original outer and nested metacell producers now persist the actual
``LocalSpace`` projection in ``LOCAL_SPACE.npz``. The geometry audit marks it as
``NATIVE_SC_FROZEN_GEOMETRY_REPLAY_V1``. Coordinates and grouping are reconstructed
read-only during native completion/cache checks and actual model-use inspection.
No new fitting is dispatched by those checks.

Original preprocessing and estimator
------------------------------------
``prepare_local_space`` factors the original preparation out of ``LocalSpace.fit``
without changing its equations or order. Training cells are selected by the
declared source donors. Per-cell weights give each donor equal total weight.
Coding genes with variance above 1e-8 are selected by stable descending variance,
up to the original maximum. A seeded per-donor cell cap produces the PCA input
rows. Standardized values are clipped to [-10, 10] and donor-weighted centered.
The original randomized, unwhitened PCA remains the fitting implementation.

The replay recomputes this preparation from actual measured coding-log matrices
and canonical donor metadata. It compares selected gene positions, means,
scales and weighted center, then verifies the saved PCA input mean and component
shape/orthonormality. A frozen sklearn PCA object applies the saved projection
through the original ``LocalSpace.transform``. It never invokes PCA.fit.

The original ``local_disjoint_groups`` computes the source-only neighborhood
radius and disjoint within-donor/type/state groups from those coordinates.
NearestNeighbors builds an index for the original geometric calculation; no
PCA, LASSO or neural parameters are fitted. The resulting coordinates, exact
cell assignments and every original radius/context audit field must match.
Small or isolated cells retain assignment -1. The original query exclusion and
nested source/target/validation donor policies remain in force.

Persisted fields and integration
-------------------------------
``genes`` contains coding-matrix column positions, not external gene IDs.
``mean``, ``scale`` and ``center_z`` hold the selected-gene preprocessing arrays.
``pca_mean``, ``pca_components`` and ``pca_explained_variance`` preserve the frozen
unwhitened estimator. Explained variance is retained for the actual sklearn
transform interface; whitening remains disabled and these values are not
independently treated as verified fitted eigenvalues.
``training_donors`` is the sorted source-donor tuple. ``coordinates`` contains
the native transformed cell rows in their actual input order. Outer coordinates
include all canonical cells; nested coordinates include only the original
source and inner-target retained cells. Parent native input inventories bind
the measured matrices, cell order and node ID order.

Both native serializers include the parameter archive in their existing hashed
output inventories. Replay/cache signatures include the archive, original raw
inputs and the replay implementation. A marker without parameters, parameters
without a marker, changed source arrays, altered normalization/projection,
coordinate substitutions and changed group/radius outputs fail verification.
The native graph and episode inventories propagate the archive into actual
model-use dependencies and register the successful geometric replay.

Limitations and no-fitting verification
--------------------------------------
This checks application of supplied parameters, not the historical act of
fitting the PCA basis or its optimality. ``SC_FITTED_GEOMETRY_NOT_INDEPENDENTLY_VERIFIED``
therefore remains blocking, as do LASSO, acquisition/QC/reference and remaining
cross-module ancestry gaps. Legacy external grouping without original
``LocalSpace`` parameters still has no geometric proof and produces no geometric
replay record. It is not silently promoted to a complete source history.

Tests supply an explicitly frozen identity projection and run original
preprocessing, transformation, radius/grouping, statistics and downstream
consumers on synthetic data. PCA fitting, LASSO fitting and neural optimizers
are prohibited. Existing frozen grouped fixtures remain explicitly separate
from the new actual geometric calculation. No real patient data, model training,
resource acquisition or deployment is part of these checks.
