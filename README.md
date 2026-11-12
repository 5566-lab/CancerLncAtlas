# CancerLncAtlas

CancerLncAtlas studies **cancer × lncRNA × exact pathway**. Biological pathway
identities and patient molecular observations are fixed before modeling lncRNA
contributions. Pathway and State retain total task weights of **1.0 and 0.5**.

## KEGG cancer mechanisms and cross-database integration

The evidence-defined backbone contains **477 KEGG NETWORK mechanisms** from
658 source definitions: 199 in official cancer categories, 23 with additional
malignant-disease annotations and 255 supported by mechanism literature.
These 477 are an evidence-curated cancer scope, not a single official KEGG category.
Original NETWORK IDs, conditions and member lists remain the backbone.

Cross-database integration uses explicit functional equivalence, hierarchy,
protein composition and regulator membership. Protein lists are mapped to fixed
identifiers and clustered by complete linkage on `1 − Jaccard`, with Jaccard
at least 0.5 within each composition cluster. A shared cluster supplies a
composition relationship; functional scope and mechanism conditions determine
whether identities can merge. Identical lists can describe different mutations.
The catalog retains 1,652 context identities, 197 State identities and the full
source protein lists for graph construction.

TF, kinase and PROGENy source readouts belong to **Pathway**. TF and kinase
programs describe regulatory or member-state evidence; an independent auxiliary
loss requires an explicit task configuration. Classification and source count
do not determine loss weights.

## Y: patient observations grouped by exact pathway

For cancer `c`, patient `i`, pathway `k` and unique observation `a`:

```text
Y[c,i,k] = {(a, value[c,i,a], observed[c,i,a], role[k,a]) : a in A[k]}
```

Each observation has a platform, measurement definition, unit and source.
Protein-coding RNA, protein abundance, phosphosites, CNV, promoter methylation
and defined pathway/State readouts retain their individual dimensions. A shared
measurement is stored once and linked to its pathways through a membership table.
TCPA antibody signals and CPTAC mass-spectrometry measurements retain distinct
observation IDs. Missing measurements remain missing.

Patient condition evidence is separate from member measurements. Reference
mechanisms require no additional event. Variant, infection or exposure mechanisms
require matching evidence for condition-specific interpretation; an unknown
condition is not a negative result.

RNA scores select protein-coding gene IDs **before** symbol aggregation. Changed
backgrounds have distinct method versions. The coding TLS12 score uses the 12
chemokines in the GDC PanImmune definition and unnormalized ssGSEA on the fixed
coding background. Published TLS12 scores and ESTIMATE-derived purity remain
source references; the latter is excluded from default Y until its original
measurement dependencies can be established.

See [Y definitions and measurement interfaces](docs/MEASURED_Y_REPRESENTATION.rst),
[biological integration](docs/CANCER_RELATION_TASK.rst),
[member-state mappings](docs/MEMBER_STATE_EVIDENCE.rst) and
[RNA scoring definitions](docs/MEASURED_READOUT_DEFINITIONS.rst).

## Read and export fixed Y

All project preparation and checks execute on host **149**. The three-cancer
development registry covers BRCA, LUAD and COAD; outer-fold-0 patients remain
sealed. Raw data and patient identifiers stay outside Git.

```python
from cla_engine.measured_y_registry import load_target
values, observed, applicability, definitions = load_target(
    "/path/to/observation_registry_r2/BRCA", canonical_id, "TCGA"
)
```

```bash
python scripts/export_fixed_y.py \
  --registry /path/to/observation_registry_r2/BRCA \
  --target CLA-P-922863C8FB5E9DF9 --cohort TCGA \
  --output /path/to/new_export
```

The exporter verifies immutable input hashes and preserves original values,
observed masks and condition applicability. It performs no fitting or association
estimation. Install the dependencies in `requirements/` in an isolated environment.

## Linear M0 comparator

Within each cancer, one output corresponds to a unique patient observation:

```text
predicted_Y[i,a] = intercept[a] + C[i] gamma[a]
                  + X_lnc[i] beta_lnc[a] + X_mRNA[i] beta_mRNA[a]
```

Compare clinical/purity only, lncRNA only, mRNA only and their combination.
Clinical and independent-purity covariates are unpenalized; RNA coefficients
use Lasso selected inside training patients. There is no correlation/FDR admission.
All comparisons use identical observed outcomes, patients and folds. A measurement
shared by pathways is fitted once. Zero-RNA fits are reported separately as RNA
prediction failures. For graph-regularized Ridge, include ordinary Ridge on the
same inputs.

Non-RNA outcomes exclude their own measurement layer and direct derivatives from
predictors. RNA scores predicted from their component mRNAs are score reconstruction.
CCHHGT comparisons hold Y and available input layers fixed while changing graph
propagation or removing a specified edge/input layer. Per-lncRNA evidence is
evaluated after fitting; it does not generate Y.

See the [full M0 specification](docs/M0_RELATION_BASELINE.rst). Historical
experiments and their original outcome definitions are recorded in `provenance/`;
they are not results for this fixed observation registry.
