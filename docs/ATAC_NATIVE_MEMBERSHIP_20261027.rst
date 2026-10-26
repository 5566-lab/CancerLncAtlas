Native GMT / GENCODE membership in the original ATAC workflow
===========================================================

The existing ``run_v32_atac_materialization.py`` CLI accepts the
``pathway_membership.parquet`` output of ``pathway_membership.run`` without
an identifier conversion file. Its input contract must contain the actual
``candidate_definition`` binding. Candidates, membership and expression
activities can therefore use the same pinned GENCODE and named GMT collection
content. The materializer rejects a different annotation or collection before
creating output or launching R. The command still runs only on host ``149``.

The existing materializer CLI arguments and patient authority behavior remain
unchanged. Use ``--pathway-membership`` for the native output and include that
file's exact path/hash/size/basename under ``pathway_membership`` in the
``CC_HHGT_V3_2_ATAC_AUTHORITATIVE_INPUT_CONTRACT_V1`` contract. Its
``candidate_definition`` points to the actual ``CANDIDATE_SOURCES.json``.
Unmarked legacy membership remains supported by its original contract.
Marked native membership without an explicit candidate definition is rejected;
it cannot use the legacy authority fallback to skip resource identity checks.

Both ``LAUNCH_PREFLIGHT.json`` and ``GENE_ACCESSIBILITY_SUCCESS.json`` include
``membership_reference`` when a native membership table is consumed. This
contains the original membership binding, actual candidate annotation/GMT
pins, reconstruction counts, the ATAC identifier/projection policies and
current consumer implementation hashes. The materializer checks it again
before success. Trainer preparation and completed-result/fusion validation
reconstruct and compare the same proof. A removed or changed receipt is not
silently upgraded to a verified native membership. A previous materialization
without this native proof must be regenerated if it consumed native membership.

Identifier and numeric semantics
-------------------------------

The original ATAC identifier normalization removes ``LNC:``, ``LNCRNA:`` or
``GENE:`` prefixes and the first dot plus following suffix. Native GENCODE
membership already uses stable IDs: ``GENE:ENSG..._PAR_Y`` becomes
``ENSG..._PAR_Y`` and remains distinct from the other copy. The reserved
``UNMAPPED_GMT:<hash>`` symbol is preserved in the required gene list and source
membership. It cannot acquire measured accessibility without a real matching
promoter and peak overlap. No synthetic promoter or zero row is added for it.

The original R materializer emits only genes with promoter/peak overlaps.
Original technical-replicate and multiple-aliquot patient averaging is retained.
The ATAC pathway projection averages observed member genes after its existing
normalization, requiring at least ``min_pathway_promoter_genes`` observed
genes. It does not require every GMT member to be observed. This differs from
the CNV event-or-all-members rule; CNV callability is not imported into ATAC.
Source membership retains unknown and multiply mapped members, while the
observed pathway projection uses unique available gene IDs. Its denominator
is the number of observed member genes, not all source symbols or all mapped
members. The structural unavailable reason remains
``ATAC_EXACT_PATHWAY_PROMOTER_COVERAGE_LT_MINIMUM`` below the threshold.

The fixture explicitly tests three measured member genes, one mapped PAR_Y
member without a promoter measurement and one unmatched GMT symbol. The
required gene list retains all five, the actual R output has the three
measured genes, the original pathway weights are 1/3, and a minimum of four
returns unavailable. Every original fold feature collector runs without
fitting. The raw-gap cancer remains unavailable.

Handoff and evidence boundaries
-------------------------------

The output is consumed by existing ``auxiliary.prepare`` with
``producer_kind=atac`` and its original ``materialization_success`` and
``pathway_membership`` references. The separate fitting command is not
dispatched. Explicitly frozen completed-result fixtures exercise
``auxiliary.collect``, current fusion preparation and its actual expert source
inspector. That inspector follows membership back to raw GMT/GTF and records
the mapping reconstruction. It retains reference acquisition/authenticity
gaps and the broader raw study/patient-use gap. No biological study identity
is inferred from a reference resource filename.

This change does not supply a new promoter annotation producer, rederive
original raw ATAC normalization, complete all aliquot/patient ancestry or
attest source-resource authenticity. It adds no chain operation: the registry
still has 33 operations. Final guarded tests use synthetic full-shape ATAC
input, original R and Python statistics, and frozen result containers.

Verification execution correction
---------------------------------

The first regression selection mistakenly included the old
``test_fresh_atac_training_preserves_scope_and_typed_gaps`` test. It executed
five synthetic logistic fits, each with 12 NumPy gradient iterations, before
that regression process was interrupted. This violated the code-only execution
scope. Its original receipt, corrected receipt and actual checkpoint evidence
are preserved in ``atac_membership_fit_scope_deviation.json`` and related
files; the interrupted run is excluded from successful code-only validation.
No real cohort or paid compute was used, and none of those synthetic results
is used as a scientific result.

Subsequent verification uses ``CANCERLNCATLAS_CODE_ONLY_ATAC_CHECKS=1`` with a
session fixture that blocks the ATAC, CNV, genomic and fusion fitting functions
and Adam/AdamW/SGD constructors before parameter updates. It also selects only
the two non-fitting cases from the old ATAC training test module. A dedicated
test exercises each blocked entry. This guard applies to module fixtures as
well as test functions; ordinary test execution outside the opt-in launcher
retains its original behavior. Final successful verification contains no
parameter fitting, real-cohort processing, download or deployment.
