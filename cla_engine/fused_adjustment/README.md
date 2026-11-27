# Frozen fused-Y adjustment

These files are a byte-for-byte snapshot of `shared_core_adjustment` from the
149 functional-Y source at `/dsk2/user/dengsc/cla_functional_y_20261109_r1/`.
That package generated the TCGA/CPTAC fused relation labels under contract
`CLA_SHARED_CORE_DNA_PURITY_RELATION_DEMO_V1`. Source-patient features must
use the same target-aware covariate policy; the generic R62 model adjustment
contract is a different estimand.

SHA256 of the source and copied files on 149:

- `adjustment_policy.py`: `f6c06eac1e6cdd1986e36c1558c1aa550a2f996ea27d549a78759e7bc81681d5`
- `covariate_contract.py`: `97c63672ac2125765f4c5cc5445764ebbd9a0a29c0e3d8b4ceb060f3e8c23f9d`
- `design.py`: `2a146a6819c17ac6ff5b63107ea2b15c700712fbec0f7abf8ba6291b3e939b57`

Changing this snapshot requires regenerating source features and verifying
their adjustment against the frozen label definition.
