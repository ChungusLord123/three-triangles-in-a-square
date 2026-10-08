# Three unit equilateral triangles in a square

Source code, support tables and saved verification records for a computer-assisted
optimality proof candidate. The claimed minimum square side is

`sqrt(3)/2 + sqrt(6)/4 = 1.47839784...`

The triangles have side one, disjoint interiors, and permitted boundary contact.
The [two-page argument](docs/two-page-argument.pdf) separates the exact feasible
construction and boundary-family lower bound from the global computational
coverage and exclusion claim. External mathematical review is pending.

## Included material

- `source/triangle_square_research/`: search, interval arithmetic, contact
  enumeration, separately implemented GPU verifiers and their support arrays.
- `proof_n3/independent_verification/`: immutable recorded certificate,
  inference-check summaries, coverage audits and detailed mathematical review.
- `proof_n3/original_candidate/`: preserved earlier candidate and coverage notes.
- `local_archival_metadata/`: selected model/controller snapshots and ledger
  metadata, with historical absolute paths preserved for provenance.
- `saved_replay_receipts/`: additional recorded replay receipts.
- `review_checks.py`: portable entry point for lightweight GPU checks.
- `MANIFEST.json` and `SHA256SUMS`: byte commitments for the committed files.

The separately written implementations were developed in the same project.
They do not constitute external peer review or formal proof-assistant verification.
Acceptance of the global claim requires the geometric reductions, complete
profile/domain coverage, and correspondence between the argument and transcript
to be sound.

## Run lightweight checks

Use macOS on Apple silicon with MLX Metal, PyTorch MPS and NumPy. The current
packaging environment is recorded in `runtime.json`; those versions were not a
separate historical lock of every full proof run. From the repository root:

```sh
python3 -m pip install -r requirements.txt
python3 review_checks.py
```

The entry point requires both GPU backends and enables no CPU numerical fallback.
Numeric geometry and exact field algebra run on GPU. The CPU handles orchestration,
metadata, file bytes and hashes.

These checks retain the exact construction/singular fixture, reject invalid
containment and interior overlap fixtures, verify strict-motion calibration,
and check 85 exact field identities on both implementations. They are sanity
checks, not full numerical-certificate replay. Fresh output is ignored under
`review_output/` and the source tree's `proof_n3/` directory, so the historical
verification reports remain unchanged.

An optional finite contact-membership coverage audit is available:

```sh
python3 review_checks.py --coverage
```

The wrapper remaps historical support-array paths to the included local copies.
The actual packing position and rotation domains remain continuous.

## Full replay requires separate archival data

This repository is the compact review source distribution. It excludes large
historical frontier arrays, intermediate inference arrays and most external-volume
generations. Full replay additionally requires those records, their matching
model/controller versions, and adaptation of historical absolute paths. The
dedicated archival data on WD_Game are not included here.

Recorded counts and passed flags do not substitute for those missing data. The
lightweight check must not be interpreted as reproducing the global proof.

## Integrity and review priorities

Source and original numerical evidence are copied byte for byte. Hashes in old
receipts may refer to omitted archival files or their originating machine paths.
`MANIFEST.json` records this distribution separately from the historical certificate.

Review should scrutinize full proper contact profiles; contact/symmetry/angle
classification; component translations and rotations; interval contractors;
strict improvement certificates; boundary and singular cases; covering
subdivision and precision transitions; and the exact family elimination.

The original problem is listed on [Erich Friedman's Triangles in Squares page](https://erich-friedman.github.io/packing/triinsqu/).
