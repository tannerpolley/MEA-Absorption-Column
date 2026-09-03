# Issue 30: nonlinear reactive-film validation gate

Status: **supported-negative no-run decision**. The accepted Issue 36 result retains zero admitted physical film-input states. Issue 30 therefore records the predeclared validation cases as not attempted; it launches no physical BVP campaign and retains no physical film values.

## Case-class result

All seven predeclared case classes remain visible. `attempt_status=not_attempted` means dependency-blocked or incomplete evidence, not model disagreement. `model_disagreement_status=not_evaluated` for every row. A timeout or failed physical state cannot be reported because no physical state was launched.

| Case class | Role | Attempt | Reason type | Non-execution reason |
| --- | --- | --- | --- | --- |
| `zero_drive` | `limiting_case` | `not_attempted` | `dependency_blocked` | Issue 36 admits zero physical film-input states; the zero-drive calculation cannot be evaluated without an admitted packet-bound state. |
| `no_reaction` | `limiting_case` | `not_attempted` | `dependency_blocked` | Issue 36 admits zero physical film-input states; the no-reaction comparison cannot be evaluated. |
| `linear_limit` | `limiting_case` | `not_attempted` | `dependency_blocked` | Issue 36 admits zero physical film-input states; the linear-limit comparison cannot be evaluated. |
| `desorption` | `limiting_case` | `not_attempted` | `dependency_blocked` | Issue 36 admits zero physical film-input states; no reverse-driving physical state may be invented. |
| `source_rate_states` | `source_comparison` | `not_attempted` | `incomplete_evidence` | The accepted Issue 36 result has no admitted concentration, kinetic, or transport intersection for a film-rate evaluation. |
| `independent_rate_observations` | `independent_physical_validation` | `not_attempted` | `incomplete_evidence` | Zero admitted Issue 36 film-input states means no independent observed-versus-calculated film rate can be compared. |
| `case3c_application_stress_states` | `application_stress` | `not_attempted` | `dependency_blocked` | Issue 36 admits zero physical film-input states; the 21-state application stress set cannot be launched. |

The limiting, desorption, initialization/order, source-rate, independent-observation, and 21-state Case 3C checks remain unrun. No flux, residual, branch, mesh, timing, uncertainty, or observation-comparison value is invented.

## Decision boundary

The accepted Issue 36 release result has 5 declared states, 0 admitted states, and no version-2 input, schema, or receipt files. Issue 19 remains open and has no recorded investigator-approved timing campaign budget. The Issue 30 outcomes **Physical film admitted** and **Model revision required** are both `not_reached`; neither no-run evidence nor a dependency blocker is a physical disagreement.

The selected issue-level result is **supported negative, no run**. It blocks column replacement and downstream Issue 31 execution. No thermodynamic, kinetic, transport, film, area, transfer, or capture quantity is fitted or changed.

## Source/result lineage

Repository `tannerpolley/MEA-Absorption-Column`, base revision `fc6fd8369ec4567694eca389c5937ce1159577b9`, source revision `2d74e2c1d4514d44996dfe1fed68b6159b2b57b0`; input SHA-256 `155bba0b34012cb25bf2bf37d8bb1d44c44263d7f5d488b7a48f6c3a0ab039ee`; generator SHA-256 `87b96ec1a466a4167ff0398b4d521bca961c60514dcbf4d5d721933ee955d979`; exact command `uv run python analyses/nccc_validation/scripts/resolve_issue30_film_validation_gate.py`; machine `Linux-7.0.0-30-generic-x86_64-with-glibc2.39`; workers `1`; run identity `issue30_gate_2d74e2c1d451`. The source revision contains none of the three generated Issue 30 files.

Issue 36 source revision `925acf52b77f31131dbb98533b80a14f2ca4768b` and accepted result commit `94503f71ab41a8841fd2aebb37ece3dec9a07cb1` are bound by the exact Issue 36 input, table, summary, report, bundle, Work Package A owner, release guard, and dependency hashes retained in the generated summary. The Issue 36 bundle outer SHA-256 is `4139fecd9b5192e7cadd12883d2ff1bff71c20d74950af5256e4f0447995f27b`, the extracted wheel is `d7b4fc5ba5cbf0e979b65af83442d565496d11b771bb559233ad9dc3a4f8414a`, the parameter document is `2666914f0f9cfebdf230e96565de843f9aadc9424035c940883147ff66af035c`, the state packet is `41017bcf727a486a8f3feb280e19c111a15c5dda5a3cca4e8c7dc5b051168fef`, and the chemistry member is `1989f3e6c8fa567a019dcdbceb4bbcf26d9ca48aec3f640dad1134bdd1fd4e7c`.

## Claim boundary

This record supports only the Issue 30 no-run decision. It does not validate limiting behavior, desorption, numerical convergence, initialization, rate observations, film flux, packed-column capture, or a model disagreement.

Regenerate with:

```text
uv run python analyses/nccc_validation/scripts/resolve_issue30_film_validation_gate.py
```
