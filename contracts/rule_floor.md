# Contract: rule_floor

## Purpose
Evaluates the deterministic statutory governance flags and returns the safety
floor for GovScore:

    RuleFloor = min(100, sum_k w_k * 1{F_k})

The floor exists so that an optimistic model output can never suppress an
established statutory breach (the Non-Masking Invariant enforced in
`paimana_engine`). This module is pure, deterministic and never touches the ML
model.

## Inputs
- `project`: a validated `ProjectInput`.
- `custom_weights`: optional per-flag weight override.

## Outputs
- `RuleFloorEvaluation`: `rule_floor` in [0,100], `active_signals_count`,
  one `RuleActivationSignal` per flag (active or not), `critical_override`,
  `summary`.

## Flags

| flag | weight | condition | statutory basis |
|---|---|---|---|
| F1 | 30 | `expenditure/original_cost - progress/100 >= 0.25` **and** `progress < 50` | GFR 2017 Rule 159 / CVC — unearned advances |
| F2 | 25 | `progress_change_recent <= 0.1` **and** `elapsed_ratio >= 0.20` **and** `progress < 95` | chronic site stagnation |
| F3 | 25 | `overdue/total_milestones >= 0.50` **or** `clearance_pending_days > 180` | critical-path / clearance breakdown |
| F4 | 20 | `days_since_last_update > 60` | MoSPI IPMD monthly reporting mandate |
| F5 | 30 | `dispute_status in {ARBITRATION, HIGH_COURT_STAY, TERMINATION_NOTICE}` | Arbitration Act / court stay |
| F6 | 25 | `original_duration_months + current_delay_months - months_elapsed <= 0` **and** `progress < 95` | declared completion date elapsed while work is incomplete |

Critical escalation (`critical_override = True`) on: F1 gap ≥ 40pp,
F3 clearance > 365 days, F5 active, **or F6 overdue by ≥ 12 months**.

## Why F6 exists (measured, not assumed)
F6 was added after measuring the shipped rule set against the real panel
(8,838 labelled transitions from 1,631 MoSPI projects). Two findings drove it:

1. **The existing flags are not early-warning signals for schedule slip.**
   Scored as a predictor of "revised completion date moves at the next report",
   F1+F2 achieve **AUC 0.4187** — worse than random, and it degrades further at
   longer horizons (0.3650 at 6 months). They fire on early-stage projects with
   distant deadlines, which are *less* likely to be revised imminently. This is
   not a claim that the rules are wrong: they identify chronic governance
   breaches, which is a different and legitimate question. It is a claim that
   they were never a forecast, and the platform should stop implying they are.

2. **The single strongest simple signal was not in the rule set at all.**
   "The declared completion date has already passed" covers 32.1% of
   project-months and carries a **4.3x lift** (46.7% slip rate when true vs
   10.9% when false), **AUC 0.725** on its own. Slip rate by remaining time:

   | months to declared date | rows | slip rate |
   |---|---|---|
   | ≥ 12 months overdue | 387 | 0.584 |
   | 6–12 months overdue | 467 | 0.355 |
   | 0–3 months overdue | 1,437 | 0.542 |
   | 3–6 months remaining | 1,006 | 0.108 |
   | > 24 months remaining | 728 | 0.022 |

### CORRECTION (2026-09-18): this evidence was inflated by the reconstructed axis
The figures above were measured on a panel whose time axis was *reconstructed*.
On the observed axis (`scripts/harvest_monthly.py`) they fall sharply:

| "Declared date has passed" | reconstructed | **observed** |
|---|---|---|
| coverage | 32.1% | **8.3%** |
| lift | 4.3x | **1.38x** |
| AUC alone | 0.725 | **0.5170** |

The reconstruction placed projects further past their deadlines than they
really were. **F6 is deliberately left unchanged**: the elapsed date is still a
true statutory fact, and retuning a rule to chase a freshly-measured
correlation is the error this contract refuses for F1. Only the claim changed.

The corrected data shows risk peaks *before* the deadline — the 0-3 months
remaining band slips at 0.4655 vs 0.0128 for >24 months out. An imminent-
deadline flag (**candidate F7**) is therefore recorded here as a proposal
**requiring human review**, not adopted unilaterally.

Weight 25 places F6 alongside F2/F3 rather than at the F1/F5 maximum: a passed
deadline is a strong statutory signal but, unlike litigation, is partly a
reporting artifact of agencies that revise late.

## F1's premise is inverted for schedule forecasting (measured)
`scripts/label_validity.py` tested rule F1's own premise — that disbursement
running ahead of physical progress predicts trouble — directly against the
outcome:

| Spend-vs-progress band | project-months | slip rate |
|---|---|---|
| Spend ≥ 25pp ahead of progress (**F1 fires**) | 1,578 | **0.0716** |
| Roughly aligned | 6,324 | 0.1420 |
| Progress ≥ 25pp ahead of spend (**opposite of F1**) | 3,629 | **0.2904** |

The condition F1 flags slips at roughly **one quarter** the rate of the exact
opposite condition — a 4.1x inversion. This is the mechanical explanation for the rule
set scoring AUC 0.433 against schedule slippage.

**F1 is nevertheless retained unchanged, and must not be "fixed" by flipping
it.** It encodes a real statutory test under GFR 2017 Rule 159 and CVC
guidance: unearned contractor advances are a financial irregularity worth
flagging on their own terms, and that is a different question from whether the
completion date is about to move. Inverting a statutory rule to chase a
correlation would be the exact confusion this contract exists to prevent.

What changes is the *claim*: F1 is a financial-irregularity flag, never a
schedule predictor, and no interface may present it as one.

## Behavior cases (input → expected output)
| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 1 | duration 36, elapsed 40, delay 0, progress 60 | F6 active, weight 25 | Declared date passed 4 months ago |
| 2 | duration 36, elapsed 40, delay 12, progress 60 | F6 inactive | Delay pushed the declared date to month 48; not yet reached |
| 3 | duration 36, elapsed 36, delay 0, progress 60 | F6 active | Boundary: exactly at the declared date counts as elapsed |
| 4 | duration 36, elapsed 40, delay 0, progress 97 | F6 inactive | Work essentially complete; a trailing date is administrative |
| 5 | duration 36, elapsed 60, delay 0, progress 20 | F6 active, `critical_override = True` | 24 months overdue ≥ 12-month escalation |
| 6 | no flags at all | `rule_floor = 0.0`, `active_signals_count = 0` | Clean project |
| 7 | F1+F2+F3+F5+F6 all active | `rule_floor = min(100, 135) = 100` | Clamp holds |
| 8 | F6 alone | `rule_floor = 25.0` | Weight applied exactly once |

## Edge cases that must be covered
- Adding F6 must not change the floor for any project where F6 is inactive —
  regression risk against the frozen golden cases.
- `original_duration_months` at its minimum (> 0) with huge `months_elapsed` —
  no overflow, F6 simply active.
- `progress = 95.0` exactly — F6 inactive (strict `< 95`), matching F2.
- Every flag must appear in `signals` whether or not it is active, so the
  dashboard can render the full statutory checklist.

## Known limitation (must stay documented)
F3, F4 and F5 are **structurally inert on the public PAIMANA feed**:
`clearance_pending_days`, `days_since_last_cuf_update` and dispute status are
not published by the portal (`Remarks` is null on all 14,917 records, as is
`StateName`). On the live 2,155-project portfolio those three flags fire on
zero projects. They are retained because they are correct rules that work on
the full internal CUF, and their inertness is the concrete argument for the
CUF 2.0 schema proposal — but the platform must never present them as active
monitoring on public data.

## Explicitly out of scope
- Combining the floor with the model — `paimana_engine` owns
  `GovScore = max(100 * P_model, RuleFloor)`.
- Capital-at-Risk — `paimana_car` owns it.

## Status
- [x] Drafted
- [ ] Reviewed by a human
- [x] Implementation matches this contract
- [x] Golden tests exist for every behavior case above
