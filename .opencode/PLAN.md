# PLAN.md

## Task

- Request:
- Owner:
- Date:

## Goal

- Primary outcome:
- Non-goals:

## Strategy Track (Required Order)

1. Baseline algorithm
2. Parametric study
3. ML/DL

Comparison note:

- [ ] All three tracks compared in one summary table
- [ ] Profitable candidate selected for ideation (if any)

## Scope

- In scope files/modules:
- Out of scope:

## Data and Infra Assumptions

- Data root: `/mnt/data/finance`
- Time-series format: hive partition
- Compute plan:
   - [ ] GPU usage considered
   - [ ] Parallel processing plan defined
- Engine plan:
   - [ ] Polars default
   - [ ] DuckDB usage rationale documented (if used)

## Milestones

1. Context scan
   - [ ] Read affected modules
   - [ ] Confirm data/paths and dependencies
2. Design
   - [ ] Define minimal change set
   - [ ] Define verification strategy
3. Implementation
   - [ ] Apply edits
   - [ ] Update docs/artifacts if needed
4. Verification
   - [ ] Run lint/tests/scripts as applicable
   - [ ] Record known gaps
5. Reporting
   - [ ] Required metrics table completed
   - [ ] Pages content update prepared
6. Handoff (for live trading scope)
   - [ ] Strategy note drafted
   - [ ] Checkpoint evidence attached
   - [ ] F/E build and B/E serving integration plan defined
   - [ ] Telegram notification plan for completed trades (`open -> close`) defined
   - [ ] Settings toggle behavior (`on/off`) for Telegram notifications defined

## Verification Gates

- Gate A (static):
- Gate B (runtime):
- Gate C (artifact consistency):

## Metrics Contract

- [ ] Return
- [ ] Alpha
- [ ] Sharpe
- [ ] Max DD
- [ ] Win Rate
- [ ] Profit Factor
- [ ] Trades
- [ ] Expected return per trade

## ML/DL Protocol

- [ ] Train/Test split is time-based
- [ ] Test window is the most recent 1 year
- [ ] Train window is all earlier history

## Risks and Mitigations

- Risk:
  - Mitigation:

## Expected Deliverables

- Code:
- Documents:
- Generated artifacts:

## Publishing Targets

- `pages/` updated for GitHub Pages
- `docs/` sync note (if legacy pages still consume docs)
- `pages/` includes practical trading simulation section with:
   - Seed capital
   - Position size per trade
   - Execution model and cost assumptions (fees/slippage/leverage)
