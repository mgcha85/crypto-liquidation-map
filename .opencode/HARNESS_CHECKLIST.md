# HARNESS_CHECKLIST.md

## Pre-Change

- [ ] Scope is explicitly defined
- [ ] Affected modules and scripts identified
- [ ] Data path assumptions are validated (`/mnt/data/finance`)
- [ ] Hive partition assumptions validated
- [ ] Baseline -> parametric -> ML/DL execution order defined

## During Change

- [ ] Edits are minimal and localized
- [ ] No unrelated refactor is introduced
- [ ] Naming and output conventions are preserved
- [ ] Polars remains default engine unless DuckDB need is documented
- [ ] GPU/parallel opportunities are evaluated and applied when feasible

## Verification

- [ ] `ruff check src scripts` passed or justified
- [ ] `pytest -q` run or test absence documented
- [ ] Modified scripts run or dry-run strategy documented
- [ ] Metrics table includes Return/Alpha/Sharpe/Max DD/Win Rate/Profit Factor/Trades/Expected return per trade
- [ ] ML/DL split uses latest 1 year as test and prior history as train

## Documentation

- [ ] Reproduction commands are written
- [ ] Output files and locations are listed
- [ ] Behavioral changes are documented
- [ ] `pages/` content is updated for GitHub Pages publication
- [ ] `docs/` sync status is documented (if applicable)
- [ ] `pages/` includes seed capital and per-trade position size
- [ ] `pages/` includes practical execution/cost simulation assumptions (fills, fees, slippage, leverage)

## Safety

- [ ] No secrets or credentials added
- [ ] No destructive git commands used
- [ ] External dependency changes are approved

## Final Gate

- [ ] Done criteria in `AGENTS.md` satisfied
- [ ] Open risks and limitations communicated
- [ ] Live trading tasks include strategy note and completed checkpoint evidence
- [ ] Svelte baseline tabs and settings fields are validated for live FE scope
