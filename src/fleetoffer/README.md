# fleetoffer

The code behind the [report](../../report/), one module per job. Each command of `uv run fleetoffer` is defined in `pipeline.py`.

```
acn.py          ACN-Data download, sessions and per-session time series, with fingerprints
data.py         sessions table, cleaning rules, seals
interface.py    what a model may see at the decision time
engine.py       replay per car and per day; the only reader of unplug times
academic.py     Caltech's academic calendar and Institute holidays, as day flags
features.py     inputs known at 10:00, from sessions that had ended
quantiles.py    per-car quantile models (boosted trees, a small network, linear) and their
                monthly refits inside the engine
baselines.py    population rule, stratified table, each driver's own past, the mixture
                predictor of Lee, Li and Low, the typed plan
models.py       Kaplan-Meier quantiles of remaining stay, and two simple stay rules the demo
                runs
percar.py       the per-car layer on development months: settings search, refits, scores
offer.py        per-car sets, the inner-approximation programme, the affine allocation,
                the nine-profile delivery checks and the one-dimensional fallback
calibrate.py    what each day-level route to 90% of days requires (the theorems' arithmetic)
daylevel.py     the day-level rules: one shared level from past days (adaptive and split
                conformal, on every bound holding or on delivery), the union bound, and the
                sweep of offers over every level
devchecks.py    the checks on development data before the tag (analysis plan, Section 10a)
options.py      calibration on delivery and the offer forms, on the model-selection months
levels.py       the level sweep: offer size against delivery at every level, and its tables
evaluation.py   the runs after the tag: frozen evaluation, sealed months, both JPL tracks,
                sensitivities and the hypotheses' readings
h0.json         the base set of every run after the tag, committed and loaded
resample.py     uncertainty by resampling whole weeks
lookahead.py    the look-ahead scramble test, per car and per day
synthetic.py    made-up fleets with known flexibility sets
figures.py      figures, each drawn from a table also written to disk
note.py         the full report: every result read from the tables into report.md, and the
                per-car scores and late-outcome counts it quotes
report.md       the report's text, with a named field for each result
pipeline.py     end-to-end runs with manifests, and the commands
```

Outside this folder: `tests/` holds the unit tests, including the look-ahead tests (no network needed); `fingerprints/` the SHA-256 of every downloaded month.
