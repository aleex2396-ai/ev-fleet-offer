# Discovery log

A dated record of how the research question changed and what caused each change. Entries
are added when the change happens and are not edited afterwards; a correction is a new
entry that points to the one it corrects.

Each entry:

```
## YYYY-MM-DD: short title

- Expected:
- Observed: (with the figure, table or commit that shows it)
- What changed:
```

## 2026-11-06: the registered offers are zero on the model-selection months

- Expected: offers of positive size from every arm, so that H1 compares a learned and a
  population offer of measurable size under the same calibration.
- Observed: on May to August 2019 the registered day-level rule gave the learned and the
  population offers zero on every scored weekday; the rules settle at shared levels between
  0.3% and 2.8%, where most cars' lower bounds on energy are zero (`docs/registration.md`,
  "Added after the first fleet run").
- What changed: the registered design stays the primary result, reported even where it is
  zero. Fixed before the tag for every later period: a secondary analysis calibrated on
  delivery rather than on bound misses, a sweep of offer size against delivery at every level,
  a proposition on when a lower bound is zero (`docs/theory.md`, "Zero ceilings at small
  levels") with its check, a diagnostic in which the cars that take nothing are known, and an
  accounting of where the perfect-knowledge offer goes.

## 2026-11-08: the per-car gain does not reach the fleet offer under a fixed allocation

- Expected: (`docs/predictions.md`, written before any fleet result was seen) the learned offer about
  10% smaller than the population offer; offers failing right after the 2021 reopening and
  recovering by shrinking; a per-car gain of about 2.5 hours.
- Observed: the registered offers are zero, or in effect zero, in every period. Calibrated on delivery, the learned
  offer is 0.18 kWh against 1.76 kWh for the population in the frozen evaluation (delivered on
  91% and 90% of days), although the learned stay bound keeps 2.93 h against 2.01 h at matched
  coverage. With each profile split among the cars in hindsight, the learned model keeps the
  most of the perfect-knowledge offer (74% against 32% in the frozen evaluation; an upper bound,
  not a method). The reopening could not be tested:
  the record ends 15 scored weekdays after it (report, Tables 1 and 2).
- What changed: the question moves from the forecasts to the decision that uses them: whether
  a rule that splits each request among the cars using only requests already seen, and
  re-plans when a departure breaks the plan, recovers what the fixed allocation loses; and how
  the gap to the exact hindsight ceiling divides between deciding in real time, simple rules,
  not knowing the cars, and calibration.
