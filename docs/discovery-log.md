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
