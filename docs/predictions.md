# Predictions

The outcomes of the analysis plan (Sections 1 and 9), predicted before any fleet-layer result
was seen. The fleet layer had already been run on the model-selection months (May to August
2019; `docs/prior-looks.md`), but none of its outputs had been read by the author: no offer
size, delivery rate, value of the oracle's α, share of days with α = 0, spread of offer sizes,
or anything computed from them. What had been read: the analysis plan, the per-car scores on
May to August 2019, the checks that do not involve an offer (`docs/data.md`) and the earlier
looks at the sealed periods (`docs/prior-looks.md`). The plan's terms were explained beforehand
in a guide written by someone who had seen the fleet-layer results; the guide used only what
had been recorded before them.

Each prediction is quoted as written, unedited, below the hypothesis it answers. "Site-wide"
means the population offer (arm 5). The labels H3a and H3b are those used when the predictions
were written, and they reverse the order of the plan's H3: H3a is its second clause and H3b
its first.

## H1

In the frozen development evaluation (September 2019 to February 2020), the learned offer
(arm 4) is at least 10% larger than the population offer (arm 5) under the same calibration,
with a paired week-block interval that excludes zero.

> H1: fail. Learned about 10% smaller than site-wide. Why: leaning. The whole day's offer depends on the difficult cars too. One unusual driver can force the method to be cautious, leading to overfitting the model to one difficult errors. on top of that, human change, so familiar drivers can also change their behaviour suddenly. and this will be more possible if the learning model does not 'bounce back' when it releases it has been to cautious.

## H2

The union-bound offer (arm 3) is delivered on at least 98% of days and is at most half the
size of arm 4.

> H2: pass. Why: leaning. The method uses extremely cautious guesses.

## H3a

From the reopening date of Section 7, an offer calibrated only on days before the closure and
never updated is delivered on fewer than 80% of the first 20 scored weekdays.

> H3a: pass. Why: sure. Human behaviour changed significantly after the reopening, and the model is trained based on data before the reopening.

## H3b

From the reopening date, adaptive day-level calibration restores a 90% running average of
delivered days, over 20 weekdays, within 40 scored weekdays.

> H3b: pass. Why: sure. The method reduces how bold its predictions are after each miss. If enough bad days occur, it can become so cautious that it offers zero. Zero offers are always deliverable.

## H4

The per-car learned bound on remaining stay keeps at least 0.5 hours more than the population
bound at matched 90% reliability on the frozen development evaluation sessions.

> H4: pass. About 2.5 hours more. Why: leaning. The learned model has lower errors, especially for known drivers. Humans learn from their mistakes and adjust their behaviour. Therefore, a machine-learning model might be able to recognise those changing patterns and make increasingly accurate predictions.

## Outcome

Which outcome of Section 1 the data give: (1) the learned offer is larger at the same delivery
rate and keeps its margin across the change; (2) once both offers are calibrated by day, the
learned offer is no larger than the population offer; (3) offers calibrated car by car fail at
fleet level right after the change, and restoring 90% of days costs a measured share of the
offer; (4) a failed data check or a degenerate offer.

> Outcome: 3 (main one, and any second). Why: Old offers fail after the change. Daily adjustment restores reliability, potentially by shrinking the offers.

## Jet Propulsion Laboratory

Whether any of the above changes at the replication site, run once with the pipeline trained
at Caltech.

> likely not, the outcome might be the same
