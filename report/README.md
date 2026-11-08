<!-- Written by `uv run fleetoffer report` from src/fleetoffer/report.md, with every number read from the result tables. Edit the template, not this page. -->

# Fleet Charging Offers from Forecast Per-Car Sets: Where the Flexibility Is Lost

Weixuan (Alex) Lu

The full report behind the [front page](../README.md). Every result on this page is read from the result tables by `uv run fleetoffer report`; the settings, and the data checks and earlier looks quoted from [`docs/data.md`](../docs/data.md) and [`docs/prior-looks.md`](../docs/prior-looks.md), are typed. The plan it follows is [`docs/analysis-plan.md`](../docs/analysis-plan.md), and every change made after that plan was fixed is listed in [`docs/registration.md`](../docs/registration.md).

**Abstract.** An aggregator can sell the charging flexibility of the cars parked at a workplace only if it delivers what it offers. This report builds the inner-approximation offer of Al Taha, Vincent and Bitar from per-car sets whose deadlines and energy ceilings are lower quantiles forecast at 10:00, sets one quantile level per day from past days so that every bound holds on 90% of days (the registered rule) or the offer is delivered on 90% of days (a secondary rule), and scores the offer by replay on the Caltech and JPL sessions of ACN-Data, with the hypotheses fixed before the test months were opened. In the frozen evaluation (September 2019 to February 2020) the offer made with perfect knowledge reaches 42% of the exact band width; made from forecasts with the allocation fixed in the morning and calibrated on delivery, the population model's offer keeps 3.9% of the perfect-knowledge offer, delivered on 90% of days. Knowing in advance which cars take nothing more does not restore the offer; a split of each profile among the cars chosen with hindsight keeps 25.5% to 74.3% of it at 90% delivery. A short remark shows why most energy bounds are zero at the small shared levels the registered rules reach: a car's energy bound is zero whenever its chance of taking nothing more is at least that level, and 16.4% of car-days take nothing more. A learned per-car model keeps 2.93 h of each car's remaining stay against 2.01 h for a population rule at matched coverage, and under the fixed allocation, calibrated on delivery, the gain does not reach the fleet offer: 0.18 against 1.76 kWh, delivered on 91% and 90% of days.

## 1. Introduction

An aggregator that controls the charging of the cars parked at a workplace can offer the grid a set of charging profiles for the rest of the day, and is paid only if it then follows whichever profile is called. The set it can promise depends on the cars plugged in: when each will leave and how much energy each will still take. Both are unknown when the offer is made.

With each car's flexibility known, the fleet's flexibility is the Minkowski sum of the cars' sets. Al Taha, Vincent and Bitar [1] place a shifted, scaled copy of a base set inside that sum by a linear programme whose solution also allocates any profile in the copy to the cars by an affine map; for charging-only cars with one connection interval the exact sum is a generalised polymatroid [2]. Both start from sets that are known, and [6] certifies an aggregate set for cars drawn independently from one known distribution. At the time of the offer the sets are not known. The question here is what is left of the offer when each car's set is built from forecasts made at that time, and the offer must be delivered on 90% of days.

The construction is one-sided: each car's set uses a deadline no later than it is forecast to leave and an energy ceiling no higher than it is forecast to take. On any day on which all these bounds hold, every learned set lies inside the car's true set, so the offer and the allocation fixed in the morning are delivered (Proposition 1). Whether they do hold is decided by one quantile level shared by every car, chosen each morning from past days by adaptive conformal inference [3]. The evaluation replays each day against the cars actually present, in time order, on the Caltech and JPL sessions of ACN-Data [4], with the analyses, thresholds and hypotheses fixed in a tagged commit before the test months were opened.

Three results are reported: an accounting that sets the offer made with perfect knowledge of every car's set beside the exact band and beside offers made from forecasts at a 90% delivery target, with the cars that take no more energy known, and with each profile split among the cars with hindsight (Section 5.1); the zero energy ceilings at the levels the registered rules reach, and why removing them does not restore the offer (Remark 1, Section 5.2); and the registered comparisons of a per-car learned model with a population rule, with every per-car model's scores, before and across the 2020 closure of the campus (Section 5.3).

## 2. Problem

Symbols follow each cited paper: $`\alpha`$ is the scaling factor of programme (24); $`\gamma_i`$ and $`\Gamma_i`$ are the per-car affine map of [1]; $`\gamma`$ without a subscript and $`a`$ are the step and the working level of adaptive conformal inference [3]; $`t`$ indexes steps in Section 2 and days in Section 3.

### 2.1 Per-car sets

The offer is made at 10:00 for 10:00 to 20:00, in $`K = 40`$ steps of $`\delta = 0.25`$ h. A car with deadline $`d`$ (a number of steps), energy ceiling $`e`$ (kWh after 10:00) and power cap $`c`$ can follow any profile in

```math
U(d,e) = \Big\{\, u \in \mathbb{R}^K :\ 0 \le u_k \le c\,\mathbf{1}_{k \le d},\ \ \delta \sum_{k=1}^{K} u_k \le e \,\Big\} \qquad (1)
```

where $`\mathbf{1}_{k \le d}`$ is 1 up to the deadline and 0 after it: a charging-only case of the generalised battery model of [1, eq. (2)], with the energy ceiling as the upper limit on cumulative energy and no lower limit. The zero profile lies in every set, so a delivered offer means the fleet can follow the profile within caps, deadlines and ceilings, not that each driver was charged. The *true set* of a car uses its unplug time rounded down to the grid (20:00 if later) and the energy it took after 10:00; the cap is an assumed $`c = 6.7`$ kW.

### 2.2 The offer and its allocation

For $`n`$ cars with sets $`U_i`$ and a base set $`U_0`$, Theorem 2 of [1] gives sufficient linear conditions for $`p + \alpha U_0 \subseteq \sum_i U_i`$; programme (24) of [1] maximises $`\alpha`$ under them, and its Section IV allocates a profile $`p + \alpha u_0`$ to car $`i`$ as $`\gamma_i + \Gamma_i u_0`$ [1, eq. (31)], fixed when the offer is made. The base set is either fixed from the training period or the *day-average base set*: the average of the present cars' constraint vectors as in [1, eq. (11)], taken over the learned sets of the day (the true sets for the perfect-knowledge offer), with each car's power limit written as the tightest its set implies, $`\min(c, e/\delta)`$ up to its deadline ([`docs/theory.md`](../docs/theory.md), "The car's vector in the base set"). When every car's set forces $`u_k = 0`$ at a step where $`U_0`$ can charge, programme (24) has no point with $`\alpha > 0`$, and with $`\alpha \ge 0`$ allowed its optimum is 0 (proved in [`docs/theory.md`](../docs/theory.md)), which a base set fixed in advance meets often. A one-dimensional offer, a segment $`b + \lambda v`$ with $`|\lambda| \le A`$ that moves charging out of 10:00–11:00 and into 11:00–13:00, was registered as the fallback for that case.

Cumulative energy by step $`t`$ is linear in $`u`$, so the exact band of the fleet's cumulative energy is

```math
\Big[\, 0,\ \sum_{i=1}^{n} \min\big(e_i,\ c\,\delta \min(t, d_i)\big) \Big] \qquad (2)
```

and the size of an offer from programme (24) is its band width averaged over 14:00, 16:00 and 18:00, in kWh, beside the same average of (2). The size of the one-dimensional offer is its amplitude $`A`$, the energy it moves out of or into 10:00–11:00, which is not on the same scale. Its reference profile, each car's flat half profile within its learned set, is checked as well, so a one-dimensional offer of size zero can still fail on a day on which some bound fails.

### 2.3 Learned sets and containment

**Proposition 1 (containment).** If $`d' \le d`$ and $`e' \le e`$, then $`U(d', e') \subseteq U(d, e)`$.

*Proof.* Take $`u \in U(d', e')`$. For $`k \le d'`$, $`0 \le u_k \le c`$ and $`k \le d`$; for $`k > d'`$, $`u_k = 0`$, which every set allows. And $`\delta \sum_k u_k \le e' \le e`$. So $`u \in U(d, e)`$. ∎

A learned set uses a lower quantile of the car's remaining stay, rounded down to the grid, and a lower quantile of its remaining energy. On the event $`\mathcal{E}`$ that every present car's two bounds hold, each learned set lies in its true set; Minkowski sums preserve inclusion, so the offer computed on the learned sets lies in the true sum, and the allocation fixed at 10:00 gives each car a profile in its true set. By the union bound, $`\Pr(\mathcal{E})`$ is at least one minus the sum of the $`2n`$ miss probabilities, with no assumption on how cars depend on each other. The event is sufficient for delivery, not necessary, and the inclusion fails for a lower limit on energy, which is why there is none.

### 2.4 How a day is scored

Each day's offer is checked on nine profiles: for each of 12:00, 14:00, 16:00 and 18:00, the two that move the most energy across it, and a least-cost profile under a stylised tariff. *Primary check*: each car follows its allocation as far as its true set allows; the day is delivered if no profile leaves more than $`10^{-4}`$ kWh undelivered. *Secondary check*: the largest energy the cars present can take within their true sets under the profile's total power, an upper bound that needs hindsight (the *hindsight split*). Offers are compared at the same delivery target, each with its achieved delivery rate beside it; where the achieved rates differ, the text says so.

## 3. Learned bounds and the shared level

### 3.1 Per-car models

The *learned* model is gradient-boosted quantile regression (pinball loss, one model per level and target) of remaining stay and remaining energy, on inputs known at 10:00: time and calendar, station, cars present, the driver's earlier sessions and a typed plan if recorded in time, refitted monthly on the past 90 days. The *population* rule gives each car the Kaplan–Meier quantiles of the site's recent sessions given the time it has been plugged in; the *stratified* rule gives empirical quantiles of earlier 10:00 car-days in its cell (plug-in hour, Friday, registered driver). All three predict the same 16 levels; a level between two is interpolated. Every per-car model, with the further baselines of the plan, is scored in Table 3 (Section 5.3).

### 3.2 One level per day

Every present car gets its two bounds at one level $`q`$. A day's score $`s`$ is the largest grid level at which every bound of the day held, so all held at $`q`$ exactly when $`q \le s`$. With $`m`$ scored days in the past 90 calendar days, a rule at miss level $`a`$ uses the $`\lfloor a(m+1) \rfloor`$-th smallest score, and adaptive conformal inference [3, eq. (2)] moves $`a`$ after each scored day:

```math
a_{t+1} = a_t + \gamma\,(0.10 - \mathrm{err}_t), \qquad a_1 = 0.10, \quad \gamma = 0.03. \qquad (3)
```

In the registered rule $`\mathrm{err}_t = 1`$ when some bound of the day failed. A secondary rule, added after the first development results and labelled as such, sets $`\mathrm{err}_t = 1`$ when the offer at the level used, or at a lower level, was not delivered, and uses the day-average base set. The union bound, each bound at level $`0.10/(2n)`$, is a further arm.

Summing (3) over $`T`$ scored days from working level $`a_S`$ gives a share of failed days of $`0.10 + (a_S - a_{S+T})/(\gamma T)`$. A working level at or below 0 gives the trivial bound (deadline at 10:00, energy 0), which cannot fail, so when each outcome is known before the next decision the level never falls below $`-0.9\gamma`$, and whatever the days do

```math
\frac{1}{T} \sum_{t=S}^{S+T-1} \mathrm{err}_t \ \le\ 0.10 + \frac{a_S + 0.9\gamma}{\gamma T}. \qquad (4)
```

This is the argument of [3, Prop. 4.1] kept on the one side a delivery target needs; its two-sided bound, from the first level, would give 36.7% here. From the levels the rules had reached when the frozen evaluation began, (4) gives 13.7% to 19.8% over its 116 scored weekdays. A day's outcome is known only when its last car unplugs, so the rules update as outcomes arrive: 85% of those days were known before the next decision and at most 3 were pending at once, which loosens the bound to 16.1% to 22.1% ([`docs/theory.md`](../docs/theory.md), "Late outcomes").

### 3.3 Zero ceilings at small levels

The level a day-level target needs is small: $`0.10/(2n)`$ is 0.28% at the median of 18 cars present, and on the model-selection months the registered conformal rules settle between 0.30% and 2.80% (Section 5); the rules calibrated on delivery run at higher levels. At this site the energy a car still takes after 10:00 has an atom at zero: 16.4% of car-days through August 2019 take nothing more. Let $`E_i \ge 0`$ be car $`i`$'s remaining energy, $`X_i`$ its inputs, $`F_i(e) = \Pr(E_i \le e \mid X_i)`$, $`\pi_i = F_i(0)`$, and its bound at level $`q`$ the quantile $`\hat e_i(q) = \inf\{e \ge 0 : F_i(e) \ge q\}`$.

**Remark 1 (zero ceilings).** $`\hat e_i(q) = 0`$ if and only if $`\pi_i \ge q`$. If $`\pi_i \ge q`$, then 0 is in the set and $`E_i \ge 0`$, so $`\hat e_i(q) = 0`$; if $`\pi_i < q`$, right-continuity gives $`F_i(\varepsilon) < q`$ for some $`\varepsilon > 0`$, so $`\hat e_i(q) \ge \varepsilon`$.

A car with a zero ceiling adds nothing to (2), and the offer lies inside the sum of the learned sets, so at level $`q`$ only cars with $`\pi_i < q`$ can contribute: at level 0.28% a car counts only if its inputs rule out, to within that probability, that it takes nothing more. The same holds for the deadline with the chance of leaving within 15 minutes. Proposition 1 holds at every level; it assumes the bounds hold, not that they are informative. Remark 1 was written after the first development run showed offers near zero and before the counts in Section 5.2 were read (it was registered as a proposition); for the empirical rules it holds exactly on the grid, for the trees only as far as they estimate the conditional quantile.

## 4. Data and protocol

ACN-Data [4] records each charging session's plug-in and unplug times, energy and, for registered drivers, a typed request. Decisions are made at 10:00 on weekdays; a day is scored if at least five present cars take more than 0.5 kWh after 10:00. The energy a car takes after 10:00 is not recorded; a per-session time series that would give it failed a quality check fixed before it was fetched, so it is estimated by spreading the session's energy evenly from plug-in to the end of charging, and the ceiling is approximate ([`docs/data.md`](../docs/data.md)).

Caltech sessions through August 2019 are development data (May–August 2019 for model selection); September 2019 to February 2020 is the frozen evaluation; from 1 March 2020 is the sealed test. It spans the campus closure, after which the median stay fell from about six hours to under two; its scored weekdays are 9 in March 2020, before the closure, and 17 from July 2021. The JPL site is a replication, run once with the Caltech-trained pipeline frozen and, separately, updating on JPL days. One of its 37 runs, the stratified model calibrated on delivery, updating on JPL days, did not finish and is left out; no table or figure here uses it ([`docs/registration.md`](../docs/registration.md), "Changes after the tag"). The sealed periods were looked at while the project was scoped, for stay statistics and a reading of raw sessions; the evaluation months also gave the energy of 50 sampled sessions and the daily counts of cars present and taking energy, which set the adaptive rule's step. Every look is listed in [`docs/prior-looks.md`](../docs/prior-looks.md), and what they touched is read as a check rather than a blind test. Hypotheses H1–H4, their thresholds and the written predictions ([`docs/predictions.md`](../docs/predictions.md)) were committed and tagged before any confirmatory run. On the model-selection months the perfect-knowledge offer's $`\alpha`$ was below 0.2 on 37.6% of scored weekdays with the fixed base set, above the registered threshold of 25%, so every registered arm uses the one-dimensional fallback.

## 5. Results

### 5.1 Where the offer goes

![Days delivered and mean offer size by month, Caltech from September 2019](figure-1.png)

**Figure 1.** By month, Caltech from September 2019, scored weekdays only: the share of days delivered under the allocation fixed at 10:00 (top) and the mean offer size (bottom). Green, the perfect-knowledge offer (day-average base set). Solid, the learned and population offers calibrated on delivery with the day-average base set, chosen after the May to August 2019 look; dashed, the registered rule on the one-dimensional fallback, whose size is an amplitude, not a band width (Section 2.2); those lines lie at or near zero. No weekday was scored from April 2020 to June 2021: on every one of them fewer than five cars took energy after 10:00. Drawn by `uv run fleetoffer report`.

Figure 1 gives the offers by month at Caltech: the perfect-knowledge offer is tens of kWh, the offers made from forecasts are near zero, and both are delivered on most days. Table 1 sets the offer made with perfect knowledge of every car's set beside the exact band and beside offers under each relaxation, on the day-average base set. Each row is read at its own delivery rate and not added to the others; the note under the table defines the rows. With perfect knowledge the offer reaches 42% of the exact band width on the model-selection months (39.4 of 93.7 kWh), and 42%, 39% and 51% in the frozen evaluation, the sealed months and at JPL. The gap is that of programme (24) as used here: the conditions of [1, Thm. 2] in their structure-preserving form ($`P = \alpha I`$), whose certificate is the affine allocation [1, eq. (31)], with this base set. For a given base set the band width is $`\alpha`$ times a constant, so the programme's objective is the width scored here.

Under this formulation (one common level, a template, fixed allocation) the offer made from forecasts keeps about 4% of the perfect-knowledge offer (1.55 of 39.4 kWh): the population model calibrated on delivery, delivered on 88% of days. The learned model calibrated in the same way keeps 0.2%, and under the registered rule the learned offer is zero on every scored weekday. With each model's level chosen with hindsight, no model keeps more than 3.6% at 90% delivery; Figure 2 shows every swept level against the days it was delivered in the frozen evaluation and in 2021. In the frozen evaluation the population rule as run keeps 3.9% (delivered on 90% of days), in the sealed months 3.4% (92%) and at JPL 0.4% (96%).

Two diagnostics locate the loss; neither is a method an operator could run. On the model-selection months, with the cars that take nothing more known at 10:00, the offer keeps at most 1.9% (Section 5.2). With the same forecasts but each profile split among the cars present with hindsight, it keeps 20.8% to 55.9% (25.5% to 74.3% in the frozen evaluation; 38.8% to 67.7% in the sealed months). The order of the models reverses between the two: under the fixed allocation the population model keeps the most, under the hindsight split the learned model does (learned and population under the hindsight split: 56% and 21%; 74% and 32%; 68% and 56%; 60% and 39%, in the order of Table 1's columns). On the model-selection months and the one-dimensional offer, the registered form, the best of the three models keeps 0% at 90% delivery and the hindsight split 72.4% to 80.5%. Both diagnostics are consistent with the allocation fixed at 10:00 [1, eq. (31)], more than the cars that take nothing, being where the offer is lost. That allocation is guaranteed only when every car's learned set lies inside its true set (Proposition 1); under it each car must follow its own share; a split chosen afterwards lets one car's slack cover another's shortfall. The hindsight split chooses both the level and the split after the fact, so it bounds what a re-split could recover; it does not show what an online re-split would. On the model-selection months, at level 0.15, every bound of the learned model held on 0% of scored weekdays and the fixed allocation delivered on 1%, while the hindsight split delivered on 100%.

**Table 1.** Where the perfect-knowledge offer goes (day-average base set). From the fourth row on, each cell gives the learned / population / stratified models in that order: shares in percent of the perfect-knowledge offer, and in the last row the share of days delivered.

| | May–Aug 2019 | Sep 2019–Feb 2020 | Sealed months | 2021 only | JPL |
|---|---|---|---|---|---|
| Scored weekdays (fewest in a row) | 85 (84) | 116 (115) | 26 (16) | 17 (7) | 636 |
| Exact band (kWh) | 93.7 | 108.5 | 63.9 | 59.8 | 250.2 |
| Perfect knowledge (kWh, share of the band) | 39.4 (42%) | 45.6 (42%) | 25.0 (39%) | 24.9 (42%) | 128.5 (51%) |
| Forecasts, 90% of days | 0 / 3.6 / 0 | 0.5 / 3.6 / 1.2 | 0.3 / 2.7 / 1.4 | 0 / 0.8 / 0.8 | – |
| Empty cars known, 90% of days | 0 / 1.9 / 0 | 4.3 / 3.5 / 0 | 5.0 / 8.1 / 7.4 | 1.0 / 0.8 / 3.3 | – |
| Hindsight split, 90% of days | 55.9 / 20.8 / 32.9 | 74.3 / 32.1 / 25.5 | 67.7 / 56.4 / 38.8 | 60.0 / 39.2 / 30.0 | – |
| Level from past days | 0.2 / 3.9 / 1.5 | 0.4 / 3.9 / 1.3 | 0 / 3.4 / 7.0 | 0 / 1.0 / 7.5 | 0.2 / 0.4 / 0.1 |
| Days delivered by that rule (%) | 94 / 88 / 94 | 91 / 90 / 91 | 100 / 92 / 88 | 100 / 94 / 88 | 99 / 96 / 99 |

The exact band is that of the exact aggregate set, which an offer of the form $`p + \alpha U_0`$ need not reach; in brackets beside the perfect-knowledge offer, its share of the band. Each share is taken over the days its row covers: a model makes no offer on a day without a fit, so a row can cover fewer scored weekdays than the perfect-knowledge offer (fewest in brackets). Sealed months: 9 scored weekdays in March 2020, before the closure, and 17 from July 2021; 2021 only: the latter. "90% of days": the largest swept offer delivered on at least 90% of scored weekdays, level chosen with hindsight. "Empty cars known": the cars that take nothing after 10:00 are dropped and the models refitted on car-days that took energy. "Hindsight split": delivery checked with the split of each profile chosen with hindsight among the cars present, an upper bound. "Level from past days": the rule calibrated on delivery, as run. JPL: track 1, the Caltech pipeline frozen; there the sweep is run on the one-dimensional offer only.

![Mean offer size against the share of scored weekdays delivered, at every shared level](figure-2.png)

**Figure 2.** Mean offer size against the share of scored weekdays delivered under the allocation fixed at 10:00, one point per shared level, day-average base set; dashed, the empty cars known; dotted, the perfect-knowledge offer; vertical line, 90% of days. Right: days delivered by month for the rules calibrated on delivery, with 95% intervals from resampling whole weeks. Drawn by `uv run fleetoffer report`.

### 5.2 Zero ceilings

Knowing in advance which cars take nothing more does not restore the offer. With those cars known and the models refitted on car-days that took energy, the offer delivered on 90% of days keeps at most 1.9% of the perfect-knowledge offer on the model-selection months, 4.3% in the frozen evaluation and 8.1% in the sealed months (Table 1). Once the empty cars are known the energy ceilings are positive, but the bounds miss more often than their level: on the model-selection months, at level 0.2%, every bound of the learned model refitted on car-days with energy held on 64% of scored weekdays, against at least 93% by the union bound at the median of 18 cars present if each bound missed with probability 0.2%.

What the empty cars do account for is that most energy bounds are zero at the levels the registered rules reach; they do not account for every zero offer. Figure 3 counts the zero ceilings behind Remark 1 on the model-selection months; the three readings below were registered before the counts were read. The second and third hold by construction for the empirical rules (their quantiles never cross, and a rule refitted on car-days that took energy has no atom at zero), so they are consistency checks; the first is the count that matters. At the level each registered rule reached (median 0.3% for the learned model, 0.9% for the stratified table, 2.8% for the population model), the share of cars that took energy but were given a zero energy ceiling is 100%, 100% and 73%. The share never rises with the level in any model or offer form (the second reading holds), and with the empty cars known it is at most 1.7% at any level. Where every car that took energy has a zero ceiling, as for the learned model and the stratified table, that alone makes the offer zero; for the population model it does not. Nor are the zero ceilings the only cause: with the empty cars known, the one-dimensional offers of the population and stratified models average at most 0.0003 kWh at every swept level up to 5%, so for them something these counts do not separate also binds. In the frozen evaluation the shares at the rules' levels are 88%, 99% and 69%. In the sealed months the median level of every rule was 0, where every bound is zero by construction, so those months do not test the proposition.

![Share of cars given a zero energy ceiling and a zero deadline at each shared level](figure-3.png)

**Figure 3.** Share of cars that took energy but were given a zero energy ceiling (left), and of all cars given a zero deadline (right), at each shared level $`q`$, May to August 2019. Dashed: the empty cars known. Dotted lines: the median level each registered rule used. Drawn by `uv run fleetoffer report`.

### 5.3 Registered comparisons

Table 2 gives each registered hypothesis with its reading. Under the registered design H1 reads "fail" (learned 0.00 against population 0.00 kWh over 116 scored weekdays, delivered on 98% and 94% of days: the registered arms run the one-dimensional fallback, whose reference profile is checked too, Section 2.2). The reading follows the registered rule, which counts an arm as offering nothing only when its mean offer is below $`10^{-9}`$ kWh: the learned offer is above that line on 0 of the 116 days and the population offer on 1, at most 0.0034 kWh. Under the design calibrated on delivery it reads "fail": 0.18 against 1.76 kWh, delivered on 91% and 90% of days, a difference of −1.57 kWh (95% week-block interval −1.78 to −1.36). H2, the union-bound offer, reads "pass", vacuously as H1 does: it is delivered on 98% of days, and it and the learned offer are 0.00 and 0.00 kWh, so the size clause holds only because both are zero. H3 cannot be tested: the reopening date fixed by the registered rule is 2 August 2021, and the record ends 15 scored weekdays later, while each clause needs 20. H4 reads "pass": at matched 90% coverage the learned bound keeps 2.93 h of each car's remaining stay against 2.01 h for the population bound (coverage 91% and 91%, over the 2,055 car-days on which both rules gave a bound), and 2.11 against 1.52 h in the sealed months. The per-car gain holds in both test periods and, under the allocation fixed at 10:00, does not reach the fleet offer. At JPL, a descriptive replication run once, the calibrated learned offer is 0.28 kWh against 0.54 kWh for the population with the Caltech pipeline frozen (delivered on 99% and 96% of days, so not at matched delivery), and 1.88 against 1.87 kWh updating on JPL days (delivered on 90% and 90%), a difference of 0.01 kWh (95% week-block interval −0.52 to 0.55).

**Table 2.** Registered hypotheses, with the design calibrated on delivery (95% intervals from one-week blocks). H1 and H4 are confirmatory on September 2019 to February 2020 only. Each offer is given with its delivery rate; where the two rates differ, the offers are not at matched delivery.

<table>
<thead><tr><th>Hypothesis</th><th>Design</th><th>Caltech, Sep 2019–Feb 2020</th><th>Caltech, from Mar 2020 (H1 and H4 descriptive)</th></tr></thead><tbody>
<tr><td>H1, learned offer at least 10% larger than population</td><td>registered</td><td>fail: 0.00 against 0.00&#160;kWh, delivered on 98% and 94% of days</td><td>not testable: both offers are zero (delivered on 100% and 100% of days)</td></tr>
<tr><td>H1</td><td>calibrated on delivery</td><td>fail: 0.18 against 1.76&#160;kWh, delivered on 91% and 90% of days; difference −1.57&#160;kWh (−1.78 to −1.36)</td><td>fail: 0.00 against 0.84&#160;kWh, delivered on 100% and 92% of days; difference −0.84&#160;kWh (−1.53 to −0.16)</td></tr>
<tr><td>H2, union bound delivered on at least 98% of days, at most half the learned offer</td><td>registered</td><td>pass: delivered on 98% of days, 0.00&#160;kWh against 0.00 for the learned offer</td><td>pass: delivered on 100% of days, 0.00&#160;kWh against 0.00 for the learned offer</td></tr>
<tr><td>H3a and H3b, after the reopening (2 August 2021)</td><td>both</td><td>–</td><td>not testable: 15 scored weekdays from the reopening, 20 needed</td></tr>
<tr><td>H4, learned stay bound at least 0.5&#160;h longer at matched 90% coverage</td><td>per car</td><td>pass: 2.93 against 2.01&#160;h, coverage 91% and 91%, 2,055 car-days</td><td>pass: 2.11 against 1.52&#160;h, coverage 89% and 89%, 426 car-days</td></tr>
<tr><td>H1 at JPL, Caltech pipeline frozen (descriptive)</td><td>registered</td><td colspan="2">not testable: both offers are zero (delivered on 100% and 100% of days)</td></tr>
<tr><td>H1 at JPL, Caltech pipeline frozen (descriptive)</td><td>on delivery</td><td colspan="2">fail: 0.28 against 0.54&#160;kWh, delivered on 99% and 96% of days; difference −0.27&#160;kWh (−0.31 to −0.23)</td></tr>
<tr><td>H1 at JPL, updating on JPL (descriptive)</td><td>registered</td><td colspan="2">not testable: both offers are zero (delivered on 99% and 94% of days)</td></tr>
<tr><td>H1 at JPL, updating on JPL (descriptive)</td><td>on delivery</td><td colspan="2">fail: 1.88 against 1.87&#160;kWh, delivered on 90% and 90% of days; difference 0.01&#160;kWh (−0.52 to 0.55)</td></tr>
</tbody>
</table>

Table 3 gives the per-car scores Section 4 of the plan requires: every model on the model-selection months (1,521 car-days), and the learned model against the population rule on the frozen evaluation (2,056 car-days). The trees have the lowest stay loss of every model, 0.444 h against 0.594 h for the population rule (0.439 against 0.611 h in the frozen evaluation). Section 4 of the plan asks whether the trees beat each driver's own past and the stratified table (a loss within 1% is called level here; the plan fixes no margin): on energy the trees are level with each driver's own past (0.683 against 0.684 kWh), within 1%, so for energy the learned model's gain over the population rule is what each driver's own past already gives; for known drivers each driver's own past is lower than the trees on energy (0.666 against 0.714 kWh); for anonymous drivers the stratified table is lower than the trees on energy (0.484 against 0.490 kWh). Neither model is calibrated at its nominal level: at level 0.10 the trees' stay bound missed on 14.2% of car-days (14.2% in the frozen evaluation) and the population rule's on 2.8% (2.2%), which is why H4 compares them at the levels that give each 90% coverage on its own past car-days. Figure 4 shows both stay bounds by month across the closure.

**Table 3.** Per-car scores: mean pinball loss over the 16 levels (lower is better), May to August 2019, each cell for all / known / anonymous drivers; the share of stay bounds missed at level 0.10, all drivers; and in the last column the loss for stay / energy, all drivers, from September 2019 to February 2020, where it was computed. "Known" drivers are linked to a registered driver by an app entry before 10:00; at this site they are also the cars with a typed plan ([`docs/data.md`](../docs/data.md)). For anonymous drivers each driver's own past has no past to use and is the population rule. The typed plan, scored only on cars that have one, is left out.

| Model | Stay (h) | Energy (kWh) | Stay missed at 0.10 | Sep 2019–Feb 2020 |
|---|---|---|---|---|
| Learned (boosted trees) | 0.444 / 0.418 / 0.610 | 0.683 / 0.714 / 0.490 | 14.2% | 0.439 / 0.723 |
| Small network | 0.485 / 0.463 / 0.625 | 0.814 / 0.852 / 0.572 | 6.7% | – |
| Linear quantile regression | 0.495 / 0.474 / 0.630 | 0.864 / 0.889 / 0.706 | 11.0% | – |
| Each driver's own past | 0.474 / 0.447 / 0.646 | 0.684 / 0.666 / 0.799 | 6.4% | – |
| Stratified table | 0.502 / 0.484 / 0.618 | 1.053 / 1.142 / 0.484 | 9.9% | – |
| Mixture of Lee, Li and Low | 0.522 / 0.497 / 0.685 | – | 14.9% | – |
| Population rule | 0.594 / 0.585 / 0.646 | 1.086 / 1.131 / 0.799 | 2.8% | 0.611 / 1.295 |

![Per-car lower bound on remaining stay at level 0.10, by month](figure-4.png)

**Figure 4.** The per-car lower bound on remaining stay at level 0.10 by month, Caltech, every car present at 10:00: the share of bounds that held (top) and the hours bounded per car (bottom), learned trees against the population rule, each refitted monthly on what was known then. Months with fewer than 20 bounds are blank. Drawn by `uv run fleetoffer report`.

The analysis plan's Figures 1 to 4 are drawn by `uv run fleetoffer report` into `results/report/`; Figures 1, 2 and 4 here are its Figures 2, 3 and 1, and its Figure 4, the mispredicted cars on each day against delivery, is drawn there and not shown here.

## 6. Discussion

**Outcome.** Of the three outcomes the analysis plan names (its Section 1), the registered design gives a degenerate offer: both offers are zero, and H1 compares two zeros. The secondary design, calibrated on delivery and chosen after the May to August 2019 look, gives the second outcome: once both are calibrated by day, the learned offer is no larger than the population offer. The third, an offer calibrated car by car that fails after the change at a measured cost of recovery, could not be tested (H3).

**What the guarantees cover.** None of the guarantees used here bounds the offer's size. Proposition 1 needs every bound to hold, not to be informative; at the levels the registered rules reach with 18 cars, most energy bounds are zero (Remark 1). The adaptive rule's bound (4) on the share of failed days (the argument of [3, Prop. 4.1]) holds over any number of days whatever the days do, loosened as stated for late outcomes, and it is met by an offer of zero. The finite-sample guarantee of [5, Thm. 4] for chance-constrained programmes under drift assumes independent days and a known bound on the drift between the distributions of days; with days as samples neither is available for these data, so no certificate from it is claimed. For scale, with a grid of 101 scaling levels as its finite decision set, it needs 312 and 1,121 days at the two settings computed here, against the 483 calendar weekdays before March 2020, fewer of them scored.

**Reliability and size.** Reliability was recovered by giving up size. Delivered on 90% of days, the offer made from forecasts keeps at most a few percent of the offer perfect knowledge supports, and the learned per-car model, whose stay bounds are tighter than the population's at matched coverage (H4), gives the smaller fleet offer under the fixed allocation once both are calibrated on delivery (at JPL, updating on JPL days, the two are level). The diagnostics are consistent with the formulation mattering more than the forecasts: a shared level makes every car's bound as cautious as the fleet's target requires, and under the allocation fixed at 10:00 a car's miss can become the fleet's. A per-car level and an online re-split were not tested.

**Limits.** The energy a car takes after 10:00 is estimated, not measured, and the site's own scheduler shapes it: cars took a median 0.64 of what their drivers asked for. Whether a car is still charging at 10:00, which an operator would see, is not recorded, so the diagnostic in which the empty cars are known stands in for it. The power cap of 6.7 kW is assumed for every station, with 3.3 and 11 kW run as sensitivities. Only nine profiles of each offer are checked. H3 is untested for lack of days after the reopening, and the sealed months held 26 scored weekdays, so their comparisons are wide. The evaluation months and both sealed periods were looked at while the project was scoped (Section 4).

## References

1. F. Al Taha, T. L. Vincent and E. Y. Bitar, "An efficient method for quantifying the aggregate flexibility of plug-in electric vehicle populations," *IEEE Transactions on Smart Grid*, vol. 16, no. 4, pp. 3142–3154, 2025.
2. K. Mukhi, G. Loho and A. Abate, "Exact characterization of aggregate flexibility via generalized polymatroids," *IEEE Transactions on Smart Grid*, vol. 17, no. 3, pp. 2131–2142, 2026.
3. I. Gibbs and E. J. Candès, "Adaptive conformal inference under distribution shift," in *Proc. NeurIPS*, 2021.
4. Z. J. Lee, T. Li and S. H. Low, "ACN-Data: analysis and applications of an open EV charging dataset," in *Proc. ACM e-Energy*, 2019.
5. S. Yan, F. Parise and E. Y. Bitar, "Data-driven approximations of chance constrained programs in nonstationary environments," *IEEE Control Systems Letters*, vol. 6, 2022.
6. K. Mukhi, C. Qu, P. You and A. Abate, "Robust aggregation of electric vehicle flexibility," in *Proc. HSCC*, 2025.

## Appendix A. Use of AI tools

**Decided by the author.**

- The research question: how drivers' behaviour, and its change over time, limits the flexibility a fleet can offer, chosen over a framing around battery wear.
- The design, a fleet offer built from per-car bounds and scored by delivery, chosen among candidate designs drafted with AI assistance.
- The roles of the data: Caltech before March 2020 for development, the months from the closure as the sealed test, JPL as a single replication.
- The predictions in [`docs/predictions.md`](../docs/predictions.md), written before any fleet result was seen and stored unedited.
- After the first fleet results: to report the registered rule's zero offers as the primary result rather than replace them; to add calibration on delivery as a secondary analysis, labelled as chosen after that look; to reset no threshold; and to add no model, tuning or controller before the evaluation.
- The diagnostic in which the cars that take nothing more are known, to test whether the charging status an operator sees at 10:00 would recover the offer; the rules for stating results (measured comparisons first, mechanisms only as "consistent with", registered results, diagnostics and future work kept apart); and the conclusion that reports both readings.
- What the project claims and what is published.

**Done with AI assistance:**

- The literature search, through the OpenAlex API; theorem statements were checked against the papers' texts.
- Majority of the code, tests and runs, written and run by an AI coding assistant (Claude Code) and reviewed by the author; each commit carries a co-author line.
- The repo organised by an AI assistant.
