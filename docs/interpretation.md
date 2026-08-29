# Interpretation guide

## Reading the diagnostic probability

| Probability | Reading |
| --- | --- |
| Above 80 percent | Strongly consistent with TANGO2 deficiency. Supports referral for sequencing. |
| 50 to 80 percent | Moderately consistent. Genetic evaluation with TANGO2 in the differential is reasonable. |
| 20 to 50 percent | Not typical, but the model is not confident either way. Clinical judgement should lead. |
| Below 20 percent | Not consistent with TANGO2 deficiency. Does not exclude the diagnosis. |

## Three things a user should understand

**A Tier 1 marker outranks the probability.** Paroxysmal lethargy and
compensatory head posture are not model inputs. They appear in zero control
patients, so they cannot belong to a shared vocabulary feature set. They are
reported separately and should be acted on regardless of what the model says.

**A low probability does not exclude the disorder.** The model learned from
patients phenotyped thoroughly after diagnosis. A child assessed early with an
incomplete record can score low and still have TANGO2 deficiency.

**Below the critical amplitude does not mean safe.** In the source research, 83
percent of stochastic realisations just below the critical value still contained
at least one crisis. The reading is lower probability, not safety.

## What calibration means here

Discrimination answers whether the model ranks patients correctly. Calibration
answers whether a stated 70 percent actually corresponds to 70 percent observed
frequency. A referral decision needs both, which is why the Brier score is
reported alongside the AUC and why the calibration curve is shown rather than
buried.

## What the crisis model output is for

The intended use is to graduate monitoring intensity, not to name a date. The
clinically meaningful quantity is where a patient sits relative to the critical
amplitude. A patient well above it would warrant closer cardiac surveillance, a
lower threshold for admission during febrile illness, and a more aggressive sick
day plan. Any use of a specific simulated day as a decision point misuses the
model.
