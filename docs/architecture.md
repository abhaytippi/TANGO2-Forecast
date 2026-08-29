# Architecture

## Layering

The project has three layers and the dependency arrow points one way only.

```
Presentation   app.py, src/tango2_forecast/ui/
Domain         src/tango2_forecast/{forecast,ml,reports}/
Scope          src/tango2_forecast/disclaimer.py
```

Nothing in the domain layer imports Streamlit. This is enforced by convention
and is checkable: `grep -r streamlit src/tango2_forecast/{forecast,ml,reports}`
returns nothing.

## Module A, the crisis model

| File | Responsibility |
| --- | --- |
| `parameters.py` | Immutable validated parameters. Every default records where it came from and whether it was calibrated or fixed by clinical information. |
| `operators.py` | Three point Laplacian with two selectable no flux treatments, LAPACK banded implicit operator, reference Thomas solver. |
| `solver.py` | The implicit explicit integrator with crisis detection. Realisations advance in lockstep so an ensemble costs one banded solve per step, not one per path. |
| `steady_state.py` | Damped Newton with pseudo transient continuation, bisection for the critical amplitude. |
| `severity.py` | Cohort relative severity scale and regime classification. |
| `sensitivity.py` | Elasticity analysis with common random numbers for variance reduction. |
| `verification.py` | Convergence, conservation, and stability studies. |

## Module B, the diagnostic classifier

| File | Responsibility |
| --- | --- |
| `cohort.py` | Phenotype vocabulary with published prevalences, and the synthetic cohort generator. |
| `classifier.py` | Random Forest, repeated stratified cross validation, provenance tracking, patient level prediction. |

## Module C, reports

| File | Responsibility |
| --- | --- |
| `pdf.py` | Report layout. Has no dependency on the dashboard, the solver, or the classifier, so it can be unit tested alone. |

## Design decisions worth knowing

**Ensembles are batched, not looped.** The implicit matrix is constant in time,
so it is assembled once and every stochastic realisation is advanced in the same
LAPACK call. An ensemble of 200 paths over 900 days runs in a couple of seconds.

**Two boundary treatments are available.** The default reproduces the paper's
scheme, whose one sided Neumann rows give first order convergence. A mirrored
ghost node option recovers second order. Having both makes the paper's own
diagnosis of its convergence behaviour checkable rather than assumed.

**The steady state solver uses pseudo transient continuation.** The reaction term
is a downward parabola with two roots, and a cold Newton start converges to the
unstable one. Relaxing by time stepping first lands in the basin of the branch
the dynamics actually reach.

**Sensitivity analysis reuses the same seed across baseline and perturbed runs.**
Without this coupling, an elasticity of 0.03 is indistinguishable from sampling
noise, and calling a parameter robust would be an artifact rather than a finding.
