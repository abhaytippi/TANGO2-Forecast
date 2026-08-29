# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Module A, crisis forecast engine

- `forecast.parameters`: Pydantic models for the reaction-diffusion parameters
  and the space-time discretisation, with the provenance of every default
  recorded on the field and boundary-consistent quadrature weights.
- `forecast.operators`: three-point Laplacian with two selectable no-flux
  treatments, LAPACK-banded implicit operator supporting batched right-hand
  sides, reference Thomas solver, and the von Neumann amplification factor.
- `forecast.solver`: IMEX integrator (Algorithm 1) with crisis detection,
  post-crisis reset, and stochastic realisations advanced in lockstep.
- `forecast.steady_state`: damped Newton steady-state solver with pseudo-
  transient continuation, bisection for the critical amplitude, bifurcation
  sweep with warm starts.
- `forecast.severity`: cohort-relative severity scale (Eq. 8) and regime
  classification relative to the critical amplitude.
- `forecast.sensitivity`: elasticity analysis (Eq. 16) with common random
  numbers, Monte Carlo standard errors, and critical-amplitude shifts.
- `forecast.verification`: convergence, conservation, and stability studies.
- `disclaimer`: single source of truth for all scope-of-use language.
- `tango2-verify` CLI regenerating every published number from source.
- 90 tests, CI across three operating systems and two Python versions.

### Notes on reproduction

- The critical amplitude computes to 0.012701 against a published 0.01273
  (−0.23%); grid refinement drives this implementation to 0.012681.
- Shifts in the critical amplitude reproduce Table 6 to three significant
  figures for all eight parameters.
- Elasticities depend on the patient they are evaluated for; Table 6 is
  reproduced within Monte Carlo error at the top of the calibrated range.
