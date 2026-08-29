# TANGO2-Forecast

**An open source clinical decision support tool for TANGO2 deficiency disorder.**

[![tests](https://img.shields.io/badge/tests-90%20passing-brightgreen)](#testing)
[![python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![status](https://img.shields.io/badge/status-research%20software-orange)](#disclaimer)

TANGO2-Forecast turns the framework described in Tippimath, Lalani and Liu into
software a clinician can open and use. It answers two questions that the
underlying research addresses from opposite ends: whether a child's baseline
phenotype is consistent with TANGO2 deficiency before any crisis has occurred,
and how a mechanistic model describes the structure of that child's ongoing
crisis risk.

---

## Table of contents

- [Clinical motivation](#clinical-motivation)
- [What the application does](#what-the-application-does)
- [Architecture and why it was chosen](#architecture-and-why-it-was-chosen)
- [Installation](#installation)
- [Running the application](#running-the-application)
- [Deploying publicly](#deploying-publicly)
- [Example workflow](#example-workflow)
- [Interpretation guide](#interpretation-guide)
- [Explanation of every output](#explanation-of-every-output)
- [The science implemented](#the-science-implemented)
- [Verification against the paper](#verification-against-the-paper)
- [A note on data](#a-note-on-data)
- [Repository structure](#repository-structure)
- [Testing](#testing)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)
- [Contributing](#contributing)
- [Citation](#citation)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Clinical motivation

TANGO2 deficiency disorder is a rare autosomal recessive condition caused by
biallelic loss of function variants in TANGO2 on chromosome 22q11.21. Its
clinical course has a distinctive two part structure.

Between crises, patients have a chronic and relatively stable neurodevelopmental
phenotype: global developmental delay, intellectual disability, dysarthria, gait
disturbance, and hypothyroidism. During a crisis, typically triggered by fasting
or an intercurrent febrile illness, patients decompensate acutely with
rhabdomyolysis, elevated creatine kinase, metabolic acidosis, encephalopathy, and
life threatening ventricular arrhythmias including torsades de pointes. Crises
can be fatal.

Two clinical problems follow from that structure, and this software addresses
both.

**The diagnostic problem.** The chronic interictal phenotype is shared with many
neurogenetic conditions, so TANGO2 is often not suspected until a crisis occurs.
This matters because effective prophylaxis exists. B complex vitamin
supplementation, in particular pantothenate and folate, together with avoidance
of fasting, substantially reduces crisis frequency. A diagnosis delivered before
the first crisis is therefore not merely earlier. It is potentially life saving.

**The prognostic problem.** Even after diagnosis, clinicians cannot tell a family
when the next crisis is likely, so surveillance is uniform rather than targeted.

The asymmetry of consequences shapes every design decision in this tool. A false
positive costs a genetic test. A false negative costs a potentially fatal crisis
in a child for whom prophylaxis was available and simply not started.

---

## What the application does

The application has six pages.

### Overview

Background on the disorder and the two tier clinical pathway the research
proposes. Tier 1 uses two bedside markers as immediate referral triggers. Tier 2
uses the less specific neurological triad.

### Patient Assessment

The main clinical screen. A clinician records the Tier 1 markers and the baseline
phenotypes a child presents with, and receives:

- a probability that the pattern is consistent with TANGO2 deficiency
- a confidence band
- a plain language interpretation written for a clinician rather than a machine
  learning audience
- the specific phenotypes that contributed most to the result
- an immediate referral alert if a Tier 1 marker is recorded

### Crisis Forecast

The mechanistic model. A source amplitude is derived from baseline phenotype
burden, and the model produces an ensemble of simulated risk trajectories over a
900 day horizon. Four interactive views are available: the ensemble with
uncertainty bands, a single realisation with threshold crossings marked, a space
and time heat map showing where risk concentrates, and the bifurcation diagram
with the patient's position marked on it.

This entire page is labelled research use only. See [Limitations](#limitations).

### Model Insights

Receiver operating characteristic curve, calibration curve, feature importances,
and a table placing this model's metrics beside the published figures.

### Clinical Report

A downloadable PDF or CSV containing the patient summary, Tier 1 status, the
assessment and its interpretation, the contributing phenotypes, optionally the
crisis model output, the methodology, the model version and provenance, a
timestamp, and the scope and limitations. The scope statements are laid out with
the same visual weight as the results rather than relegated to a footnote.

### Documentation

Intended use, model assumptions, limitations, an interpretation guide, and
references, available inside the application so that a user never has to leave it
to find out what a number means.

---

## Architecture and why it was chosen

The application is a **Streamlit front end over a pure Python scientific core**,
with the scientific code having no knowledge that a user interface exists.

```
Presentation      app.py, src/tango2_forecast/ui/
                  Streamlit pages, Plotly figures, clinical theme
                            |
                            | imports, never the reverse
                            v
Domain            src/tango2_forecast/forecast/   the crisis model
                  src/tango2_forecast/ml/         the diagnostic classifier
                  src/tango2_forecast/reports/    PDF and CSV generation
                            |
                            v
Scope             src/tango2_forecast/disclaimer.py
                  every scope statement, defined once
```

**Why Streamlit rather than Flask, FastAPI, or a React front end.** A clinical
research tool of this size is dominated by scientific correctness, not by request
routing or state management. A Flask or FastAPI application would require a
separate front end, an API contract, a build step, and a deployment target, and
every one of those is a place where a number can silently diverge from the number
the science produces. Streamlit removes that entire surface. The same Python
object that comes out of the solver goes straight into the figure, so there is no
serialisation boundary at which a unit or a label can be lost. It also deploys
free and publicly from a GitHub repository in a few minutes, which matters for a
tool intended to be shown to clinicians and rare disease foundations rather than
run behind a corporate login.

**Why the scientific core is a separate installable package.** The dependency
arrow points one way. Nothing in `forecast/`, `ml/`, or `reports/` imports
Streamlit. This has three consequences that matter. The science can be unit
tested without a browser, and it is: 90 tests run in about 45 seconds. A
researcher can install the package and use the solver in a notebook without ever
launching the application. And if a future version needs a different front end,
the front end is the only thing that has to be rewritten.

**Why scope statements live in one module.** Every surface of this tool, the
library docstrings, the dashboard, the exported PDF, draws its scope language
from `disclaimer.py`. If a claim in the underlying research is weakened or
strengthened, it changes in one place and cannot drift between surfaces. This is
the single most important architectural decision in the project, because the
failure mode this software most needs to avoid is a result travelling without the
caveat that bounds it.

---

## Installation

Requires Python 3.12 or newer.

```bash
git clone https://github.com/YOUR-USERNAME/tango2-forecast.git
cd tango2-forecast
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
| --- | --- |
| numpy | Array computation throughout |
| scipy | Banded linear solves for the implicit scheme |
| pandas | Tabular data handling |
| pydantic | Validated, immutable model parameters |
| scikit-learn | Random Forest classifier and cross validation |
| plotly | Interactive figures |
| streamlit | Web application framework |
| reportlab | PDF report generation |
| pytest | Test suite |

### Using a virtual environment

Recommended, though not required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows the activation command is `.venv\Scripts\activate`.

---

## Running the application

```bash
streamlit run app.py
```

Your browser opens at `http://localhost:8501`.

### In PyCharm

1. File, then Open, then select this folder.
2. Choose a Python 3.12 or newer interpreter when prompted.
3. Click Install requirements on the banner that appears.
4. Open the terminal at the bottom of the window and run `streamlit run app.py`.

Do not press the green Run button on `app.py`. Streamlit applications must be
started with `streamlit run`. This is true of every Streamlit project and is not
specific to this one.

---

## Deploying publicly

This gives you a permanent public URL, at no cost.

1. Push this repository to GitHub as a public repository.
2. Go to `share.streamlit.io` and sign in with your GitHub account.
3. Click New app, select the repository, and set the main file path to `app.py`.
4. Click Deploy.

After a few minutes you have a link such as
`https://tango2-forecast.streamlit.app` that you can share with clinicians,
include in an application, or send to a rare disease foundation.

---

## Example workflow

A pediatric neurologist is evaluating a five year old with global developmental
delay and unexplained episodes of profound sleepiness.

1. **Open Patient Assessment.** Under Tier 1 referral triggers, tick
   **Paroxysmal lethargy**. An immediate alert appears: this phenotype appeared in
   a majority of TANGO2 patients and in none of 141 controls with other confirmed
   molecular diagnoses.

2. **Record the baseline phenotypes.** Tick dysarthria, motor delay, gait
   disturbance, intellectual disability, muscle weakness, and poor speech.

3. **Read the assessment.** A summary panel shows the model probability. Below it, a sentence in plain language, and a table of the findings that contributed most, expressed as a percentage share of the assessment basis.

4. **Optionally explore the crisis model.** Open Crisis Forecast. The patient's
   source amplitude is derived from their phenotype burden and their position
   relative to the critical amplitude is shown. Everything here is research use
   only and is labelled as such.

5. **Generate the report.** Open Clinical Report, enter a patient reference and
   your name, and download the PDF. The report carries the model version,
   provenance, timestamp, and the full scope statement.

---

## Interpretation guide

### The diagnostic probability

| Probability | Reading |
| --- | --- |
| Above 80 percent | Strongly consistent with TANGO2 deficiency. Supports referral for sequencing. |
| 50 to 80 percent | Moderately consistent. Genetic evaluation with TANGO2 in the differential is reasonable. |
| 20 to 50 percent | Not typical, but the model is not confident either way. Clinical judgement should lead. |
| Below 20 percent | Not consistent with TANGO2 deficiency. This does not exclude the diagnosis. |

Two things about this table are important.

**A Tier 1 marker should be acted on regardless of the probability shown.** The
two pathognomonic markers are not model inputs, by construction, because they
appear in zero control patients and therefore cannot be part of a shared
vocabulary feature set. They are reported separately and are arguably the more
directly useful finding.

**A low probability does not exclude TANGO2.** The model was trained on patients
who were deeply phenotyped after diagnosis. A child assessed early, with an
incomplete phenotype recorded, can score low and still have the disorder.

### The crisis model regime

The model divides patients at a critical source amplitude. Above it, crises recur
endogenously. Below it, a bounded sub threshold state exists.

**Below the critical amplitude does not mean safe.** In the underlying research,
83 percent of stochastic realisations just below the critical value still
contained at least one crisis. Noise alone drives a nominally sub threshold
patient across. The correct reading is lower probability, not safety, and this is
the model's direct argument against false reassurance for mildly affected
children.

---

## Explanation of every output

| Output | What it is | What it is not |
| --- | --- | --- |
| Model probability | The fraction of 500 decision trees voting for TANGO2, calibrated so that a stated 70 percent corresponds to roughly 70 percent observed frequency | A posterior probability in a clinic population, where prevalence is far lower |
| Confidence band | Whether the probability sits outside the ambiguous middle of the distribution | A confidence interval in the statistical sense |
| Contributing phenotypes | Present phenotypes ranked by mean decrease in Gini impurity | Causal contributions. Correlated phenotypes share credit, so these identify informative clusters of findings |
| AUC | Cross validated discrimination, every patient predicted by a model that never saw them | Expected accuracy in an unselected developmental clinic |
| Brier score | Calibration. The mean squared error of the predicted probabilities | A measure of discrimination |
| Source amplitude | The single patient specific parameter of the crisis model, derived from baseline phenotype burden | A measured biological quantity |
| Critical amplitude | The value separating quiescent from recurrent regimes in the model | A validated clinical threshold |
| Simulated crisis days | Threshold crossings in the model's dynamics | Predictions with demonstrated accuracy. No forecast has been compared against an observed crisis |
| Ensemble bands | The spread of outcomes across stochastic realisations | A confidence interval on a real world event |

---

## The science implemented

Everything below is implemented directly from the paper, with the section or
equation recorded in the relevant docstring.

### Diagnostic classifier

A Random Forest of 500 trees on a binary phenotype vector of 31 shared vocabulary
baseline HPO terms, with class weights inversely proportional to class frequency
to correct the cohort imbalance. Evaluated by repeated stratified 10 fold cross
validation, so every patient is predicted by a model that never saw them in
training. Calibration is reported alongside discrimination using the Brier score,
because a probability that is merely rank ordered cannot responsibly inform a
referral decision.

Two safeguards from the research are preserved in the implementation. All 23
crisis defining phenotypes are excluded before fitting, so no result can reflect a
crisis that has already occurred. The feature set is restricted to terms used by
both annotating teams, which closes the most plausible route by which a strong
result could be an artefact of annotation style rather than biology.

### Crisis model

A stochastic reaction diffusion equation on a latent biomarker coordinate
indexing coupled physiological subsystems:

```
dr/dt = D d2r/dx2  +  alpha r (1 - r)  +  beta S(x)  -  delta r  +  noise
        diffusion     logistic growth    patient source  clearance
```

with no flux boundaries, a Gaussian source representing a constitutional
vulnerability locus, and multiplicative noise localised on the vulnerable band.
The noise prefactor vanishes at both ends of the risk range, which guarantees the
solution cannot leave the unit interval.

A crisis is declared when peak risk reaches the threshold, after which a partial
reset removes 48 percent of supra resting risk while leaving the underlying
vulnerability unchanged. Crises therefore recur endogenously. No period is
imposed.

The equation is integrated with an implicit explicit scheme: diffusion implicit
for unconditional stability, reaction explicit so that no nonlinear solve is
needed. The resulting matrix is tridiagonal and diagonally dominant, so each step
costs O(N).

### Why a partial differential equation rather than an ordinary one

This is the design decision most worth understanding, and the software makes it
checkable rather than asking you to take it on faith.

Crisis risk is one number per patient, so an ordinary differential equation looks
sufficient. It is not. Decompensation begins in one physiological subsystem and
recruits others: metabolic instability under fasting stress propagates to muscle,
producing rhabdomyolysis, and then to the myocardium, producing arrhythmia. Risk
therefore has structure. It is concentrated in a vulnerable subsystem and spreads
to coupled ones. A scalar model collapses all subsystems into one number and
cannot distinguish a patient whose risk is concentrated and near threshold from
one whose identical total risk is spread harmlessly.

Because the source is localised, diffusion continuously leaks risk away from the
peak, so a bounded sub threshold state exists and a quiescent patient is possible.
In the well mixed limit, under the paper's convention, every patient in the
calibrated range eventually crosses threshold, which is clinically false. The
spatial term is not decoration. It is what makes the mild patient possible.

---

## Verification against the paper

The software recomputes the paper's numerical results from source rather than
asserting them. Run `python main.py` to regenerate this table.

| Check | Published | This implementation |
| --- | --- | --- |
| Spatial convergence order | 1.05, 1.10, 1.23, 1.59 | 1.04, 1.10, 1.22, 1.58 |
| Temporal convergence order | 1.05, 1.10, 1.22 | 1.05, 1.10, 1.22 |
| Mass drift, pure diffusion | 1.4e-12 | 5.5e-12 |
| von Neumann amplification | 1.000000 | 1.000000 |
| Critical amplitude | 0.01273 | 0.012701 |
| First crisis at high severity | day 229 plus or minus 7 | day 229 plus or minus 8 |
| First crisis at mild severity | day 347 plus or minus 17 | day 346 plus or minus 18 |
| Probability of a crisis below the critical amplitude | 0.83 | 0.82 |
| Shift in critical amplitude, all eight parameters | 28.1, 16.3, -8.1, -9.0, 0.0, 2.4, -2.3, 0.0 | 28.14, 16.34, -8.12, -8.99, 0.00, 2.39, -2.30, 0.00 |

Two notes recorded in the interest of accuracy rather than buried.

**The critical amplitude differs by 0.23 percent.** Grid refinement drives this
implementation to 0.012681, so the residual gap is a discretisation and bisection
tolerance difference rather than a modelling one. For scale, the paper's own
sensitivity analysis moves this quantity by 16 percent under a 10 percent change
in the clearance rate.

**Elasticities depend on which patient they describe.** The crisis count sits in
the denominator of the elasticity definition, so a mildly affected patient has few
crises for a perturbation to move. Measured here, the threshold elasticity runs
from about -8.7 at the bottom of the calibrated range to -5.3 at the top. The
published values are reproduced within Monte Carlo error at the top of the range,
which is this implementation's documented default.

---

## A note on data

The cohorts behind the research, 90 molecularly confirmed TANGO2 patients and 141
Undiagnosed Diseases Network controls, were provided under institutional agreement
by Baylor College of Medicine and are not publicly redistributable.

This repository therefore ships two things instead of data.

**A fitting and evaluation harness** that reproduces the published analysis when
pointed at the real cohorts.

**A synthetic cohort generator** built from the published prevalences, so the
application can be demonstrated, tested, and deployed without any patient record.

Every model fitted to synthetic data carries a provenance field, and the application header and Model Insights page indicate when a development dataset is in use. This labelling is deliberate and should not be removed. A plausible looking AUC produced from simulated data is the single most misleading artefact this software could generate.

The generator does not sample phenotypes independently. Independent sampling would
make the classes almost perfectly separable and produce a flattering, meaningless
result. Instead each simulated patient draws a latent burden that shifts every
phenotype together, so phenotypes co occur within a patient, mildly affected
simulated patients present incompletely, and the classification problem has the
overlapping structure the real one has.

When you have access to the real cohorts,
`train_model(cohort, provenance="institutional")` swaps them in and the warnings
disappear on their own.

---

## Repository structure

```
tango2-forecast/
    app.py                        The web application. Run with streamlit run app.py
    main.py                       Regenerates every number in the paper from source
    requirements.txt
    pyproject.toml                Packaging, linting, and type checking configuration
    README.md
    START_HERE.md                 Short setup guide
    CHANGELOG.md
    LICENSE

    src/tango2_forecast/
        disclaimer.py             Every scope statement, defined once
        cli.py                    The verification command

        forecast/                 Module A, the crisis model
            parameters.py         Validated parameters with provenance per field
            operators.py          Discrete Laplacian and tridiagonal solvers
            solver.py             The implicit explicit integrator
            steady_state.py       Steady states and the critical amplitude
            severity.py           Phenotype burden to source amplitude
            sensitivity.py        Parameter elasticity analysis
            verification.py       Convergence, conservation, and stability

        ml/                       Module B, the diagnostic classifier
            cohort.py             Phenotype vocabulary and synthetic generator
            classifier.py         Random Forest, evaluation, and provenance

        reports/                  Module C, clinical reports
            pdf.py                Professional PDF generation

        ui/                       Presentation layer
            theme.py              Clinical palette and Plotly template
            figures.py            Interactive figure builders

    tests/                        90 tests
    docs/                         Extended documentation
    examples/                     Runnable usage examples
    .github/workflows/ci.yml      Continuous integration
```

---

## Testing

```bash
pytest
```

90 tests, about 45 seconds. The suite is unusual in one respect worth noting: it
asserts against the published numbers in the paper rather than only against
internal consistency. A test failure therefore means either the software or the
reproduction has changed, not merely that an implementation detail moved.

Continuous integration runs the suite on Ubuntu, macOS, and Windows across Python
3.12 and 3.13, along with linting, formatting, and type checking, and separately
regenerates the paper's verification tables on every push.

---

## Limitations

These are carried directly from the research and are surfaced inside the
application as well as here.

**The crisis model has not been validated against outcomes.** This is the most
important limitation. The source dataset contains no crisis dates, so the model's
parameters were calibrated to reproduce qualitative published behaviour rather
than fitted to observed events. No forecast in this software has ever been
compared against a real crisis. The solver is verified, which is a strictly
different claim from the model being right. Any use of a specific predicted day as
a clinical decision point would misuse the model.

**Classifier performance will not transfer unchanged.** Both cohorts are
specialist referred and deeply phenotyped, and the control group, while
adversarial in phenotype, is not a clinic population. In an unselected
developmental clinic the prevalence of TANGO2 deficiency is far lower and
phenotyping far less complete, so both positive predictive value and
discrimination will be lower than the published figures.

**Zero is not a zero rate.** Observing no false positives among 141 controls
bounds the rate only loosely. By the rule of three, the one sided 95 percent upper
bound is approximately 2.1 percent.

**Sample size.** With 90 TANGO2 patients, cross validated estimates carry real
uncertainty despite their tight spread. Near perfect scores indicate a strong and
consistent signal, not near perfect real world accuracy.

**The crisis model is treatment naive and trigger naive.** It omits B complex
supplementation, which substantially reduces crisis frequency, so it describes an
untreated baseline and will over predict crises in adequately supplemented
patients. Specific triggers such as fasting or febrile illness are folded into a
generic noise term rather than represented explicitly.

**Residual annotation bias.** The shared vocabulary restriction removes the
crudest form of annotation bias, but it cannot remove the fact that TANGO2
patients were annotated after diagnosis, potentially with greater attention to
relevant findings. Only prospective testing on undiagnosed patients can eliminate
this.

---

## Future improvements

- PDF reports with embedded figures rather than tabular summaries alone
- SHAP values alongside Gini importance, for per patient rather than global
  attribution
- A FHIR or HL7 import path so phenotypes can be pulled from a record rather than
  entered by hand
- Fitting the crisis model to longitudinal crisis dates once such data exist, at
  which point the model could be scored honestly by a Brier score for crisis
  occurrence within a horizon
- Explicit representation of B complex supplementation as a reduction in the
  source amplitude or an increase in the clearance rate, which the research
  identifies as a concrete falsifiable prediction
- Batch assessment for cohort level research use
- Internationalisation of the clinical interface

---

## Contributing

Contributions are welcome, particularly from clinicians who can tell us where the
interface misleads.

1. Fork the repository and create a branch.
2. Run `ruff check .`, `ruff format .`, and `pytest` before opening a pull
   request.
3. Any change touching the scientific core must include a test asserting against
   the paper where the paper provides a number.
4. Scope and limitation language lives only in `src/tango2_forecast/disclaimer.py`.
   Please do not duplicate it elsewhere or weaken it.

---

## Citation

If you use this software, please cite the underlying research.

```bibtex
@unpublished{tippimath2025tango2,
  author = {Tippimath, Abhay and Lalani, Seema R. and Liu, Zhandong},
  title  = {A Hybrid Random Forest and Reaction Diffusion Framework for Early
            Identification and Crisis Forecasting in TANGO2 Deficiency Disorder},
  note   = {Preprint},
  year   = {2025}
}
```

### Key references

Lalani, S. R., Liu, P., Rosenfeld, J. A., et al. (2016). Recurrent muscle weakness
with rhabdomyolysis, metabolic crises, and cardiac arrhythmia due to bi-allelic
TANGO2 mutations. American Journal of Human Genetics, 98(2), 347 to 357.

Dines, J. N., Golden-Grant, K., LaCroix, A., et al. (2019). TANGO2 deficiency:
expanding the clinical phenotype and spectrum of disease. Genetics in Medicine,
21(3), 601 to 607.

Kohler, S., Carmody, L., Vasilevsky, N., et al. (2019). Expansion of the Human
Phenotype Ontology knowledge base and resources. Nucleic Acids Research, 47(D1),
D1018 to D1027.

Ascher, U. M., Ruuth, S. J., and Wetton, B. T. R. (1995). Implicit explicit methods
for time dependent partial differential equations. SIAM Journal on Numerical
Analysis, 32(3), 797 to 823.

Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5 to 32.

---

## Disclaimer

**TANGO2-Forecast is research software. It is not a medical device.**

It has not been reviewed or cleared by any regulator, and it must not be used as
the sole basis for any diagnostic or treatment decision. All clinical decisions
remain the responsibility of a qualified clinician.

The diagnostic component is intended to support one decision, whether to refer a
child for TANGO2 sequencing. It does not confirm a diagnosis and does not exclude
one. Molecular testing remains the only route to a diagnosis.

The crisis model component is hypothesis generating and is marked research use
only throughout the application. It has never been validated against observed
crisis outcomes.

---

## License

MIT. See [LICENSE](LICENSE).
