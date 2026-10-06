# Reproducing the manuscript

This folder regenerates every reaction diffusion (PDE) number and Figures 8, 9
and 10 of

> Tippimath, A. (2026). *Dynamic Mathematical PDE Modeling and Machine Learning
> for TANGO2 Identification and Crisis Prediction.* IJSCAR 1(1).

```bash
pip install numpy scipy matplotlib pandas openpyxl
python paper/reproduce_pde.py     # one run, fixed seeds, about 25 to 40 minutes
python paper/make_figures.py      # Figures 8, 9, 10 -> paper/results/
```

`python paper/reproduce_pde.py --quick` skips the slow critical amplitude
sweep (about 5 minutes).

## What is in here

| File | Purpose |
| --- | --- |
| `pde_model.py` | The exact solver used for the paper (Section 4.3 to 4.6). |
| `reproduce_pde.py` | Runs every PDE analysis once with fixed seeds and prints each number next to its section. |
| `make_figures.py` | Draws Figures 8, 9 and 10 from the saved results. |
| `cohort_data.py` | Feature construction for the restricted cohorts (used only if the data are present). |
| `results/pde_results.json` | Output of the run reported in the manuscript. |

The resting level used by the post crisis reset is `r_rest = 0` (Table 2).

## Expected output

```
Sec 4.5  beta* = 0.01273
Sec 5.8  beta* with r* +20% = 0.0202, r* -20% = 0.0065
Sec 5.8  x0 +-10% shifts beta* by at most 2.3 percent
Sec 5.5  high 4.79 +- 0.40, first ~day 229; moderate 4.00, first ~day 256; mild 2.76 +- 0.43, first ~day 347
Sec 5.6  beta=0.0121: 83% of realisations with a crisis, mean 0.83, max per realisation 1
Sec 5.7  scalar peak 26.9 to 40.9; scalar average 0.0 to 1.4; PDE 0.8 to 5.0
Sec 5.7  equal total source: 4.5, 4.0, 4.0, 3.4, 3.0
Sec 5.7  recovery r = 0.992, MAE = 1.6e-04, median rel. error = 0.9%, regime correct 60/60
Sec 5.8  elasticities: r_star -5.05, delta -2.90, sigma +2.47, alpha +1.44, x0 +0.51, kappa +0.79, D -0.78, sigma_eta +0.06
Sec 5.8  temporal orders 1.01 to 1.10; spatial orders 1.02, 1.09, 1.22, 1.58; mass drift 1.3e-12; max |g| = 1.000000
Sec 5.5  correlations: count r = 0.97, first crisis r = -0.90     (only with the restricted data)
```

The mild stratum's mean first crisis day is 346.6, reported in the paper as
"near day 346".

## Seeds

All seeds are listed at the top of `reproduce_pde.py`: severity strata 3000;
scalar baselines 11 (PDE), 12 (scalar peak), 13 (scalar average); equal total
source 21; Figure 8b 700 + i; per patient correlations 5000 + i; elasticities 1
(40 realisations, base beta 0.0167); recovery reference 0 + j, true values 0,
synthetic patients 1000 + i.

## Data

No patient data is stored in this repository. The one analysis that uses the
cohort (the correlation between severity and simulated crisis count across the
90 patients) runs only when the restricted Baylor spreadsheets are placed in
`data/`. All other results need no data.

## Relation to the interactive application

The application in `src/` implements the same model for interactive use, with
its own grid (dx = 1/(N-1)) and seeds, so `tango2-verify` agrees with the
manuscript within Monte Carlo error rather than digit for digit. The scripts in
this folder are the reference for the published numbers.
