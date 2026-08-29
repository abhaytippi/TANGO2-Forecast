# Data

This directory is intentionally empty of patient data.

The cohorts behind the research, 90 molecularly confirmed TANGO2 patients and
141 Undiagnosed Diseases Network controls, were provided under institutional
agreement by Baylor College of Medicine and are not publicly redistributable.

The application generates a synthetic cohort at runtime from the published
prevalences instead. See `src/tango2_forecast/ml/cohort.py`.

If you have authorised access to the real cohorts, place them here and pass them
to `train_model(cohort, provenance="institutional")`. Do not commit them.
