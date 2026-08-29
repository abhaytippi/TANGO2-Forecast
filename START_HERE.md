# Start here

This is a **web application**. You run one command and it opens in your browser.

---

## Option 1 — run it locally (2 minutes)

Open a terminal in this folder and type:

```
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens automatically at `http://localhost:8501`. That's the whole
setup.

### In PyCharm instead

1. **File → Open**, select this folder, click OK.
2. Choose a **Python 3.12 or newer** interpreter when prompted.
3. Click **Install requirements** on the yellow banner that appears (or open the
   terminal at the bottom and run `pip install -r requirements.txt`).
4. Open the terminal at the bottom of the PyCharm window and type:
   ```
   streamlit run app.py
   ```

Do **not** press the green ▶ Run button on `app.py`. Streamlit apps have to be
started with `streamlit run`, and pressing Run will print a warning instead of
launching. This is normal for every Streamlit project, not a problem with this
one.

---

## Option 2 — deploy it publicly and free (10 minutes)

This gives you a URL anyone can open. No server, no credit card.

1. Create a GitHub account if you don't have one, then create a new **public**
   repository called `tango2-forecast`.
2. Upload this entire folder to it. Either drag the files into GitHub's web
   uploader, or from the terminal:
   ```
   git init
   git add .
   git commit -m "TANGO2-Forecast platform"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/tango2-forecast.git
   git push -u origin main
   ```
3. Go to **share.streamlit.io** and sign in with GitHub.
4. Click **New app**, pick your repository, and set the main file path to
   **`app.py`**.
5. Click **Deploy**.

After a few minutes you get a permanent link like
`https://tango2-forecast.streamlit.app` that you can put in an application, send
to a clinician, or share with a foundation.

---

## What's in the app

Five pages in the left sidebar.

**Overview** — what the disorder is, and the two-tier clinical pathway.

**Patient Assessment** — the main clinical screen. Tick the Tier 1 bedside
markers and the baseline phenotypes a child presents with. The app returns a
probability gauge, a plain-language interpretation written for a clinician
rather than a machine learning audience, and the phenotypes that drove the
result. Ticking a Tier 1 marker fires an immediate referral alert.

**Crisis Forecast** — the reaction–diffusion model. Interactive ensemble
forecast, single trajectories, a space–time heatmap of where risk concentrates,
and the bifurcation diagram with the patient's position marked. Everything on
this page is labelled research use only.

**Model Insights** — ROC curve, calibration curve, feature importances, and a
table comparing this model against the published figures.

**Clinical Report** — an exportable summary with model version, provenance and
timestamp. CSV today; PDF is next.

---

## An important note about the data

The real cohorts (90 TANGO2 patients, 141 UDN controls) were provided under
institutional agreement and **cannot be redistributed**. So the app ships a
synthetic cohort generated from the published prevalences in Table 5, and the
model is labelled **SYNTHETIC** in red on every page that displays its output.

That labelling is deliberate and should stay. A plausible-looking AUC from
simulated data is the single most misleading thing this software could produce.
When you have access to the real cohorts, `train_model(cohort, provenance="institutional")`
swaps them in and the warnings disappear on their own.

---

## Other things you can run

```
python main.py     # regenerates every number in the paper's Section 4 and Table 6
pytest             # 90 tests, ~45 seconds
```

---

## If something goes wrong

- **`streamlit: command not found`** — the requirements didn't install. Run
  `pip install -r requirements.txt` again and read the output for errors.
- **Blank page or connection refused** — check the terminal; Streamlit prints
  the real URL there, occasionally on a port other than 8501.
- **Syntax error on line 1** — you're on Python 3.11 or older. This needs 3.12+.
- **PyCharm underlines imports in red but everything runs** — cosmetic.
  Right-click the `src` folder → Mark Directory As → Sources Root.
