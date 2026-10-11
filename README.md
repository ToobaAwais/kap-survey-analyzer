# KAP Survey Analyzer

A web app that takes a raw **Knowledge, Attitude and Practice (KAP)** questionnaire export and returns publication-ready tables, figures and a drafted Methods and Results section.

Built by **Tooba Mujtaba** · NovaGen Bioinformatics

## What it does

| Step | Output |
|---|---|
| Scoring | Answer-key scoring for knowledge items (correct = 1, incorrect/don't know = 0); Likert scoring with editable label-to-value mapping and reverse-scored items |
| Classification | Bloom's cut-off (≥80% / 60–79% / <60%), custom cut-offs, or median/mean split, with editable level names |
| Descriptives | Participant characteristics table, item-level response tables, score summaries |
| Reliability | KR-20 / Cronbach's α with Feldt 95% CI, corrected item–total correlations, α if item deleted |
| Associations | Pearson χ²; Fisher's exact (2×2) or Monte Carlo Fisher–Freeman–Halton when expected counts are small; Cramér's V |
| Group comparisons | t-test / ANOVA or Mann–Whitney U / Kruskal–Wallis, chosen automatically by Shapiro–Wilk or set by the user, with effect sizes |
| Correlations | Spearman ρ between domains and with continuous characteristics |
| Regression | Binary logistic regression with crude (COR) and adjusted (AOR) odds ratios, 95% CI, model χ², Nagelkerke R², events-per-variable and separation checks |
| Report | Word report (three-line tables, figures, Methods and Results drafts), Excel workbook of every table, scored dataset |

## Data format

One row per respondent, one column per question, original answers kept as text (e.g. `True / False / Don't know`, `Strongly agree … Strongly disagree`, `Never … Always`) or numeric codes. CSV or Excel (choose the sheet if there are several). Columns named `K1`, `A1`, `P1` (or `Knowledge_1` …) are detected automatically; any other names can be selected manually.

See `sample_data/demo_kap_survey.csv` (simulated, 240 respondents) and its codebook.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. On share.streamlit.io choose **Create app → Deploy a public app from GitHub**, select the repository, branch `main`, main file `app.py`.
3. Under **Advanced settings** choose Python 3.12, then **Deploy**.

## Project structure

```
app.py           Streamlit interface
kap_core.py      Statistics engine (pandas/NumPy/SciPy; usable on its own in Colab)
reporting.py     Figures, Word and Excel export
demo_data.py     Simulated demo dataset
sample_data/     Demo CSV and codebook
```

## Validation

The statistics engine was checked against reference implementations: logistic regression coefficients match scikit-learn's unpenalised fit (difference < 1e-6) and the closed-form 2×2 odds ratio and standard error; KR-20 matches the textbook formula; χ², Mann–Whitney and Kruskal–Wallis results match SciPy.

## Privacy

Uploaded data is processed in memory for the session and is not saved by the app.
