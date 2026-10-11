"""
KAP Survey Analyzer - statistics engine.

Pure pandas / numpy / scipy. No Streamlit imports, so every function here can be
tested or reused in Colab on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Response dictionaries used to pre-fill Likert mappings
# ---------------------------------------------------------------------------
KNOWN_SCALES = {
    # agreement
    "strongly disagree": 1, "disagree": 2, "neutral": 3, "neither agree nor disagree": 3,
    "undecided": 3, "not sure": 3, "unsure": 3, "agree": 4, "strongly agree": 5,
    # frequency
    "never": 1, "rarely": 2, "seldom": 2, "sometimes": 3, "occasionally": 3,
    "often": 4, "usually": 4, "frequently": 4, "always": 5,
    # importance
    "not important": 1, "slightly important": 2, "moderately important": 3,
    "important": 4, "very important": 5,
    # three-point
    "yes": 3, "maybe": 2, "no": 1,
}

DEFAULT_LEVEL_LABELS = {
    "knowledge": ("Good", "Moderate", "Poor"),
    "attitude": ("Positive", "Neutral", "Negative"),
    "practice": ("Good", "Fair", "Poor"),
}

CORRECT_GUESSES = ["true", "yes", "correct", "1"]

# Three-point scales get 1-2-3 rather than their 5-point positions (1-3-5)
THREE_POINT = [
    {"never": 1, "sometimes": 2, "always": 3},
    {"disagree": 1, "neutral": 2, "agree": 3},
    {"disagree": 1, "not sure": 2, "agree": 3},
    {"disagree": 1, "undecided": 2, "agree": 3},
    {"no": 1, "maybe": 2, "yes": 3},
    {"no": 1, "not sure": 2, "yes": 3},
    {"rarely": 1, "sometimes": 2, "often": 3},
]


def norm(x) -> str | None:
    """Normalise a response for matching: strip, casefold, collapse spaces."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    # 1.0 -> 1 so numeric answer keys match
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return re.sub(r"\s+", " ", s).casefold()


def level_labels_for(name: str) -> tuple[str, str, str]:
    key = name.strip().lower()
    for k, v in DEFAULT_LEVEL_LABELS.items():
        if key.startswith(k[:4]):
            return v
    return ("High", "Moderate", "Low")


# ---------------------------------------------------------------------------
# Domain configuration
# ---------------------------------------------------------------------------
@dataclass
class DomainConfig:
    name: str
    items: list[str]
    method: str = "key"                 # "key" (correct answer) | "likert"
    key: dict[str, str] = field(default_factory=dict)        # item -> correct answer(s), '|' separated
    mapping: dict[str, float] = field(default_factory=dict)  # normalised label -> value (likert)
    reverse: list[str] = field(default_factory=list)
    scale_min: float = 1
    scale_max: float = 5
    cutoff_method: str = "bloom"        # bloom | two | one | median | mean
    cut_high: float = 80.0
    cut_low: float = 60.0
    labels: tuple[str, str, str] = ("Good", "Moderate", "Poor")
    pct_from_min: bool = False          # % = (score-min)/(max-min) instead of score/max

    # ----- derived -----
    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def max_possible(self) -> float:
        return self.n_items * (1 if self.method == "key" else self.scale_max)

    @property
    def min_possible(self) -> float:
        return 0 if self.method == "key" else self.n_items * self.scale_min

    @property
    def level_order(self) -> list[str]:
        """Levels from best to worst, as they will appear in tables."""
        hi, mid, lo = self.labels
        if self.cutoff_method in ("bloom", "two"):
            return [hi, mid, lo]
        return [hi, lo]


def guess_correct_answer(values: list[str]) -> str | None:
    normed = {norm(v): v for v in values if norm(v) is not None}
    for g in CORRECT_GUESSES:
        if g in normed:
            return normed[g]
    return None


def guess_mapping(labels: list[str]) -> dict[str, float | None]:
    """Pre-fill a label->value mapping from KNOWN_SCALES (None when unknown)."""
    out = {}
    for lab in labels:
        n = norm(lab)
        if n is None:
            continue
        try:
            out[lab] = float(n)
            continue
        except ValueError:
            pass
        out[lab] = KNOWN_SCALES.get(n)
    keys = {norm(l) for l in out}
    for scale in THREE_POINT:
        if keys == set(scale):
            return {l: float(scale[norm(l)]) for l in out}
    if keys <= {"yes", "no"}:
        out = {l: (1.0 if norm(l) == "yes" else 0.0) for l in out}
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass
class DomainResult:
    cfg: DomainConfig
    item_scores: pd.DataFrame       # respondents x items, float with NaN
    total: pd.Series                # NaN when any item missing
    pct: pd.Series
    level: pd.Series                # categorical
    n_scored: int
    n_excluded: int
    unmapped: dict[str, list[str]]  # item -> labels that had no numeric value
    cut_values: tuple[float | None, float | None]  # cut points actually used (% scale)


def score_domain(df: pd.DataFrame, cfg: DomainConfig) -> DomainResult:
    scores = pd.DataFrame(index=df.index)
    unmapped: dict[str, list[str]] = {}

    if cfg.method == "key":
        for item in cfg.items:
            answers = {norm(a) for a in str(cfg.key.get(item, "")).split("|") if norm(a)}
            # python lists avoid pandas string-dtype turning None back into NaN-as-text
            normed = [norm(v) for v in df[item].tolist()]
            s = [np.nan if v is None else (1.0 if v in answers else 0.0) for v in normed]
            scores[item] = pd.Series(s, index=df.index, dtype=float)
    else:
        nmap = {norm(k): float(v) for k, v in cfg.mapping.items() if norm(k) is not None and v is not None
                and not pd.isna(v)}
        for item in cfg.items:
            raw = df[item].tolist()
            normed = [norm(v) for v in raw]
            vals = [np.nan if v is None else nmap.get(v, _try_float(v)) for v in normed]
            bad = sorted({str(r).strip() for r, n, x in zip(raw, normed, vals) if n is not None and pd.isna(x)})
            if bad:
                unmapped[item] = bad
            vals = pd.Series(vals, index=df.index, dtype=float)
            if item in cfg.reverse:
                vals = (cfg.scale_min + cfg.scale_max) - vals
            scores[item] = vals

    complete = scores.notna().all(axis=1)
    total = scores.sum(axis=1).where(complete)
    if cfg.pct_from_min and cfg.max_possible > cfg.min_possible:
        pct = (total - cfg.min_possible) / (cfg.max_possible - cfg.min_possible) * 100
    else:
        pct = total / cfg.max_possible * 100 if cfg.max_possible else total * np.nan

    level, cuts = classify(pct, cfg)
    return DomainResult(cfg, scores, total, pct, level, int(complete.sum()),
                        int((~complete).sum()), unmapped, cuts)


def _try_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def classify(pct: pd.Series, cfg: DomainConfig):
    hi, mid, lo = cfg.labels
    m = cfg.cutoff_method
    out = pd.Series(pd.NA, index=pct.index, dtype="object")
    valid = pct.notna()
    if m in ("bloom", "two"):
        h, l = (80.0, 60.0) if m == "bloom" else (cfg.cut_high, cfg.cut_low)
        out[valid & (pct >= h)] = hi
        out[valid & (pct >= l) & (pct < h)] = mid
        out[valid & (pct < l)] = lo
        cuts = (h, l)
    else:
        if m == "one":
            c = cfg.cut_high
        elif m == "median":
            c = float(pct[valid].median())
        else:
            c = float(pct[valid].mean())
        out[valid & (pct >= c)] = hi
        out[valid & (pct < c)] = lo
        cuts = (c, None)
    cat = pd.Categorical(out, categories=cfg.level_order, ordered=True)
    return pd.Series(cat, index=pct.index, name=cfg.name), cuts


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "–"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def p_text(p) -> str:
    """'p = 0.012' or 'p < 0.001' for running text."""
    f = fmt_p(p)
    return f"p < {f[1:]}" if f.startswith("<") else f"p = {f}"


def n_pct(n, total, dp=1) -> str:
    if total == 0:
        return f"{n} (–)"
    return f"{int(n)} ({n / total * 100:.{dp}f})"


def mean_sd(x: pd.Series, dp=2) -> str:
    x = x.dropna()
    if len(x) == 0:
        return "–"
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return f"{x.mean():.{dp}f} ± {sd:.{dp}f}"


def median_iqr(x: pd.Series, dp=1) -> str:
    x = x.dropna()
    if len(x) == 0:
        return "–"
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:.{dp}f} ({q1:.{dp}f}–{q3:.{dp}f})"


def is_continuous(s: pd.Series, max_levels: int = 6) -> bool:
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() < 0.9 * s.notna().sum() or s.notna().sum() == 0:
        return False
    return num.nunique() > max_levels


# ---------------------------------------------------------------------------
# Descriptives
# ---------------------------------------------------------------------------
def demographics_table(df: pd.DataFrame, variables: list[str], continuous: list[str]) -> pd.DataFrame:
    rows = []
    N = len(df)
    for v in variables:
        s = df[v]
        if v in continuous:
            x = pd.to_numeric(s, errors="coerce")
            rows.append({"Variable": v, "Category": "Mean ± SD", "n (%)": mean_sd(x, 1)})
            rows.append({"Variable": "", "Category": "Median (IQR)", "n (%)": median_iqr(x, 1)})
            rows.append({"Variable": "", "Category": "Range", "n (%)":
                         f"{x.min():g}–{x.max():g}" if x.notna().any() else "–"})
        else:
            counts = s.dropna().astype(str).value_counts()
            counts = counts.reindex(_natural_order(counts.index))
            first = True
            for lev, n in counts.items():
                rows.append({"Variable": v if first else "", "Category": lev, "n (%)": n_pct(n, N)})
                first = False
        miss = int(s.isna().sum())
        if miss:
            rows.append({"Variable": "", "Category": "Missing", "n (%)": n_pct(miss, N)})
    return pd.DataFrame(rows)


def _natural_order(levels) -> list:
    """Sort levels so '20-29' < '30-39' and '<5' comes first; text alphabetically."""
    def key(l):
        m = re.search(r"\d+(\.\d+)?", str(l))
        lead = str(l).strip()[:1]
        bump = -0.5 if lead in "<≤" else (0.5 if lead in ">≥" else 0)
        return (0, float(m.group()) + bump, str(l)) if m else (1, 0, str(l).lower())
    return sorted(levels, key=key)


def item_table(df: pd.DataFrame, res: DomainResult) -> pd.DataFrame:
    cfg = res.cfg
    rows = []
    if cfg.method == "key":
        for item in cfg.items:
            s = res.item_scores[item]
            answered = int(s.notna().sum())
            correct = int((s == 1).sum())
            raw = df[item].map(norm)
            dk = int(raw.isin({"don't know", "dont know", "do not know", "not sure", "i don't know"}).sum())
            rows.append({
                "Item": item,
                "Correct answer": cfg.key.get(item, ""),
                "Answered (n)": answered,
                "Correct n (%)": n_pct(correct, answered),
                "Incorrect n (%)": n_pct(answered - correct - dk, answered),
                "Don't know n (%)": n_pct(dk, answered) if dk else "–",
                "% correct": round(correct / answered * 100, 1) if answered else np.nan,
            })
    else:
        # ordered response labels by their mapped value
        labels = _ordered_labels(df, cfg)
        for item in cfg.items:
            raw = df[item]
            answered = int(raw.map(norm).notna().sum())
            row = {"Item": item + (" (R)" if item in cfg.reverse else ""), "Answered (n)": answered}
            for lab in labels:
                n = int((raw.map(norm) == norm(lab)).sum())
                row[lab] = n_pct(n, answered)
            row["Mean ± SD"] = mean_sd(res.item_scores[item], 2)
            rows.append(row)
    return pd.DataFrame(rows)


def _ordered_labels(df, cfg) -> list[str]:
    seen = {}
    for item in cfg.items:
        for v in df[item].dropna().unique():
            n = norm(v)
            if n is None or n in {norm(k) for k in seen}:
                continue
            val = {norm(k): x for k, x in cfg.mapping.items()}.get(n, _try_float(n))
            seen[str(v).strip()] = val if val is not None else np.nan
    return sorted(seen, key=lambda k: (np.isnan(seen[k]) if isinstance(seen[k], float) else False,
                                       seen[k] if not pd.isna(seen[k]) else 0))


def domain_summary(results: list[DomainResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        cfg = r.cfg
        row = {
            "Domain": cfg.name,
            "Items": cfg.n_items,
            "Possible range": f"{cfg.min_possible:g}–{cfg.max_possible:g}",
            "n scored": r.n_scored,
            "Score mean ± SD": mean_sd(r.total, 2),
            "% score mean ± SD": mean_sd(r.pct, 1),
        }
        counts = r.level.value_counts()
        order = cfg.level_order
        row["High level n (%)"] = f"{order[0]}: " + n_pct(int(counts.get(order[0], 0)), r.n_scored)
        row["Middle level n (%)"] = (f"{order[1]}: " + n_pct(int(counts.get(order[1], 0)), r.n_scored)
                                     if len(order) == 3 else "–")
        row["Low level n (%)"] = f"{order[-1]}: " + n_pct(int(counts.get(order[-1], 0)), r.n_scored)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------
def cronbach_alpha(items: pd.DataFrame) -> dict:
    X = items.dropna()
    n, k = X.shape
    if k < 2 or n < 3:
        return {"alpha": np.nan, "n": n, "k": k, "ci": (np.nan, np.nan), "items": pd.DataFrame()}
    alpha = _alpha(X.values)
    # Feldt (1965) 95% CI
    df1, df2 = n - 1, (n - 1) * (k - 1)
    lo = 1 - (1 - alpha) * stats.f.ppf(0.975, df1, df2)
    hi = 1 - (1 - alpha) * stats.f.ppf(0.025, df1, df2)
    total = X.sum(axis=1)
    rows = []
    for c in X.columns:
        rest = total - X[c]
        r = X[c].corr(rest) if X[c].std() > 0 and rest.std() > 0 else np.nan
        a_del = _alpha(X.drop(columns=c).values) if k > 2 else np.nan
        rows.append({"Item": c, "Mean": round(X[c].mean(), 2), "SD": round(X[c].std(ddof=1), 2),
                     "Corrected item–total r": round(r, 3) if pd.notna(r) else np.nan,
                     "α if item deleted": round(a_del, 3) if pd.notna(a_del) else np.nan})
    return {"alpha": alpha, "n": n, "k": k, "ci": (lo, hi), "items": pd.DataFrame(rows)}


def _alpha(M: np.ndarray) -> float:
    k = M.shape[1]
    item_var = M.var(axis=0, ddof=1).sum()
    total_var = M.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return k / (k - 1) * (1 - item_var / total_var)


def alpha_label(a: float) -> str:
    if pd.isna(a):
        return "–"
    for cut, lab in [(0.9, "Excellent"), (0.8, "Good"), (0.7, "Acceptable"),
                     (0.6, "Questionable"), (0.5, "Poor")]:
        if a >= cut:
            return lab
    return "Unacceptable"


def reliability_table(results: list[DomainResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        a = cronbach_alpha(r.item_scores)
        stat = "KR-20" if r.cfg.method == "key" else "Cronbach's α"
        rows.append({"Domain": r.cfg.name, "Items": a["k"], "n": a["n"], "Statistic": stat,
                     "Coefficient": f"{a['alpha']:.3f}" if pd.notna(a["alpha"]) else "–",
                     "95% CI": f"{a['ci'][0]:.3f}–{a['ci'][1]:.3f}" if pd.notna(a["alpha"]) else "–",
                     "Interpretation": alpha_label(a["alpha"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Associations: level x characteristic
# ---------------------------------------------------------------------------
def _chi2_stat(table: np.ndarray) -> float:
    rs, cs, N = table.sum(1, keepdims=True), table.sum(0, keepdims=True), table.sum()
    exp = rs @ cs / N
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum((table - exp) ** 2 / np.where(exp == 0, np.nan, exp)))


def monte_carlo_exact(a_codes: np.ndarray, b_codes: np.ndarray, B: int = 10000, seed: int = 2024) -> float:
    """Monte Carlo estimate of the Fisher-Freeman-Halton exact test (chi-square statistic,
    margins fixed by permutation)."""
    a_codes, b_codes = np.asarray(a_codes, dtype=np.int64), np.asarray(b_codes, dtype=np.int64)
    ka, kb = a_codes.max() + 1, b_codes.max() + 1
    cells = ka * kb
    obs = np.bincount(a_codes * kb + b_codes, minlength=cells).reshape(ka, kb)
    # margins are fixed under permutation, so expected counts are constant
    exp = obs.sum(1, keepdims=True) @ obs.sum(0, keepdims=True) / obs.sum()
    exp_safe = np.where(exp == 0, np.inf, exp)
    stat = float(((obs - exp) ** 2 / exp_safe).sum())
    rng = np.random.default_rng(seed)
    exceed, done, n = 0, 0, len(b_codes)
    batch = max(1, min(2000, 2_000_000 // max(n, 1)))
    while done < B:
        m = min(batch, B - done)
        perm = np.argsort(rng.random((m, n)), axis=1)
        codes = a_codes[None, :] * kb + b_codes[perm]
        flat = (np.arange(m)[:, None] * cells + codes).ravel()
        tabs = np.bincount(flat, minlength=m * cells).reshape(m, ka, kb)
        t = ((tabs - exp) ** 2 / exp_safe).sum(axis=(1, 2))
        exceed += int((t >= stat - 1e-9).sum())
        done += m
    return (exceed + 1) / (B + 1)


def association_test(level: pd.Series, group: pd.Series) -> dict:
    d = pd.DataFrame({"level": level, "group": group}).dropna()
    d["group"] = d["group"].astype(str)
    d["level"] = d["level"].astype(str)
    ct = pd.crosstab(d["group"], d["level"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"test": "–", "stat": np.nan, "df": np.nan, "p": np.nan, "cramers_v": np.nan,
                "note": "fewer than two groups or levels", "n": len(d)}
    obs = ct.values
    chi2, p, dof, exp = stats.chi2_contingency(obs, correction=False)
    N = obs.sum()
    v = np.sqrt(chi2 / (N * (min(obs.shape) - 1)))
    sparse = (exp < 5).mean() > 0.2 or (exp < 1).any()
    if obs.shape == (2, 2) and sparse:
        _, p = stats.fisher_exact(obs)
        test = "Fisher's exact"
    elif sparse:
        a = pd.Categorical(d["group"]).codes
        b = pd.Categorical(d["level"]).codes
        p = monte_carlo_exact(a, b)
        test = "Fisher–Freeman–Halton (Monte Carlo)"
    else:
        test = "Pearson χ²"
    return {"test": test, "stat": chi2, "df": dof, "p": p, "cramers_v": v, "note": "", "n": int(N)}


TEST_MARK = {"Pearson χ²": "a", "Fisher's exact": "b", "Fisher–Freeman–Halton (Monte Carlo)": "c",
             "Independent t-test (Welch)": "d", "Mann–Whitney U": "e",
             "One-way ANOVA": "f", "Kruskal–Wallis H": "g", "Spearman ρ": "h"}


def association_table(df: pd.DataFrame, res: DomainResult, variables: list[str],
                      continuous: list[str], bins: dict[str, pd.Series] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Wide table: characteristic/level x domain level n (row %), p-value.
    Continuous variables are used through `bins` (a binned version) when given, else skipped."""
    order = res.cfg.level_order
    rows, tests = [], []
    for v in variables:
        if v in continuous:
            if not bins or v not in bins:
                continue
            g = bins[v]
        else:
            g = df[v]
        t = association_test(res.level, g)
        t["variable"] = v
        tests.append(t)
        d = pd.DataFrame({"level": res.level, "group": g}).dropna()
        d["group"] = d["group"].astype(str)
        ct = pd.crosstab(d["group"], d["level"]).reindex(columns=order, fill_value=0)
        ct = ct.reindex(_natural_order(ct.index))
        first = True
        for lev, r in ct.iterrows():
            tot = int(r.sum())
            row = {"Characteristic": v if first else "", "Category": lev, "n": tot}
            for o in order:
                row[f"{o} n (%)"] = n_pct(int(r[o]), tot)
            row["p-value"] = (fmt_p(t["p"]) + TEST_MARK.get(t["test"], "")) if first else ""
            rows.append(row)
            first = False
    return pd.DataFrame(rows), tests


# ---------------------------------------------------------------------------
# Scores by group
# ---------------------------------------------------------------------------
def group_test(score: pd.Series, group: pd.Series, method: str = "auto") -> dict:
    d = pd.DataFrame({"y": pd.to_numeric(score, errors="coerce"), "g": group}).dropna()
    d["g"] = d["g"].astype(str)
    groups = {k: v["y"].values for k, v in d.groupby("g") if len(v) >= 2}
    dropped = sorted(set(d["g"]) - set(groups))
    k = len(groups)
    out = {"groups": groups, "dropped": dropped, "test": "–", "stat": np.nan, "p": np.nan,
           "effect": np.nan, "effect_name": "", "parametric": None, "n": int(sum(len(x) for x in groups.values()))}
    if k < 2:
        return out
    if method == "auto":
        normal = all(stats.shapiro(x[:5000]).pvalue > 0.05 for x in groups.values() if len(x) >= 3)
        parametric = normal
    else:
        parametric = method == "parametric"
    out["parametric"] = parametric
    vals = list(groups.values())
    N = sum(len(x) for x in vals)
    if k == 2:
        a, b = vals
        if parametric:
            r = stats.ttest_ind(a, b, equal_var=False)
            sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
            out.update(test="Independent t-test (Welch)", stat=r.statistic, p=r.pvalue,
                       effect=(a.mean() - b.mean()) / sp if sp > 0 else np.nan, effect_name="Cohen's d")
        else:
            r = stats.mannwhitneyu(a, b, alternative="two-sided")
            out.update(test="Mann–Whitney U", stat=r.statistic, p=r.pvalue,
                       effect=1 - 2 * r.statistic / (len(a) * len(b)), effect_name="rank-biserial r")
    else:
        if parametric:
            r = stats.f_oneway(*vals)
            grand = np.concatenate(vals).mean()
            ssb = sum(len(x) * (x.mean() - grand) ** 2 for x in vals)
            sst = ((np.concatenate(vals) - grand) ** 2).sum()
            out.update(test="One-way ANOVA", stat=r.statistic, p=r.pvalue,
                       effect=ssb / sst if sst > 0 else np.nan, effect_name="η²")
        else:
            r = stats.kruskal(*vals)
            out.update(test="Kruskal–Wallis H", stat=r.statistic, p=r.pvalue,
                       effect=r.statistic / (N - 1), effect_name="ε²")
    return out


def score_by_group_table(df: pd.DataFrame, results: list[DomainResult], variables: list[str],
                         continuous: list[str], method: str = "auto",
                         bins: dict[str, pd.Series] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Wide table: characteristic/category x (score summary, p) for each domain.
    Summary is mean ± SD for parametric tests, median (IQR) otherwise.
    Continuous characteristics go to Spearman correlations (returned in the test list)."""
    rows_by_var: dict[str, list[dict]] = {}
    tests = []
    for v in variables:
        if v in continuous:
            for r in results:
                d = pd.DataFrame({"x": pd.to_numeric(df[v], errors="coerce"), "y": r.pct}).dropna()
                if len(d) >= 3:
                    rho, p = stats.spearmanr(d["x"], d["y"])
                    tests.append({"variable": v, "domain": r.cfg.name, "test": "Spearman ρ",
                                  "stat": rho, "p": p, "n": len(d), "effect": rho, "effect_name": "ρ"})
            if not bins or v not in bins:
                continue
            g_all = bins[v]
        else:
            g_all = df[v]
        gs = g_all.dropna().astype(str)
        cats = _natural_order(gs.unique())
        rows = [{"Characteristic": v if i == 0 else "", "Category": c, "n": int((gs == c).sum())}
                for i, c in enumerate(cats)]
        for r in results:
            t = group_test(r.pct, g_all, method)
            t.update(variable=v, domain=r.cfg.name)
            tests.append(t)
            for row in rows:
                x = pd.Series(t["groups"].get(row["Category"], []), dtype=float)
                summ = mean_sd(x, 1) if t["parametric"] else median_iqr(x, 1)
                row[f"{r.cfg.name} % score"] = summ
                row[f"{r.cfg.name} p"] = ""
            if rows:
                rows[0][f"{r.cfg.name} p"] = fmt_p(t["p"]) + TEST_MARK.get(t["test"], "")
        rows_by_var[v] = rows
    out = [row for v in rows_by_var for row in rows_by_var[v]]
    return pd.DataFrame(out), tests


def bin_continuous(s: pd.Series, method: str = "median", edges: list[float] | None = None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if method == "custom" and edges:
        edges = sorted(edges)
        bins = [-np.inf] + edges + [np.inf]
        labels = []
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            if np.isinf(lo):
                labels.append(f"<{hi:g}")
            elif np.isinf(hi):
                labels.append(f"≥{lo:g}")
            else:
                labels.append(f"{lo:g}–<{hi:g}")
        return pd.cut(x, bins, labels=labels, right=False).astype(object)
    med = x.median()
    return x.map(lambda v: np.nan if pd.isna(v) else (f"≥{med:g}" if v >= med else f"<{med:g}"))


# ---------------------------------------------------------------------------
# Correlations between domains
# ---------------------------------------------------------------------------
def domain_correlations(results: list[DomainResult]) -> pd.DataFrame:
    rows = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            d = pd.DataFrame({"a": a.pct, "b": b.pct}).dropna()
            if len(d) < 3:
                continue
            rho, p = stats.spearmanr(d["a"], d["b"])
            rows.append({"Pair": f"{a.cfg.name} – {b.cfg.name}", "n": len(d), "Spearman ρ": f"{rho:.3f}",
                         "p-value": fmt_p(p), "Strength": _rho_label(rho), "_p": p, "_rho": rho})
    return pd.DataFrame(rows)


def _rho_label(r: float) -> str:
    a = abs(r)
    s = ("very weak" if a < 0.2 else "weak" if a < 0.4 else "moderate" if a < 0.6
         else "strong" if a < 0.8 else "very strong")
    return f"{s} {'positive' if r > 0 else 'negative'}"


# ---------------------------------------------------------------------------
# Logistic regression (Newton-Raphson / IRLS, Wald CIs - same as SPSS & statsmodels)
# ---------------------------------------------------------------------------
@dataclass
class LogitFit:
    names: list[str]
    beta: np.ndarray
    se: np.ndarray
    p: np.ndarray
    n: int
    events: int
    llf: float
    ll0: float
    converged: bool
    separation: bool

    @property
    def lr_chi2(self):
        return 2 * (self.llf - self.ll0)

    @property
    def lr_df(self):
        return len(self.beta) - 1

    @property
    def lr_p(self):
        return stats.chi2.sf(self.lr_chi2, self.lr_df) if self.lr_df > 0 else np.nan

    @property
    def mcfadden(self):
        return 1 - self.llf / self.ll0 if self.ll0 != 0 else np.nan

    @property
    def nagelkerke(self):
        cs = 1 - np.exp(2 * (self.ll0 - self.llf) / self.n)
        mx = 1 - np.exp(2 * self.ll0 / self.n)
        return cs / mx if mx > 0 else np.nan


def fit_logit(X: np.ndarray, y: np.ndarray, names: list[str], max_iter: int = 100, tol: float = 1e-8) -> LogitFit:
    X = np.column_stack([np.ones(len(y)), X]) if X.size else np.ones((len(y), 1))
    names = ["(Intercept)"] + list(names)
    beta = np.zeros(X.shape[1])
    ybar = y.mean()
    beta[0] = np.log(ybar / (1 - ybar)) if 0 < ybar < 1 else 0.0
    converged = False
    H = None
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        g = X.T @ (y - p)
        H = X.T @ (X * W[:, None])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            raise ValueError("Predictors are collinear (a variable is a combination of others "
                             "or a category has no respondents). Remove or merge categories.")
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    eta = np.clip(X @ beta, -30, 30)
    p = 1 / (1 + np.exp(-eta))
    W = p * (1 - p)
    H = X.T @ (X * W[:, None])
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = beta / np.where(se == 0, np.nan, se)
    pv = 2 * stats.norm.sf(np.abs(z))
    eps = 1e-12
    llf = float(np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    ll0 = float(len(y) * (ybar * np.log(ybar + eps) + (1 - ybar) * np.log(1 - ybar + eps)))
    separation = (not converged) or bool(np.any(np.abs(beta[1:]) > 10)) or bool(np.any(se[1:] > 50))
    return LogitFit(names, beta, se, pv, len(y), int(y.sum()), llf, ll0, converged, separation)


def design_matrix(df: pd.DataFrame, predictors: list[str], continuous: list[str],
                  refs: dict[str, str]) -> tuple[pd.DataFrame, dict[str, list[tuple[str, str]]]]:
    """Dummy-code categorical predictors. Returns X and, for each predictor,
    the list of (column name, display level)."""
    cols = {}
    layout: dict[str, list[tuple[str, str]]] = {}
    for v in predictors:
        if v in continuous:
            cols[v] = pd.to_numeric(df[v], errors="coerce")
            layout[v] = [(v, "per unit increase")]
        else:
            s = df[v].astype("object").where(df[v].notna())
            s = s.map(lambda x: x if pd.isna(x) else str(x))
            levels = _natural_order(s.dropna().unique())
            ref = refs.get(v) if refs.get(v) in levels else (s.value_counts().idxmax() if s.notna().any() else None)
            layout[v] = [(None, f"{ref} (ref)")]
            for lev in levels:
                if lev == ref:
                    continue
                cname = f"{v}={lev}"
                cols[cname] = (s == lev).astype(float).where(s.notna())
                layout[v].append((cname, lev))
    return pd.DataFrame(cols, index=df.index), layout


def logistic_table(df: pd.DataFrame, y: pd.Series, predictors: list[str], continuous: list[str],
                   refs: dict[str, str], multivariable: bool = True) -> dict:
    """Crude (univariable) and adjusted (multivariable) odds ratios."""
    rows = []
    crude = {}
    for v in predictors:
        X, lay = design_matrix(df, [v], continuous, refs)
        d = pd.concat([X, y.rename("_y")], axis=1).dropna()
        if d["_y"].nunique() < 2 or len(d) < 5:
            continue
        try:
            f = fit_logit(d.drop(columns="_y").values, d["_y"].values.astype(float), list(X.columns))
        except ValueError:
            continue
        for i, nm in enumerate(f.names):
            crude[nm] = (f.beta[i], f.se[i], f.p[i], f.separation)
    adj, fit = {}, None
    warn = []
    if multivariable and predictors:
        X, lay_all = design_matrix(df, predictors, continuous, refs)
        d = pd.concat([X, y.rename("_y")], axis=1).dropna()
        if d["_y"].nunique() == 2:
            fit = fit_logit(d.drop(columns="_y").values, d["_y"].values.astype(float), list(X.columns))
            for i, nm in enumerate(fit.names):
                adj[nm] = (fit.beta[i], fit.se[i], fit.p[i])
            epv = min(fit.events, fit.n - fit.events) / max(1, fit.lr_df)
            if epv < 10:
                warn.append(f"Only {epv:.1f} events per predictor parameter (recommended ≥10); "
                            "adjusted estimates may be unstable - consider fewer predictors.")
            if fit.separation:
                warn.append("Possible (quasi-)complete separation: at least one category almost perfectly "
                            "predicts the outcome, so its odds ratio is unreliable. Merge sparse categories.")
    _, layout = design_matrix(df, predictors, continuous, refs)

    def orci(b, se):
        return f"{np.exp(b):.2f} ({np.exp(b - 1.96 * se):.2f}–{np.exp(b + 1.96 * se):.2f})"

    for v in predictors:
        first = True
        for cname, lev in layout[v]:
            row = {"Variable": v if first else "", "Category": lev}
            if cname is None:
                row.update({"COR (95% CI)": "1.00", "p (crude)": "", "AOR (95% CI)": "1.00" if fit else "",
                            "p (adjusted)": ""})
            else:
                c = crude.get(cname)
                a = adj.get(cname)
                row["COR (95% CI)"] = orci(c[0], c[1]) if c else "–"
                row["p (crude)"] = fmt_p(c[2]) if c else "–"
                row["AOR (95% CI)"] = orci(a[0], a[1]) if a else ("–" if fit else "")
                row["p (adjusted)"] = fmt_p(a[2]) if a else ("–" if fit else "")
                row["_p_adj"] = a[2] if a else np.nan
                row["_aor"] = np.exp(a[0]) if a else np.nan
                row["_lo"] = np.exp(a[0] - 1.96 * a[1]) if a else np.nan
                row["_hi"] = np.exp(a[0] + 1.96 * a[1]) if a else np.nan
                row["_p_crude"] = c[2] if c else np.nan
            rows.append(row)
            first = False
    return {"table": pd.DataFrame(rows), "fit": fit, "warnings": warn}


# ---------------------------------------------------------------------------
# Narrative text
# ---------------------------------------------------------------------------
def methods_text(results: list[DomainResult], group_method: str, alpha_level: float = 0.05,
                 logistic: dict | None = None) -> str:
    parts = []
    for r in results:
        c = r.cfg
        if c.method == "key":
            sc = (f"The {c.name.lower()} section comprised {c.n_items} items; each correct response scored 1 "
                  f"and each incorrect or \"don't know\" response scored 0 (possible range 0–{c.n_items}).")
        else:
            rev = (f" Negatively worded item{'s' if len(c.reverse) > 1 else ''} ({', '.join(c.reverse)}) "
                   f"{'were' if len(c.reverse) > 1 else 'was'} reverse-scored.") if c.reverse else ""
            sc = (f"The {c.name.lower()} section comprised {c.n_items} items rated on a {c.scale_min:g}–{c.scale_max:g} "
                  f"scale (possible range {c.min_possible:g}–{c.max_possible:g}).{rev}")
        parts.append(sc)
    base = "the minimum-to-maximum possible range" if any(r.cfg.pct_from_min for r in results) else "the maximum possible score"
    parts.append(f"Total scores were converted to percentages of {base}.")
    m = results[0].cfg.cutoff_method if results else "bloom"
    lv = results[0].cfg.labels if results else ("Good", "Moderate", "Poor")
    if all(r.cfg.cutoff_method == "bloom" for r in results):
        parts.append(f"Using Bloom's cut-off points, scores of 80–100% were classified as {lv[0].lower()}, "
                     f"60–79% as {lv[1].lower()}, and below 60% as {lv[2].lower()}.")
    else:
        for r in results:
            c = r.cfg
            h, l = r.cut_values
            if c.cutoff_method in ("bloom", "two"):
                parts.append(f"{c.name} scores ≥{h:g}% were classified as {c.labels[0].lower()}, {l:g}–<{h:g}% as "
                             f"{c.labels[1].lower()}, and <{l:g}% as {c.labels[2].lower()}.")
            else:
                basis = {"one": "a cut-off of", "median": "the median score of", "mean": "the mean score of"}[c.cutoff_method]
                parts.append(f"{c.name} was dichotomised at {basis} {h:.1f}% ({c.labels[0].lower()} vs "
                             f"{c.labels[2].lower()}).")
    parts.append("Internal consistency was assessed using the Kuder–Richardson 20 coefficient for "
                 "dichotomous items and Cronbach's α for Likert-type items, with 95% confidence intervals by "
                 "Feldt's method." if any(r.cfg.method == "key" for r in results) else
                 "Internal consistency was assessed using Cronbach's α with Feldt 95% confidence intervals.")
    parts.append("Categorical variables are presented as frequencies and percentages, and continuous variables "
                 "as mean ± standard deviation or median (interquartile range).")
    parts.append("Associations between participant characteristics and KAP levels were tested with Pearson's "
                 "chi-square test, or Fisher's exact test (2×2 tables) and the Monte Carlo Fisher–Freeman–Halton "
                 "test (larger tables) when more than 20% of expected cell counts were below 5.")
    gm = {"auto": "selected according to the Shapiro–Wilk test of normality: the independent-samples "
                  "t-test (Welch) or one-way ANOVA for normally distributed scores, and the Mann–Whitney U or "
                  "Kruskal–Wallis H test otherwise",
          "parametric": "the independent-samples t-test (Welch) or one-way ANOVA",
          "nonparametric": "the Mann–Whitney U or Kruskal–Wallis H test"}[group_method]
    lead = ("Percentage scores were compared across groups with tests " if group_method == "auto"
            else "Percentage scores were compared across groups using ")
    parts.append(f"{lead}{gm}. Correlations between scores, and between scores and continuous "
                 "characteristics, were assessed with Spearman's rank correlation coefficient (ρ).")
    if logistic and logistic.get("fit") is not None:
        parts.append(f"Binary logistic regression was used to identify factors associated with "
                     f"{logistic['outcome_label']}; crude (COR) and adjusted odds ratios (AOR) are reported with "
                     "95% confidence intervals.")
    parts.append(f"A two-sided p-value < {alpha_level:g} was considered statistically significant.")
    return " ".join(parts)


def results_text(N: int, results: list[DomainResult], assoc_tests: dict[str, list[dict]],
                 corr: pd.DataFrame, logistic: dict | None, alpha_level: float = 0.05) -> list[str]:
    paras = []
    for r in results:
        c = r.cfg
        counts = r.level.value_counts()
        bits = [f"{counts.get(l, 0)} ({counts.get(l, 0) / r.n_scored * 100:.1f}%) {l.lower()}"
                for l in c.level_order] if r.n_scored else []
        if bits:
            bits[0] = bits[0].replace(") ", ") had ", 1)
        lv = _join(bits)
        excl = (f" ({r.n_excluded} respondent{'s' if r.n_excluded > 1 else ''} with incomplete responses "
                f"{'were' if r.n_excluded > 1 else 'was'} excluded)") if r.n_excluded else ""
        paras.append(f"{c.name}: the mean score was {r.total.mean():.2f} ± {r.total.std():.2f} out of "
                     f"{c.max_possible:g} (mean {r.pct.mean():.1f}%) among {r.n_scored} respondents{excl}. "
                     f"Overall, {lv} {c.name.lower()}.")
        sig = [t for t in assoc_tests.get(c.name, []) if pd.notna(t["p"]) and t["p"] < alpha_level]
        if sig:
            names = _join([f"{t['variable']} ({p_text(t['p'])})" for t in sig])
            paras.append(f"{c.name} level was significantly associated with {names}.")
        elif assoc_tests.get(c.name):
            paras.append(f"No participant characteristic was significantly associated with {c.name.lower()} level.")
    if corr is not None and len(corr):
        bits = []
        for _, r in corr.iterrows():
            bits.append(f"{r['Pair'].replace(' – ', ' and ')} (ρ = {r['_rho']:.3f}, {p_text(r['_p'])}; "
                        f"{r['Strength']})")
        paras.append("Pairwise Spearman correlations between domain scores: " + _join(bits) + ".")
    if logistic and logistic.get("fit") is not None:
        t = logistic["table"]
        sig = t[(t.get("_p_adj").notna()) & (t["_p_adj"] < alpha_level)] if "_p_adj" in t else t.iloc[0:0]
        if len(sig):
            items = []
            cur_var = None
            for idx in t.index:
                if t.loc[idx, "Variable"]:
                    cur_var = t.loc[idx, "Variable"]
                if idx in sig.index:
                    ref = _ref_of(t, idx)
                    vs = f" vs {ref}" if ref else ""
                    items.append(f"{cur_var} ({t.loc[idx, 'Category']}{vs}: AOR {t.loc[idx, '_aor']:.2f}, 95% CI "
                                 f"{t.loc[idx, '_lo']:.2f}–{t.loc[idx, '_hi']:.2f}, {p_text(t.loc[idx, '_p_adj'])})")
            paras.append(f"In multivariable logistic regression, {logistic['outcome_label']} was independently "
                         "associated with " + _join(items) + ".")
        else:
            paras.append(f"In multivariable logistic regression, no characteristic was independently associated "
                         f"with {logistic['outcome_label']}.")
    return paras


def _join(items: list[str]) -> str:
    """'a', 'a and b', 'a, b, and c' (semicolons when items contain commas)."""
    items = [i for i in items if i]
    if len(items) <= 2:
        return " and ".join(items)
    sep = "; " if any("," in i for i in items) else ", "
    return sep.join(items[:-1]) + sep + "and " + items[-1]


def _ref_of(t: pd.DataFrame, idx) -> str | None:
    """Find the '(ref)' row of the variable block that contains row idx."""
    pos = list(t.index).index(idx)
    for j in range(pos, -1, -1):
        cat = str(t.iloc[j]["Category"])
        if cat.endswith(" (ref)"):
            return cat[:-6]
        if t.iloc[j]["Variable"]:
            break
    return None


def continuous_corr_table(tests: list[dict]) -> pd.DataFrame:
    rows = [{"Characteristic": t["variable"], "Domain": t["domain"], "n": t["n"],
             "Spearman ρ": f"{t['stat']:.3f}", "p-value": fmt_p(t["p"])}
            for t in tests if t["test"] == "Spearman ρ"]
    return pd.DataFrame(rows)
