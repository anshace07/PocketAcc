"""
anomaly.py - STEP 8 (traditional ML): flag unusual transactions with Isolation Forest.

Unsupervised: no anomaly labels are needed to train. Each transaction becomes a feature vector
that describes it RELATIVE TO ITS OWN ACCOUNT, so one model works for a student and a trader:

  log_amount      size of the transaction
  is_credit       money in or out
  day_of_month    1..31 (scaled)
  category_z      how far the amount is from this account's usual amount for this category
                  (robust z-score: median / MAD of log-amounts, per account and category)
  rarity          how rare this category is for this account
  vs_income       amount / this account's typical monthly inflow (log)
  same_day_dup    other DEBITS with the same amount on the same date (double debit; repeated
                  same-amount credits are normal for shops receiving UPI, so credits are not counted;
                  a same-day pair that recurs in 2+ months is a standing instruction, e.g. two SIPs)
  round_large     a large round figure (>= 10,000 and a multiple of 1,000)

Category comes from the transaction classifier (classifier.TxnClassifier), as in the app.

Deployed detector = HYBRID: Isolation Forest (unusual combinations) OR two transparent rules that
cover its blind spots:
  * duplicate debit : same amount debited twice on one day. Clean statements contain no such
                      rows, so the feature is constant in training and a forest can never split on it.
  * robust z > 3.5  : the amount is far above this account's usual for the category (modified
                      z-score, Iglewicz & Hoaglin 1993). Forests are weak at points that are extreme
                      in only one feature.

Evaluation (no real labelled anomalies exist, so we inject them - like the corrector simulation):
  leave-one-statement-out on the original 20 six-month statements: fit on 19 clean statements
  (no labels used), test on the 20th + ~1% injected anomalies of 4 kinds; mean of 5 seeds
          amount_spike        a merchant/bill/ATM debit 3-10x bigger than the account's largest
                              ever debit of that category
          large_new_transfer  IMPS/NEFT out of 0.8-2.5x the account's monthly inflow
          duplicate_debit     an existing debit repeated on the same day
          big_cash_deposit    round cash deposit of 1.5-4x the monthly inflow
  baselines: (a) 3-sigma rule on amount per account, (b) Isolation Forest on raw features only
             (amount, direction, day, category one-hot) - no account context, (c) Isolation Forest
             alone on our features
  (A first version trained on the 140 generated statements: they are ~1 month long, so per-account
   averages were noisy and big normal payments looked extreme -> 1.8% false alarms. See FAILURES.md.)

Usage:
    python src/anomaly.py --out results/anomaly.json
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from classifier import TxnClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["log_amount", "is_credit", "day_of_month", "category_z", "rarity", "vs_income", "same_day_dup",
            "round_large"]
KINDS = ["amount_spike", "large_new_transfer", "duplicate_debit", "big_cash_deposit"]
SPIKE_TYPES = {"upi_p2m", "pos", "bill_pay", "atm_wdl"}
FALSE_ALARM_RATE = 0.01          # threshold = 99th percentile of scores on normal training data
ROBUST_Z = 3.5                   # modified z-score outlier cut-off (Iglewicz & Hoaglin, 1993)


def amount(t):
    return float(t.get("debit") or t.get("credit") or 0.0)


def monthly_inflow(txns):
    """Median total credit per calendar month (floor 5,000 so tiny accounts don't explode)."""
    per = defaultdict(float)
    for t in txns:
        if t.get("credit") and t.get("date"):
            per[t["date"][:7]] += t["credit"]
    return max(float(np.median(list(per.values()))) if per else 0.0, 5000.0)


def _norm(t):
    """Narration without digits (reference numbers change every month)."""
    return re.sub(r"[\d\s.]+", " ", (t.get("narration") or "").upper()).strip()


def _robust(vals):
    v = np.asarray(vals)
    med = float(np.median(v))
    return med, 1.4826 * float(np.median(np.abs(v - med)))


def account_features(txns, cats):
    """One row of FEATURES per transaction, computed against the same account's other transactions."""
    n = len(txns)
    if n == 0:
        return np.zeros((0, len(FEATURES)))
    la = np.array([np.log1p(amount(t)) for t in txns])
    cr = np.array([t.get("credit") is not None for t in txns])
    by_cat, by_dir = defaultdict(list), defaultdict(list)
    for i, c in enumerate(cats):
        by_cat[(c, cr[i])].append(la[i])
        by_dir[cr[i]].append(la[i])
    stats = {k: _robust(v) for k, v in by_cat.items() if len(v) >= 3}
    dstats = {k: _robust(v) for k, v in by_dir.items()}
    cat_count = Counter(cats)
    income = monthly_inflow(txns)
    key = Counter((t.get("date"), round(t["debit"], 2)) for t in txns if t.get("debit"))
    dup_months = defaultdict(set)
    for t in txns:
        if t.get("debit") and key[(t.get("date"), round(t["debit"], 2))] > 1:
            dup_months[(round(t["debit"], 2), _norm(t))].add((t.get("date") or "")[:7])
    recurring = {k for k, months in dup_months.items() if len(months) >= 2}
    X = np.zeros((n, len(FEATURES)))
    for i, t in enumerate(txns):
        med, mad = stats.get((cats[i], cr[i]), dstats[cr[i]])
        a = amount(t)
        day = int(t["date"][8:10]) if t.get("date") else 15
        X[i] = [la[i], float(cr[i]), day / 31,
                float(np.clip((la[i] - med) / (mad + 0.3), -10, 10)),
                -np.log((cat_count[cats[i]] + 1) / (n + len(cat_count))),
                np.log(max(a, 1.0) / income),
                (key[(t.get("date"), round(a, 2))] - 1
                 if t.get("debit") and (round(a, 2), _norm(t)) not in recurring else 0),
                float(a >= 10000 and abs(a / 1000 - round(a / 1000)) < 1e-9)]
    return X


def raw_features(txns, cats, labels):
    """Baseline features: no account context."""
    onehot = np.array([[c == l for l in labels] for c in cats], dtype=float).reshape(len(txns), len(labels))
    base = np.array([[np.log1p(amount(t)), t.get("credit") is not None,
                      (int(t["date"][8:10]) if t.get("date") else 15) / 31] for t in txns], dtype=float)
    return np.hstack([base, onehot])


def reason(f, cat):
    """Plain-language reason for a flagged transaction (shown in the app)."""
    d = dict(zip(FEATURES, f))
    if d["same_day_dup"] >= 1:
        return "same amount twice on the same day - possible double debit"
    if d["category_z"] > ROBUST_Z:
        return f"much larger than this account's usual {cat.replace('_', ' ')}"
    if d["vs_income"] >= 0:
        return f"{np.exp(d['vs_income']):.1f}x this account's monthly inflow"
    if d["rarity"] >= 4:
        return f"rare transaction type for this account ({cat.replace('_', ' ')})"
    return "unusual amount, date and type for this account"


class AnomalyDetector:
    def __init__(self, seed=0):
        self.model = IsolationForest(n_estimators=300, random_state=seed)
        self.threshold = None

    def fit(self, accounts):
        """accounts: list of (txns, cats) - normal data, no labels."""
        X = np.vstack([account_features(t, c) for t, c in accounts])
        self.model.fit(X)
        self.threshold = float(np.quantile(-self.model.score_samples(X), 1 - FALSE_ALARM_RATE))
        return self

    def score(self, txns, cats):
        """Higher = more unusual. Returns (scores, flags, reasons)."""
        if not txns:
            return [], [], []
        X = account_features(txns, cats)
        forest = -self.model.score_samples(X)
        rules = (X[:, FEATURES.index("same_day_dup")] >= 1) | (X[:, FEATURES.index("category_z")] > ROBUST_Z)
        flags = (forest > self.threshold) | rules
        score = forest + rules            # rule hits rank above everything else
        return score.tolist(), flags.tolist(), [reason(x, c) if f else "" for x, c, f in zip(X, cats, flags)]

    def forest_only(self, txns, cats):
        s = -self.model.score_samples(account_features(txns, cats))
        return s.tolist(), (s > self.threshold).tolist()

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)


# ------------------------------------------------------------------------------- evaluation

def inject(txns, rng, n, pool):
    """Add n labelled anomalies to one statement. Returns rows with 'label' and 'kind'."""
    out = [dict(t, label=0, kind="") for t in txns]
    debits = [t for t in txns if t.get("debit")]
    dates = sorted({t["date"] for t in txns})
    income = monthly_inflow(txns)
    for k in range(n):
        kind = KINDS[k % len(KINDS)]
        if kind == "amount_spike":
            base = rng.choice([t for t in debits if t["txn_type"] in SPIKE_TYPES] or debits)
            largest = max(t["debit"] for t in debits if t["txn_type"] == base["txn_type"])
            new = dict(base, date=rng.choice(dates), debit=round(largest * rng.uniform(3, 10), 2))
        elif kind == "large_new_transfer":
            typ = rng.choice(["imps_out", "neft_out"])
            new = dict(date=rng.choice(dates), narration=rng.choice(pool[typ]), debit=round(income * rng.uniform(0.8, 2.5), 2),
                       credit=None, txn_type=typ)
        elif kind == "duplicate_debit":
            new = dict(rng.choice(debits))
        else:
            new = dict(date=rng.choice(dates), narration=rng.choice(pool["cash_dep"]), debit=None,
                       credit=float(round(income * rng.uniform(1.5, 4) / 1000) * 1000), txn_type="cash_dep")
        new.update(label=1, kind=kind)
        out.append(new)
    out.sort(key=lambda t: t["date"])
    return out


def rule_scores(txns):
    """Baseline (a): classic 3-sigma rule on log-amount per account and direction."""
    la = np.array([np.log1p(amount(t)) for t in txns])
    cr = np.array([t.get("credit") is not None for t in txns])
    s = np.zeros(len(txns))
    for d in (True, False):
        m = cr == d
        if m.sum() >= 3:
            s[m] = (la[m] - la[m].mean()) / (la[m].std() + 1e-9)
    return s


def metrics(y, score, flag, kinds):
    y, flag = np.asarray(y), np.asarray(flag)
    tp = int((flag & (y == 1)).sum())
    out = dict(roc_auc=100 * roc_auc_score(y, score), avg_precision=100 * average_precision_score(y, score),
               precision=100 * tp / max(int(flag.sum()), 1), recall=100 * tp / max(int(y.sum()), 1),
               false_alarm_rate=100 * float((flag & (y == 0)).sum()) / max(int((y == 0).sum()), 1))
    for k in KINDS:
        m = np.array([kk == k for kk in kinds])
        out[f"recall_{k}"] = 100 * float(flag[m].mean()) if m.any() else float("nan")
    return out


def run(seeds=5, rate=0.01):
    gen = json.load(open(ROOT / "data/generated/ground_truth.json"))
    sts = list(json.load(open(ROOT / "data/original/ground_truth.json")).values())
    clf = TxnClassifier().fit([t for st in gen.values() for t in st["transactions"]])   # never saw these banks' statements
    pool = defaultdict(list)
    for st in sts:
        for t in st["transactions"]:
            pool[t["txn_type"]].append(t["narration"])
    clean = [(st["transactions"], [t["txn_type"] for t in st["transactions"]]) for st in sts]
    labels = sorted({c for _, cs in clean for c in cs})
    no_day = [i for i, f in enumerate(FEATURES) if f not in ("day_of_month", "is_credit")]

    # one model per held-out statement, trained on the other 19 (no labels used)
    models = []
    for i in range(len(sts)):
        tr = [c for j, c in enumerate(clean) if j != i]
        ours = AnomalyDetector().fit(tr)
        Xr = np.vstack([raw_features(t, c, labels) for t, c in tr])
        raw = IsolationForest(n_estimators=300, random_state=0).fit(Xr)
        Xa = np.vstack([account_features(t, c) for t, c in tr])[:, no_day]
        abl = IsolationForest(n_estimators=300, random_state=0).fit(Xa)
        models.append((ours, raw, np.quantile(-raw.score_samples(Xr), 1 - FALSE_ALARM_RATE),
                       abl, np.quantile(-abl.score_samples(Xa), 1 - FALSE_ALARM_RATE)))

    per_seed, clf_acc = defaultdict(list), []
    for seed in range(seeds):
        rng = random.Random(seed)
        ys, kinds, sc, fl = [], [], defaultdict(list), defaultdict(list)
        for st, (ours, raw, raw_thr, abl, abl_thr) in zip(sts, models):
            rows = inject(st["transactions"], rng, max(2, round(rate * len(st["transactions"]))), pool)
            cats = clf.predict(rows)
            clf_acc += [c == r["txn_type"] for c, r in zip(cats, rows)]
            ys += [r["label"] for r in rows]
            kinds += [r["kind"] for r in rows]
            r = rule_scores(rows)
            sc["rule"] += r.tolist(); fl["rule"] += (r > 3).tolist()
            r = -raw.score_samples(raw_features(rows, cats, labels))
            sc["raw"] += r.tolist(); fl["raw"] += (r > raw_thr).tolist()
            r, f = ours.forest_only(rows, cats)
            sc["forest"] += r; fl["forest"] += f
            r = -abl.score_samples(account_features(rows, cats)[:, no_day])
            sc["ablation"] += r.tolist(); fl["ablation"] += (r > abl_thr).tolist()
            r, f, _ = ours.score(rows, cats)
            sc["hybrid"] += r; fl["hybrid"] += f
            r, f, _ = ours.score(rows, [x["txn_type"] for x in rows])
            sc["hybrid_true"] += r; fl["hybrid_true"] += f
        for name in sc:
            per_seed[name].append(metrics(ys, np.array(sc[name]), np.array(fl[name]), kinds))
        print(f"seed {seed}: {sum(ys)} injected anomalies in {len(ys)} rows", flush=True)

    names = {"rule": "3-sigma amount rule (per account)",
             "raw": "Isolation Forest, raw features",
             "forest": "Isolation Forest, account features",
             "ablation": "  same, without day-of-month & direction",
             "hybrid": "Hybrid: forest + 2 rules (deployed)",
             "hybrid_true": "  same, with true categories"}
    res = {}
    for k, label in names.items():
        res[label] = {m: round(float(np.nanmean([d[m] for d in per_seed[k]])), 1) for m in per_seed[k][0]}
    return res, round(100 * float(np.mean(clf_acc)), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rate", type=float, default=0.01)
    a = ap.parse_args()
    res, clf_acc = run(a.seeds, a.rate)
    print(f"\nclassifier accuracy on the test statements (trained on generated data): {clf_acc}%")
    cols = ["roc_auc", "avg_precision", "precision", "recall", "false_alarm_rate"] + [f"recall_{k}" for k in KINDS]
    short = ["AUC", "AP", "prec", "recall", "FA%", "spike", "transfer", "dup", "cash"]
    print(f"{'method':44}" + "".join(f"{h:>9}" for h in short))
    for name, m in res.items():
        print(f"{name:44}" + "".join(f"{m[c]:9.1f}" for c in cols))
    if a.out:
        json.dump(dict(setup=dict(protocol="leave-one-statement-out on the 20 original statements: fit on 19 clean "
                                           "statements (no labels), test on the 20th with injected anomalies",
                                  categories="predicted by Linear SVM trained on the 140 generated statements",
                                  rate=a.rate, seeds=a.seeds, robust_z=ROBUST_Z,
                                  false_alarm_target=FALSE_ALARM_RATE, features=FEATURES),
                       classifier_accuracy_on_test=clf_acc, results=res), open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
