"""
classifier.py - STEP 7 (traditional ML): transaction categorisation.

Input  : narration text + amount + debit/credit direction
Output : one of the transaction types (upi_p2m, salary, nach_emi, atm_wdl, ...)
Features: TF-IDF on character 2-5-grams of the narration (digits masked) + log(amount) + direction
Models : Logistic Regression, Multinomial Naive Bayes, Linear SVM, Random Forest
Splits : (a) stratified 5-fold cross-validation
         (b) leave-one-bank-out - train on 6 banks, test on the unseen 7th

Usage:
    python src/classifier.py data/original/ground_truth.json --out results/classifier_original.json
    python src/classifier.py data/generated/ground_truth.json --out results/classifier_generated.json
"""
import argparse
import json
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=3000, C=10),
    "Naive Bayes": lambda: MultinomialNB(alpha=0.05),
    "Linear SVM": lambda: LinearSVC(C=1.0),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
}


def clean(s: str) -> str:
    s = re.sub(r"\d", "0", s.upper())          # reference numbers are noise -> mask digits
    return re.sub(r"\s+", " ", s)


def load(path):
    gt = json.load(open(path))
    rows = [(st["account"]["bank_code"], t) for st in gt.values() for t in st["transactions"]]
    banks = np.array([b for b, _ in rows])
    y = np.array([t["txn_type"] for _, t in rows])
    text = [clean(t["narration"]) for _, t in rows]
    amount = np.array([(t["debit"] or t["credit"]) for _, t in rows])
    is_credit = np.array([t["credit"] is not None for _, t in rows], dtype=float)
    num = np.c_[np.log1p(amount) / 16, is_credit, 1 - is_credit]     # non-negative (Naive Bayes)
    return banks, y, text, num


def features(text, num, tr, te):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform([text[i] for i in tr])
    Xte = vec.transform([text[i] for i in te])
    return hstack([Xtr, csr_matrix(num[tr])]).tocsr(), hstack([Xte, csr_matrix(num[te])]).tocsr()


def evaluate(y, text, num, splits):
    out = {}
    for name, make in MODELS.items():
        yt, yp = [], []
        for tr, te in splits:
            Xtr, Xte = features(text, num, tr, te)
            yt += list(y[te])
            yp += list(make().fit(Xtr, y[tr]).predict(Xte))
        out[name] = dict(accuracy=round(100 * accuracy_score(yt, yp), 2),
                         macro_f1=round(100 * f1_score(yt, yp, average="macro", zero_division=0), 2),
                         weighted_f1=round(100 * f1_score(yt, yp, average="weighted", zero_division=0), 2))
        out[name]["_pred"] = (yt, yp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ground_truth")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    banks, y, text, num = load(a.ground_truth)
    idx = np.arange(len(y))
    cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(idx, y))
    lobo = [(idx[banks != b], idx[banks == b]) for b in sorted(set(banks))]
    res_cv, res_lobo = evaluate(y, text, num, cv), evaluate(y, text, num, lobo)
    labels = sorted(set(y))
    yt, yp = res_cv["Linear SVM"].pop("_pred")
    cm = confusion_matrix(yt, yp, labels=labels).tolist()
    for r in (res_cv, res_lobo):
        for m in r.values():
            m.pop("_pred", None)
    print(f"{len(y)} transactions, {len(labels)} classes, banks {sorted(set(banks))}")
    print(f"{'model':20} {'CV acc':>8} {'CV F1':>8} {'unseen acc':>11} {'unseen F1':>10}")
    for n in MODELS:
        print(f"{n:20} {res_cv[n]['accuracy']:8} {res_cv[n]['macro_f1']:8} {res_lobo[n]['accuracy']:11} {res_lobo[n]['macro_f1']:10}")
    if a.out:
        json.dump(dict(rows=len(y), classes=labels, cv=res_cv, lobo=res_lobo, svm_confusion_matrix=cm),
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
