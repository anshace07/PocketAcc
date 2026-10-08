"""
corrector_sim.py - Simulation study for the balance corrector.

Takes ground-truth statements, injects OCR-style errors into amount/balance
fields at a chosen rate, runs the corrector, and measures what happened.
"""
import copy, json, random, sys
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from corrector import Txn, correct, CONFUSABLE

GT = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
SUBS = {}
for a, b in sorted(CONFUSABLE):
    SUBS.setdefault(a, []).append(b)


def corrupt(value: float, rng: random.Random, kind: str) -> float:
    s = f"{value:.2f}"
    digits = [i for i, ch in enumerate(s) if ch.isdigit() and ch in SUBS]
    if kind in ("confusion1", "confusion2") and digits:
        for i in rng.sample(digits, min(len(digits), 1 if kind == "confusion1" else 2)):
            s = s[:i] + rng.choice(SUBS[s[i]]) + s[i + 1:]
    elif kind == "drop_digit":
        idx = [i for i, ch in enumerate(s) if ch.isdigit()]
        i = rng.choice(idx[:-2] or idx)
        s = s[:i] + s[i + 1:]
    elif kind == "decimal_shift":
        return round(value * rng.choice([10, 100, 0.1]), 2)
    elif kind == "random_digit":            # not a known confusion -> should be flagged
        idx = [i for i, ch in enumerate(s) if ch.isdigit()]
        i = rng.choice(idx)
        s = s[:i] + str((int(s[i]) + rng.choice([2, 4])) % 10) + s[i + 1:]
    v = float(s)
    return v if abs(v - value) > 0.004 else round(value + 1, 2)


KINDS = [("confusion1", .55), ("confusion2", .10), ("drop_digit", .10),
         ("decimal_shift", .05), ("swap_column", .10), ("random_digit", .10)]


def run(rate: float, seed: int):
    rng = random.Random(seed)
    tot = dict(fields=0, wrong_before=0, wrong_after=0, injected=0, fixed=0,
               flagged=0, missed=0, mis_inj=0, mis_clean=0, clean_rows_flagged=0, rows=0)
    for name, st in GT.items():
        truth = [Txn(t["debit"], t["credit"], t["balance"]) for t in st["transactions"]]
        pred = copy.deepcopy(truth)
        hit = set()
        for i, t in enumerate(pred):
            if rng.random() >= rate:
                continue
            kind = rng.choices([k for k, _ in KINDS], [w for _, w in KINDS])[0]
            target = "balance" if rng.random() < 0.5 else "amount"
            if kind == "swap_column":
                t.debit, t.credit = t.credit, t.debit
            elif target == "balance":
                t.balance = corrupt(t.balance, rng, kind)
            elif t.debit is not None:
                t.debit = corrupt(t.debit, rng, kind)
            else:
                t.credit = corrupt(t.credit, rng, kind)
            hit.add(i)
        def wrong(a, b):
            # two fields per row: the amount (debit/credit) and the balance
            return int((a.debit, a.credit) != (b.debit, b.credit)) + int(abs(a.balance - b.balance) > .004)
        before = [wrong(p, g) for p, g in zip(pred, truth)]
        correct(st["summary"]["opening_balance"], pred, st["summary"]["closing_balance"])
        after = [wrong(p, g) for p, g in zip(pred, truth)]
        for i, (p, b, a) in enumerate(zip(pred, before, after)):
            unresolved = any("UNRESOLVED" in f for f in p.flags)
            changed = any("->" in f for f in p.flags)
            if i in hit:
                tot["injected"] += 1
                if a == 0:
                    tot["fixed"] += 1
                elif unresolved:
                    tot["flagged"] += 1
                elif changed:
                    tot["mis_inj"] += 1
                else:
                    tot["missed"] += 1
            else:
                if a > 0:
                    tot["mis_clean"] += 1
                if unresolved:
                    tot["clean_rows_flagged"] += 1
        tot["rows"] += len(truth)
        tot["fields"] += 2 * len(truth)                      # amount + balance
        tot["wrong_before"] += sum(min(b, 2) for b in before)
        tot["wrong_after"] += sum(min(a, 2) for a in after)
    return tot


results = []
for rate in [0.01, 0.02, 0.05, 0.10, 0.20]:
    agg = None
    for seed in range(5):
        r = run(rate, seed)
        agg = r if agg is None else {k: agg[k] + r[k] for k in agg}
    n = agg["injected"]
    results.append(dict(
        error_rate=rate,
        injected=n / 5,
        acc_before=round(100 * (1 - agg["wrong_before"] / agg["fields"]), 2),
        acc_after=round(100 * (1 - agg["wrong_after"] / agg["fields"]), 2),
        detected=round(100 * (agg["fixed"] + agg["flagged"] + agg["mis_inj"]) / n, 1),
        auto_fixed=round(100 * agg["fixed"] / n, 1),
        flagged=round(100 * agg["flagged"] / n, 1),
        missed=round(100 * agg["missed"] / n, 1),
        wrong_fix=round(100 * agg["mis_inj"] / n, 1),
        clean_rows_damaged=agg["mis_clean"] / 5,
        clean_rows_flagged=agg["clean_rows_flagged"] / 5,
    ))
    print(results[-1])
json.dump(results, open(OUT, "w"), indent=1)
