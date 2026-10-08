"""
corrector.py - Balance-chain verifier and corrector for PocketAcc.

A bank statement is a chain:   balance[i] = balance[i-1] - debit[i] + credit[i]
If the extractor misreads one number, the chain breaks at a predictable place:
  * a wrong AMOUNT in row i breaks only check(i)
  * a wrong BALANCE in row i breaks check(i) and check(i+1)
The arithmetic tells us the exact value the field should have. We accept that
value only if it is a *plausible misread* of what was extracted (e.g. 3<->8,
1<->7, a dropped digit, a shifted decimal). Otherwise the row is flagged for
human review instead of being silently "fixed".
"""
from dataclasses import dataclass, field

TOL = 0.005  # half a paisa

# Digit pairs that OCR / vision models commonly confuse.
CONFUSION_PAIRS = {("0", "8"), ("3", "8"), ("1", "7"), ("5", "6"), ("6", "8"),
                   ("4", "9"), ("2", "7"), ("0", "6"), ("0", "9"), ("5", "8")}
CONFUSABLE = CONFUSION_PAIRS | {(b, a) for a, b in CONFUSION_PAIRS}


@dataclass
class Txn:
    debit: float | None
    credit: float | None
    balance: float
    flags: list = field(default_factory=list)


def _digits(x: float) -> str:
    return f"{abs(x):.2f}".replace(".", "")


def is_plausible_misread(read: float, truth: float, max_subs: int = 2) -> bool:
    """True if `read` could be an OCR misreading of `truth`."""
    if read is None or truth is None:
        return False
    a, b = _digits(read), _digits(truth)
    if a == b:                                   # same digits -> decimal slip
        return True
    if len(a) == len(b):                         # substitutions only
        diffs = [(x, y) for x, y in zip(a, b) if x != y]
        return len(diffs) <= max_subs and all(d in CONFUSABLE for d in diffs)
    if abs(len(a) - len(b)) == 1:                # one dropped / extra digit
        s, l = (a, b) if len(a) < len(b) else (b, a)
        return any(l[:k] + l[k + 1:] == s for k in range(len(l)))
    return False


def chain_ok(prev_bal: float, t: Txn) -> bool:
    return abs(prev_bal - (t.debit or 0) + (t.credit or 0) - t.balance) < TOL


def verify(opening: float, txns: list[Txn]) -> list[bool]:
    """Return ok[i] for every row of the balance chain."""
    ok, prev = [], opening
    for t in txns:
        ok.append(chain_ok(prev, t))
        prev = t.balance
    return ok


def _fix_amount(prev: float, t: Txn) -> bool:
    """Row's own check fails: try to repair debit/credit (incl. column swap)."""
    delta = round(t.balance - prev, 2)            # +credit / -debit expected
    need, col = abs(delta), ("credit" if delta > 0 else "debit")
    read = t.credit if t.credit is not None else t.debit
    read_col = "credit" if t.credit is not None else "debit"
    if not is_plausible_misread(read, need):
        return False
    if col != read_col:
        t.flags.append(f"swapped {read_col}->{col}")
    else:
        t.flags.append(f"{col} {read} -> {need}")
    t.debit, t.credit = (need, None) if col == "debit" else (None, need)
    return True


def _fix_balance(prev: float, t: Txn, nxt: Txn | None) -> bool:
    """Row i and i+1 both fail: the balance printed on row i is the suspect."""
    need = round(prev - (t.debit or 0) + (t.credit or 0), 2)
    if nxt is not None and abs(need - (nxt.debit or 0) + (nxt.credit or 0) - nxt.balance) >= TOL:
        return False                              # fix would not heal row i+1
    if not is_plausible_misread(t.balance, need):
        return False
    t.flags.append(f"balance {t.balance} -> {need}")
    t.balance = need
    return True


def correct(opening: float, txns: list[Txn], closing: float | None = None) -> dict:
    """Detect and repair chain breaks. Returns a small report."""
    for _ in range(4 * len(txns) + 1):          # each pass repairs one field
        ok = verify(opening, txns)
        if all(ok):
            break
        changed = False
        for i, good in enumerate(ok):
            if good:
                continue
            prev = opening if i == 0 else txns[i - 1].balance
            nxt = txns[i + 1] if i + 1 < len(txns) else None
            next_bad = nxt is not None and not ok[i + 1]
            if next_bad and _fix_balance(prev, txns[i], nxt):
                changed = True
            elif _fix_amount(prev, txns[i]):
                changed = True
            elif nxt is None and closing is not None and _fix_balance(prev, txns[i], None) \
                    and abs(txns[i].balance - closing) < TOL:
                changed = True
            if changed:
                break                             # re-verify after every repair
        if not changed:
            break
    ok = verify(opening, txns)
    for t, good in zip(txns, ok):
        if not good:
            t.flags.append("UNRESOLVED - needs human review")
    return {"rows": len(txns), "verified": sum(ok),
            "trust_score": round(100 * sum(ok) / max(len(ok), 1), 1),
            "corrections": sum(1 for t in txns if any("->" in f for f in t.flags)),
            "flagged": len(ok) - sum(ok)}
