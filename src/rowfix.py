"""
rowfix.py - STEP 5a: repair the COLUMN STRUCTURE of rows a reader returned, before the balance check.

Why: error analysis of the fine-tuned model's test outputs (results/ft_*.jsonl) showed that it usually
reads the numbers correctly but sometimes puts them in the wrong place in the row:
  * an extra column copied from the page (cheque/ref number, Axis "Init. Br" code, empty cells):
        [date, narration, "33210353862", "952.00", "", "94,062.45"]          (PNB, 6 values)
        [date, narration, "3,436.81", "", "14,480.45", "", "1001"]          (Axis, 7 values)
  * an empty cell left out:
        [date, narration, "947.00", "2,76,049.19"]                          (BOB, 4 values)
  * the "OPENING BALANCE B/F" row, which the prompt says to skip.
With 5 positional slots [date, narration, debit, credit, balance] every shifted value lands in the
wrong field, so all three amounts of such a row count as wrong.

What it does (only for rows that are NOT already a clean [date, narration, amount/"" , amount/"", amount]):
  * keep only values that look like money (Indian or plain format with 2 decimals) - reference numbers,
    codes and empty cells are ignored
  * the LAST money value is the balance, the one before it is the transaction amount
  * debit or credit is decided by the statement's own arithmetic: previous balance - amount = balance
    -> debit; previous balance + amount = balance -> credit; if neither adds up exactly (another misread
    in the row), by the direction the balance moved
  * a row with a single money value is left as it was (it cannot be placed safely)
Rows that were already clean are never changed, so Tesseract output (always 5 clean slots) is unaffected.
"""
import re

from common import parse_amount

MONEY = re.compile(r"^-?\d{1,3}(,\d{2})*(,\d{3})?\.\d{2}$|^\d+\.\d{2}$")
SKIP = re.compile(r"OPENING BALANCE|BALANCE B/F|BROUGHT FORWARD|STATEMENT SUMMARY", re.I)
TOL = 0.005


def _is_money(x):
    return bool(MONEY.match(str(x).strip()))


def _clean_shape(r):
    """[date, narration, debit, credit, balance] with exactly one of debit/credit and a balance."""
    if len(r) != 5:
        return False
    dr, cr, bal = (str(x).strip() for x in r[2:])
    return _is_money(bal) and ((_is_money(dr) and cr == "") or (dr == "" and _is_money(cr)))


def repair_rows(rows, opening=None):
    """rows: reader output (lists of printed strings). opening: balance before the first row, if known.
    Returns (rows with 5 slots each, info) - info: fixed / dropped counts and the indices of fixed rows."""
    out, fixed_rows, dropped = [], [], 0
    prev = parse_amount(opening) if opening is not None else None
    for r in rows:
        if not isinstance(r, (list, tuple)) or len(r) < 2:
            continue
        r = [str(x) if x is not None else "" for x in r]
        if SKIP.search(r[1]) or SKIP.search(r[0]):
            dropped += 1                                   # opening-balance / summary line, not a transaction
            continue
        if not _clean_shape(r):
            money = [x.strip() for x in r[2:] if _is_money(x)]
            if len(money) >= 2:
                amt, bal = parse_amount(money[-2]), parse_amount(money[-1])
                if prev is not None and abs(prev - amt - bal) < TOL:
                    debit = True
                elif prev is not None and abs(prev + amt - bal) < TOL:
                    debit = False
                elif prev is not None:
                    debit = bal < prev
                else:                                      # no previous balance: keep the reader's column
                    slots = (r + [""] * 5)[2:5]
                    debit = slots[0].strip() == money[-2]
                new = [r[0], r[1], money[-2] if debit else "", "" if debit else money[-2], money[-1]]
                if [x.strip() for x in new] != [x.strip() for x in (r + [""] * 5)[:5]]:
                    fixed_rows.append(len(out))           # the values really moved (not just extra empty cells)
                r = new
            else:
                r = (r + [""] * 5)[:5]
        bal = parse_amount(r[4])
        if bal is not None:
            prev = bal
        out.append(r)
    return out, {"fixed": len(fixed_rows), "dropped": dropped, "fixed_rows": fixed_rows}
