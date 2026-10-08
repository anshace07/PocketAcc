"""
common.py - small helpers shared by every PocketAcc module.

* Indian number formatting / parsing (1,41,897.90)
* date parsing for every bank's date style
* the JSON schema the extraction model must produce
"""
import json
import re
from datetime import date, datetime

# ---------------------------------------------------------------- numbers

def inr(x: float | None, decimals: int = 2) -> str:
    """Format a number in the Indian lakh/crore style: 1234567.5 -> '12,34,567.50'."""
    if x is None:
        return ""
    neg = x < 0
    whole, frac = f"{abs(x):.{decimals}f}".split(".") if decimals else (f"{abs(round(x))}", None)
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups) + "," + tail
    out = whole + (f".{frac}" if frac is not None else "")
    return ("-" if neg else "") + out


_NUM = re.compile(r"[^0-9.\-]")


def parse_amount(s) -> float | None:
    """'1,41,897.90' -> 141897.9 ; '' / None / '-' -> None ; tolerates 'Cr', 'Dr', '₹'."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if not s or s in {"-", "--"}:
        return None
    s = _NUM.sub("", s)
    if s.count(".") > 1:                       # '1.41.897.90' -> keep last dot as decimal
        head, _, tail = s.rpartition(".")
        s = head.replace(".", "") + "." + tail
    try:
        return round(float(s), 2)
    except ValueError:
        return None


# ---------------------------------------------------------------- dates

DATE_FORMATS = ["%d/%m/%y", "%d/%m/%Y", "%d-%m-%Y", "%d-%m-%y", "%d %b %Y", "%d-%b-%y",
                "%d-%b-%Y", "%d %b %y", "%Y-%m-%d", "%d.%m.%Y"]


def parse_date(s: str) -> str | None:
    """Any bank date style -> ISO 'YYYY-MM-DD' (or None)."""
    if not s:
        return None
    s = re.sub(r"\s+", " ", str(s).strip())
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def fmt_date(d: date, fmt: str) -> str:
    return d.strftime(fmt)


# ---------------------------------------------------------------- model I/O schema

PROMPT = (
    "You are reading one page of an Indian bank statement. "
    "Return ONLY JSON with this shape: "
    '{"header": {"bank": str, "holder": str, "account_no": str, "period": str, '
    '"opening_balance": str, "closing_balance": str} or null, '
    '"rows": [[date, narration, debit, credit, balance], ...]}. '
    "Copy dates, narration and amounts exactly as printed (keep commas). "
    "Use \"\" for an empty debit or credit. Join wrapped narration lines with no space. "
    "Skip the OPENING BALANCE B/F row. header is null on continuation pages."
)


def target_json(header: dict | None, rows: list[list[str]]) -> str:
    """Compact JSON string the model is trained to emit."""
    return json.dumps({"header": header, "rows": rows}, ensure_ascii=False, separators=(",", ":"))


def parse_model_output(text: str) -> dict:
    """Robustly pull the JSON object out of a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rstrip("`").strip()
    start = text.find("{")
    if start < 0:
        return {"header": None, "rows": []}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
    # truncated output: keep every complete row we can recover
    rows = re.findall(r'\[\s*"([^"]*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\]', text)
    return {"header": None, "rows": [list(r) for r in rows]}


def rows_to_transactions(rows: list) -> list[dict]:
    """Model rows (printed strings) -> normalised transactions."""
    out = []
    for r in rows:
        if not isinstance(r, (list, tuple)) or len(r) < 5:
            continue
        d, nar, dr, cr, bal = (list(r) + [""] * 5)[:5]
        out.append({"date": parse_date(d), "narration": str(nar).strip(),
                    "debit": parse_amount(dr), "credit": parse_amount(cr),
                    "balance": parse_amount(bal)})
    return out
