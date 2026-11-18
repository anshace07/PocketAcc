"""
make_demo_statements.py - demo statements for the project presentation.

Uses our own generator (src/generate.py) but with a NEW, made-up bank layout -
"PocketAcc Demo Bank" - that the model never saw in training. So the demo is an honest
"unseen bank" test, and nothing here looks like a real bank's document.

Creates in demo/samples/:
    A_salaried_statement.pdf        salaried customer, 1 month (digital PDF)
    A_page1_photo_light.jpg         page 1 photographed: light damage
    A_page1_photo_medium.jpg        page 1 photographed: medium damage
    A_page1_photo_heavy.jpg         page 1 photographed: heavy damage
    B_shop_statement.pdf            kirana-shop owner (self-employed), 1 month
    ground_truth.json               the answer key for every file (to score the demo results)

Statement A contains one planted "double debit" (the same shop payment taken twice on the
same day) so the unusual-transaction check has something real to find.

Usage:
    python scripts/make_demo_statements.py            # deterministic (fixed seeds)
"""
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

import generate as G  # noqa: E402
from common import inr  # noqa: E402
from degrade import degrade  # noqa: E402
from preprocess import render_pages  # noqa: E402

OUT = ROOT / "demo" / "samples"

DEMO_BANK = dict(
    name="PocketAcc Demo Bank", tag="Sample statement - synthetic data for demonstration only", color="#0F766E",
    title="ACCOUNT STATEMENT", date_fmt="%d/%m/%Y", ifsc="DEMO0", code="DEMO",
    cols=[("date", "Date", 9, "L"), ("narration", "Particulars", 40, "L"), ("ref", "Chq/Ref No.", 13, "L"),
          ("debit", "Withdrawals", 11, "R"), ("credit", "Deposits", 11, "R"), ("balance", "Balance", 12, "R")])

DEMO_GRAMMAR = {
    "upi_p2m": ["UPI/DR/{r12}/{Merch}/{vpa}", "UPI-{MERCH}-{vpa}-{r12}"],
    "upi_p2p_in": ["UPI/CR/{r12}/{Name}/{vpa}"], "upi_p2p_out": ["UPI/DR/{r12}/{Name}/{vpa}"],
    "salary": ["NEFT CR-{ifsc}-{EMPLOYER}-SALARY"], "neft_in": ["NEFT CR-{ifsc}-{PAYER}"],
    "neft_out": ["NEFT DR-{ifsc}-{PAYEE}"], "imps_out": ["IMPS/P2A/{r12}/{Name}/{purpose}"],
    "nach_emi": ["NACH DR-{LENDER}-{PRODUCT} EMI"], "sip": ["NACH DR-{FUND}-SIP"],
    "atm_wdl": ["ATM WDL-{CITY}-{card}"], "pos": ["POS {card} {MERCH}"], "cash_dep": ["CASH DEPOSIT-{CITY} BRANCH"],
    "bill_pay": ["BILLPAY-{BILLER}-{n10}"], "interest": ["SB INTEREST CREDIT"],
    "charge": ["SMS ALERT CHARGES INCL GST {MON}{YY}"], "cheque_paid": ["CHQ PAID-{n6}-{PAYEE}"],
    "cheque_dep": ["CHQ DEP-{n6}-{PAYER}"], "rtgs_in": ["RTGS CR-{PAYER}-{n9}"]}


def register():
    """Add the demo bank to the generator at run time (generate.py itself is unchanged, so the
    training data it produces with --seed 7 stays exactly the same)."""
    G.BANKS["DEMO"] = DEMO_BANK
    G.N["DEMO"] = DEMO_GRAMMAR


def rebalance(st):
    bal = st.opening
    for t in st.txns:
        bal = round(bal + (t.amount if t.credit else -t.amount), 2)
        t.balance = bal
    return min(t.balance for t in st.txns)


def plant_double_debit(st, rng):
    """Repeat one mid-month shop payment on the same day (a classic double charge)."""
    cands = [t for t in st.txns if t.ttype == "upi_p2m" and 400 <= t.amount <= 2500 and 8 <= t.d.day <= 22]
    t = rng.choice(cands)
    dup = G.Txn(t.d, t.ttype, t.amount, False, t.counterparty)
    st.txns.insert(st.txns.index(t) + 1, dup)
    return dup


def make(seed, profile, occupation, start, months, opening, double_debit=False):
    rng = random.Random(seed)
    city, state_pin = rng.choice(G.CITIES)
    end = date(start.year + (start.month - 1 + months) // 12, (start.month - 1 + months) % 12 + 1, 1) - timedelta(days=1)
    acct_type = "CURRENT" if profile == "self_employed" else "SAVINGS"
    st = G.Statement(bank="DEMO", holder=G.person(rng), profile=profile, occupation=occupation, city=city,
                     state_pin=state_pin, account_no=G.digits(rng, 14), account_type=acct_type,
                     ifsc=f"DEMO0{G.digits(rng, 6)}", period_from=start, period_to=end, opening=opening)
    st.txns = G.simulate(st, rng)
    planted = plant_double_debit(st, rng) if double_debit else None
    assert rebalance(st) > 0, "planted debit overdrew the account - pick another seed"
    card = f"5XXXXXXXX{G.digits(rng, 4)}"
    for t in st.txns:
        t.narration, t.ref = G.make_narration("DEMO", t, st, rng, card)
        if t.ttype in ("interest", "charge", "atm_wdl", "cash_dep") and rng.random() < 0.5:
            t.ref = ""
    return st, rng, planted


def answer_key(st, pages, planted=None, only_page=None):
    fmt = DEMO_BANK["date_fmt"]
    txns = [t for t in st.txns if only_page is None or t.page == only_page]
    closing = st.txns[-1].balance
    return {
        "account": {"bank": DEMO_BANK["name"], "bank_code": "DEMO", "holder": st.holder, "occupation": st.occupation,
                    "profile": st.profile, "city": st.city, "account_no": st.account_no,
                    "account_type": st.account_type, "ifsc": st.ifsc,
                    "period_from": st.period_from.isoformat(), "period_to": st.period_to.isoformat()},
        "summary": {"opening_balance": st.opening, "closing_balance": closing, "pages": pages,
                    "transaction_count": len(txns), "only_page": only_page},
        "transactions": [{"sr": k + 1, "date": t.d.isoformat(), "narration": t.narration, "reference": t.ref,
                          "debit": None if t.credit else t.amount, "credit": t.amount if t.credit else None,
                          "balance": t.balance, "txn_type": t.ttype, "page": t.page,
                          "planted_double_debit": t is planted,
                          "printed": [t.d.strftime(fmt), t.narration, "" if t.credit else inr(t.amount),
                                      inr(t.amount) if t.credit else "", inr(t.balance)]}
                         for k, t in enumerate(txns)],
    }


def main():
    register()
    OUT.mkdir(parents=True, exist_ok=True)
    key = {}

    # A: salaried employee, October 2026, with one planted double debit
    st, rng, planted = make(seed=2611, profile="salaried", occupation="Software Engineer",
                            start=date(2026, 10, 1), months=1, opening=48250.75, double_debit=True)
    pdf_a = OUT / "A_salaried_statement.pdf"
    pages = G.Renderer(st, rng).draw(pdf_a)
    key[pdf_a.name] = answer_key(st, pages, planted)
    page1 = render_pages(str(pdf_a), 150)[0]
    for level, seed in (("light", 3), ("medium", 5), ("heavy", 7)):
        img, info = degrade(page1, level, seed=seed)
        out = OUT / f"A_page1_photo_{level}.jpg"
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        key[out.name] = answer_key(st, pages, planted, only_page=1) | {"damage": info}
    print(f"{pdf_a.name}: {pages} pages, {len(st.txns)} transactions, holder {st.holder}, "
          f"double debit on {planted.d} of {planted.amount} ({planted.counterparty})")

    # B: kirana-shop owner (self-employed), September 2026
    st, rng, _ = make(seed=905, profile="self_employed", occupation="Kirana Store Owner",
                      start=date(2026, 9, 1), months=1, opening=186400.00)
    pdf_b = OUT / "B_shop_statement.pdf"
    pages = G.Renderer(st, rng).draw(pdf_b)
    key[pdf_b.name] = answer_key(st, pages)
    print(f"{pdf_b.name}: {pages} pages, {len(st.txns)} transactions, holder {st.holder}")

    json.dump(key, open(OUT / "ground_truth.json", "w"), indent=1, ensure_ascii=False)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
