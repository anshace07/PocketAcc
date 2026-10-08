"""
generate.py - STEP 1: synthetic Indian bank statement generator.

Creates realistic statement PDFs for 7 banks (HDFC, SBI, PNB, Kotak, ICICI, Bank of
Baroda, Axis) together with a perfect answer key (ground truth). Every account holder
follows a spending profile (salaried, gig worker, trader, student, retiree, farmer,
homemaker, professional) so the transactions look like real cash flows, and the
running balance is always arithmetically correct.

Unlike the hand-made first dataset, this generator also records WHICH PAGE every
transaction is printed on - needed to train the page-by-page extraction model.

Usage:
    python src/generate.py --n 140 --out data/generated --seed 7
"""
import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from common import inr

# ------------------------------------------------------------------ bank layouts
# columns: (key, header, relative width, alignment)
BANKS = {
    "HDFC": dict(name="HDFC Bank", tag="We understand your world", color="#004C8F", title="STATEMENT OF ACCOUNT",
                 date_fmt="%d/%m/%y", ifsc="HDFC0", code="HDFC",
                 cols=[("date", "Date", 8, "L"), ("narration", "Narration", 38, "L"), ("ref", "Chq./Ref.No.", 13, "L"),
                       ("value_date", "Value Dt", 8, "L"), ("debit", "Withdrawal Amt.", 11, "R"),
                       ("credit", "Deposit Amt.", 11, "R"), ("balance", "Closing Balance", 12, "R")]),
    "SBI": dict(name="State Bank of India", tag="The Banker to Every Indian", color="#22409A", title="ACCOUNT STATEMENT",
                date_fmt="%d %b %Y", ifsc="SBIN0", code="SBIN",
                cols=[("date", "Txn Date", 9, "L"), ("value_date", "Value Date", 9, "L"),
                      ("narration", "Description", 36, "L"), ("ref", "Ref No./Cheque No.", 13, "L"),
                      ("debit", "Debit", 10, "R"), ("credit", "Credit", 10, "R"), ("balance", "Balance", 12, "R")]),
    "PNB": dict(name="Punjab National Bank", tag="The Name You Can Bank Upon", color="#A20A3A",
                title="STATEMENT OF ACCOUNT", date_fmt="%d/%m/%Y", ifsc="PUNB0", code="PUNB",
                cols=[("date", "Date", 9, "L"), ("ref", "Instrument ID", 12, "L"), ("narration", "Remarks", 45, "L"),
                      ("debit", "Amount(Dr)", 11, "R"), ("credit", "Amount(Cr)", 11, "R"), ("balance", "Balance", 12, "R")]),
    "KOTAK": dict(name="Kotak Mahindra Bank", tag="Let's make money simple", color="#ED1C24",
                  title="STATEMENT OF ACCOUNT", date_fmt="%d-%b-%y", ifsc="KKBK0", code="KKBK",
                  cols=[("date", "Date", 8, "L"), ("narration", "Narration", 44, "L"), ("ref", "Chq/Ref No", 14, "L"),
                        ("debit", "Withdrawal(Dr)", 11, "R"), ("credit", "Deposit(Cr)", 11, "R"),
                        ("balance", "Balance", 12, "R")]),
    "ICICI": dict(name="ICICI Bank", tag="Khayaal Aapka", color="#B02A30", title="Statement of Transactions",
                  date_fmt="%d-%m-%Y", ifsc="ICIC0", code="ICIC",
                  cols=[("sr", "S No.", 4, "L"), ("value_date", "Value Date", 9, "L"), ("date", "Transaction\nDate", 9, "L"),
                        ("ref", "Cheque\nNumber", 8, "L"), ("narration", "Transaction Remarks", 36, "L"),
                        ("debit", "Withdrawal Amount\n(INR)", 11, "R"), ("credit", "Deposit Amount\n(INR)", 11, "R"),
                        ("balance", "Balance (INR)", 12, "R")]),
    "BOB": dict(name="Bank of Baroda", tag="India's International Bank", color="#F15A22", title="STATEMENT OF ACCOUNT",
                date_fmt="%d/%m/%Y", ifsc="BARB0", code="BARB",
                cols=[("sr", "Sr", 4, "L"), ("date", "Txn Date", 9, "L"), ("value_date", "Value Date", 9, "L"),
                      ("narration", "Description", 36, "L"), ("ref", "Cheque No", 9, "L"), ("debit", "Debit", 10, "R"),
                      ("credit", "Credit", 10, "R"), ("balance", "Balance", 12, "R")]),
    "AXIS": dict(name="Axis Bank", tag="Badhti Ka Naam Zindagi", color="#97144D", title="STATEMENT OF ACCOUNT",
                 date_fmt="%d-%m-%Y", ifsc="UTIB0", code="UTIB",
                 cols=[("date", "Tran Date", 9, "L"), ("ref", "Chq No", 8, "L"), ("narration", "Particulars", 40, "L"),
                       ("debit", "Debit", 11, "R"), ("credit", "Credit", 11, "R"), ("balance", "Balance", 12, "R"),
                       ("init", "Init.\nBr", 5, "L")]),
}

# ------------------------------------------------------------------ narration grammars per bank
N = {
    "HDFC": {
        "upi_p2m": ["UPI-{MERCH}-{vpa}-HDFC-{r12}-UPI PAYMENT", "UPI-{MERCH}-{vpa}-HDFC-{r12}-PAY TO MERCHANT", "UPI-{MERCH}-{vpa}-HDFC-{r12}-UPI"],
        "upi_p2p_in": ["UPI-{Name}-{vpa}-HDFC-{r12}-CREDIT"], "upi_p2p_out": ["UPI-{Name}-{vpa}-HDFC-{r12}-UPI"],
        "salary": ["NEFT CR-{ifsc}-{EMPLOYER}-{HOLDER}-HDFCN{n14}"], "neft_in": ["NEFT CR-{ifsc}-{PAYER}-{HOLDER}-HDFCN{n14}"],
        "neft_out": ["NEFT DR-{ifsc}-{PAYEE}-HDFCN{n14}"], "imps_out": ["IMPS-{r12}-{Name}-{BNK}-{purpose}"],
        "nach_emi": ["ACH D- {LENDER}-{PRODUCT} EMI"], "sip": ["ACH D- {FUND}-SIP"], "atm_wdl": ["ATW-{card}-S1ANW{n3}-{CITY}"],
        "pos": ["POS {card} {MERCH}"], "cash_dep": ["CASH DEP-{CITY} BRANCH-SELF"], "bill_pay": ["BILLPAY DR-{BILLER}-{n10}"],
        "interest": ["CREDIT INTEREST CAPITALISED"], "charge": ["SMS ALERT CHARGES INCL GST {MON}{YY}"],
        "cheque_paid": ["CHQ PAID-{n6}-{PAYEE}"], "cheque_dep": ["CHQ DEP-{n6}-{PAYER}"]},
    "SBI": {
        "upi_p2m": ["TO TRANSFER-UPI/DR/{r12}/{Merch}/SBIN/{vpa}/Payment"], "upi_p2p_in": ["BY TRANSFER-UPI/CR/{r12}/{Name}/SBIN/{vpa}/UPI"],
        "upi_p2p_out": ["TO TRANSFER-UPI/DR/{r12}/{Name}/SBIN/{vpa}/UPI"], "salary": ["BY TRANSFER-NEFT*{ifsc}*SBINN{n9}*{EMPLOYER}"],
        "neft_in": ["BY TRANSFER-NEFT*{ifsc}*SBINN{n9}*{PAYER}"], "neft_out": ["TO TRANSFER-NEFT*{ifsc}*SBINN{n9}*{PAYEE}"],
        "imps_out": ["TO TRANSFER-INB IMPS/P2A/{r12}/{NAME}"], "nach_emi": ["TO TRANSFER-NACH DR {LENDER} {PRODUCT} EMI"],
        "sip": ["TO TRANSFER-NACH DR {FUND} SIP"], "atm_wdl": ["ATM WDL-ATM CASH S6ANW{n3} {CITY}"],
        "pos": ["TO TRANSFER-POS {card} {MERCH}"], "cash_dep": ["BY CASH-{CITY} BRANCH"],
        "bill_pay": ["TO TRANSFER-INB BILLPAY {BILLER} {n10}"], "interest": ["CREDIT INTEREST"],
        "charge": ["DEBIT-SMS ALERT CHARGES INCL GST {MON}{YY}"], "cheque_paid": ["TO CLG-CHEQUE PAID {PAYEE}"],
        "cheque_dep": ["BY CLG-{PAYER}"], "dbt": ["BY TRANSFER-DBT {SCHEME}"]},
    "PNB": {
        "upi_p2m": ["UPI/DR/{r12}/{MERCH18}/PUNB/{vpa}"], "upi_p2p_in": ["UPI/CR/{r12}/{NAME}/PUNB/{vpa}"],
        "upi_p2p_out": ["UPI/DR/{r12}/{NAME}/PUNB/{vpa}"], "salary": ["NEFT CR PUNBN{n9} {EMPLOYER} SALARY"],
        "neft_in": ["NEFT CR PUNBN{n9} {PAYER}"], "neft_out": ["NEFT DR PUNBN{n9} {PAYEE}"], "imps_out": ["IMPS DR {r12} {NAME}"],
        "nach_emi": ["NACH DR {LENDER} {PRODUCT} EMI"], "sip": ["NACH DR {FUND} SIP"], "atm_wdl": ["ATM CASH WDL {CITY} PNB{n4}"],
        "pos": ["POS {card} {MERCH}"], "cash_dep": ["BY CASH DEP {CITY} BRANCH"], "bill_pay": ["BILL PAYMENT {BILLER} {n10}"],
        "interest": ["INT CREDITED FOR THE QUARTER"], "charge": ["SMS ALERT CHARGES INCL GST {MON}{YY}"],
        "cheque_paid": ["CHQ PAID {n6} {PAYEE}"], "cheque_dep": ["CLG CR {n6} {PAYER}"], "dbt": ["DBT CR {SCHEME}"]},
    "KOTAK": {
        "upi_p2m": ["UPI/{r12}/Payment to {Merch}"], "upi_p2p_in": ["UPI/{r12}/Received from {Name}/{vpa}"],
        "upi_p2p_out": ["UPI/{r12}/Sent to {Name}/{vpa}"], "salary": ["NEFT INWARD-{EMPLOYER} SALARY"],
        "neft_in": ["NEFT INWARD-{PAYER}"], "neft_out": ["NEFT OUTWARD-{PAYEE}"], "imps_out": ["MB:Fund Trf to {Name} XXXX{n4}"],
        "nach_emi": ["NACH DR-{LENDER}-{PRODUCT} EMI"], "sip": ["NACH DR-{FUND}-SIP"], "atm_wdl": ["ATM WDL {CITY} {card}"],
        "pos": ["POS PURCHASE {MERCH} {n4}"], "cash_dep": ["CASH DEPOSIT {CITY} BRANCH"], "bill_pay": ["BILL PAY-{BILLER}-{n10}"],
        "interest": ["INTEREST CREDIT"], "charge": ["SMS ALERT CHARGES {MON}{YY} INCL GST"],
        "remit_in": ["INWARD REMITTANCE-{FOREIGN}"], "self": ["MB:Self Trf Recurring Deposit Instal"]},
    "ICICI": {
        "upi_p2m": ["UPI/{r12}/Payment to {Merch}/{vpa}/ICIC"], "upi_p2p_in": ["UPI/{r12}/{Name}/{vpa}/ICIC/Received"],
        "upi_p2p_out": ["UPI/{r12}/Sent to {Name}/{vpa}/ICIC"], "salary": ["NEFT-ICICN{n9}-{EMPLOYER}-SALARY"],
        "neft_in": ["NEFT-ICICN{n9}-{PAYER}"], "neft_out": ["NEFT/ICICN{n9}/{PAYEE}"],
        "imps_out": ["MMT/IMPS/{r12}/{purpose}/{Name}/{BNK}"], "nach_emi": ["ACH/{LENDER}/{PRODUCT} EMI"], "sip": ["ACH/{FUND}/SIP"],
        "atm_wdl": ["NFS/CASH WDL/{CITY}/{card}"], "pos": ["VIN/{Merch}/{dmy}/{card}"], "cash_dep": ["CASH DEP/{CITY} BR"],
        "bill_pay": ["BIL/ONL/{n9}/{BILLER}/{n9}"], "interest": ["CREDIT INTEREST CAPITALISED"],
        "charge": ["SMS ALERT CHARGES INCL GST {MON}{YY}"], "remit_in": ["IRM/{FOREIGN}/{n9}"]},
    "BOB": {
        "upi_p2m": ["UPI/{r12}/Paid to {MERCH}/{vpa}"], "upi_p2p_in": ["UPI/{r12}/Recd from {NAME}/{vpa}"],
        "upi_p2p_out": ["UPI/{r12}/Paid to {NAME}/{vpa}"], "salary": ["NEFT-CR-{ifsc}-{EMPLOYER}-SALARY"],
        "neft_in": ["NEFT-CR-{ifsc}-{PAYER}"], "neft_out": ["NEFT-DR-{PAYEE}"], "imps_out": ["IMPS-DR-{r12}-{NAME}"],
        "nach_emi": ["NACH-DR-{LENDER}-{PRODUCT} EMI"], "sip": ["NACH-DR-{FUND}-SIP"], "atm_wdl": ["ATM-WDL-{CITY}-BOB{n4}"],
        "pos": ["POS-{card}-{MERCH}"], "cash_dep": ["CASH-DEP-{CITY} BRANCH"], "bill_pay": ["BILLPAY-{BILLER}-{n10}"],
        "interest": ["SB INT CREDIT"], "charge": ["SMS ALERT CHARGES INCL GST {MON}{YY}"], "dbt": ["DBTL-{SCHEME}-GOVT OF INDIA"],
        "cheque_paid": ["CHQ-PAID-{PAYEE}"], "cheque_dep": ["CLG-CR-{PAYER}"]},
    "AXIS": {
        "upi_p2m": ["UPI/P2M/{r12}/{MERCH}/{vpa}"], "upi_p2p_in": ["UPI/P2A/{r12}/{NAME}/{vpa}"],
        "upi_p2p_out": ["UPI/P2A/{r12}/{NAME}/{vpa}/DR"], "salary": ["NEFT/UTIBN{n9}/{EMPLOYER}/SALARY"],
        "neft_in": ["NEFT/UTIBN{n9}/{PAYER}"], "neft_out": ["NEFT/DR/UTIBN{n9}/{PAYEE}"], "imps_out": ["IMPS/P2A/{r12}/{NAME}"],
        "nach_emi": ["ACH-DR-{LENDER}-{PRODUCT} EMI"], "sip": ["ACH-DR-{FUND}-SIP"], "atm_wdl": ["ATM-CASH/{CITY}/{card}"],
        "pos": ["POS/{MERCH}/{card}"], "cash_dep": ["BY CASH DEPOSIT-{CITY}"], "bill_pay": ["BILLPAY/{BILLER}/{n10}"],
        "interest": ["SB:INTEREST CREDIT"], "charge": ["CHEQUE BOOK ISSUE CHARGES INCL GST", "SMS ALERT CHARGES {MON}{YY}"],
        "cheque_paid": ["CHQ/PAID/{PAYEE}"], "cheque_dep": ["CLG/CR/{PAYER}"], "rtgs_in": ["RTGS/UTIBN{n9}/{PAYER}"]},
}
GENERIC = {"rtgs_in": ["RTGS CR-{PAYER}-{n9}"], "remit_in": ["INWARD REMITTANCE-{FOREIGN}"], "dbt": ["DBT-{SCHEME}"],
           "self": ["SELF TRANSFER-FD SWEEP"], "cheque_paid": ["CHQ PAID {PAYEE}"], "cheque_dep": ["CHQ DEP {PAYER}"],
           "pos": ["POS {MERCH}"], "cash_dep": ["CASH DEPOSIT {CITY}"]}

# ------------------------------------------------------------------ vocabulary
FIRST = ["Aarav", "Rohan", "Priya", "Sneha", "Vikram", "Anjali", "Rahul", "Pooja", "Amit", "Neha", "Suresh", "Kavita",
         "Arjun", "Divya", "Manoj", "Ritu", "Sanjay", "Meera", "Karan", "Nisha", "Deepak", "Asha", "Imran", "Fatima",
         "Gurpreet", "Harpreet", "Joseph", "Mary", "Lakshmi", "Venkat", "Ramesh", "Sunita", "Ajay", "Swati", "Nitin", "Rekha"]
LAST = ["Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Shah", "Reddy", "Nair", "Iyer", "Pillai", "Das", "Bose",
        "Chatterjee", "Mehta", "Joshi", "Kulkarni", "Deshpande", "Rao", "Naidu", "Khan", "Ansari", "Thomas", "Fernandes",
        "Mishra", "Pandey", "Yadav", "Chauhan", "Saikia", "Bhatt", "Malhotra", "Kapoor"]
MERCHANTS = ["SWIGGY", "ZOMATO", "BIGBASKET", "BLINKIT", "ZEPTO", "AMAZON PAY", "FLIPKART", "DMART", "RELIANCE FRESH",
             "MCDONALDS", "DOMINOS PIZZA", "KFC INDIA", "STARBUCKS", "CAFE COFFEE DAY", "UBER INDIA", "OLA CABS",
             "RAPIDO", "IRCTC", "MAKEMYTRIP", "BOOKMYSHOW", "INDIAN OIL PETROL", "HP PETROL PUMP", "SHELL INDIA",
             "APOLLO PHARMACY", "MEDPLUS", "NETMEDS", "MYNTRA", "AJIO", "NYKAA", "DECATHLON", "CROMA", "JIO RECHARGE",
             "AIRTEL PREPAID", "FASTAG NETC RECHARGE", "RAJU TEA STALL", "SHARMA GENERAL STORE", "ANNAPURNA TIFFIN",
             "BALAJI SWEETS", "MAA DURGA DAIRY", "KHAN LAUNDRY", "PARKING CHARGES", "METRO CASH CARRY", "AMMA MESS",
             "SAI MEDICAL", "GUPTA FRUITS", "BARBEQUE NATION", "HALDIRAMS", "BIKANERVALA", "CHAAYOS"]
VPA_SUFFIX = ["@ybl", "@paytm", "@okaxis", "@oksbi", "@okicici", "@ibl", "@axl", "@ptys", "@apl", "@icici", "@hdfcbank"]
EMPLOYERS = ["INFOWAVE TECHNOLOGIES PVT", "TATA CONSULTANCY SERVICES", "INFOSYS LTD", "WIPRO LIMITED",
             "HCL TECHNOLOGIES", "DIRECTORATE OF EDUCATION", "APOLLO HOSPITALS ENTERPRISE", "RELIANCE RETAIL LTD",
             "MARUTI SUZUKI INDIA", "BAJAJ AUTO LTD", "EASTERN LIFESTYLE RETAIL", "STATE HEALTH SOCIETY"]
PAYERS_BIZ = ["SHREE BALAJI FABRICS", "GOKUL FASHION HOUSE", "MAHAVIR TEXTILES PVT LTD", "NEW ERA GARMENTS",
              "ARIHANT SILK MILLS", "KRISHNA TRADERS", "OM SAI ENTERPRISES", "GANESH AGENCIES", "PATEL BROTHERS"]
PAYEES_BIZ = ["SANJAY YARN AGENCY", "SURAT POLY PACK", "ALOK INDUSTRIES", "GST PAYMENT", "INCOME TAX ADVANCE TAX",
              "SHREE CEMENT DEALER", "HINDUSTAN UNILEVER DIST", "ITC LTD DISTRIBUTOR", "ADITYA BIRLA FASHION"]
GIG_PAYERS = ["BUNDL TECHNOLOGIES PAYOUT", "ZOMATO MEDIA PAYOUT", "ANI TECHNOLOGIES PAYOUT", "UBER INDIA SYSTEMS",
              "RAPIDO ROPPEN TRANSPORT", "URBANCLAP TECHNOLOGIES"]
LENDERS = [("BAJAJ FINANCE LTD", "PERSONAL LOAN"), ("HDFC BANK LTD", "HOME LOAN"), ("LIC HOUSING FINANCE", "HOME LOAN"),
           ("TVS CREDIT SERVICES", "TWO WHEELER"), ("MAHINDRA FINANCE", "VEHICLE LOAN"), ("HDB FINANCIAL SERVICES", "BUSINESS LOAN"),
           ("TATA CAPITAL", "CONSUMER DURABLE"), ("ICICI BANK", "CAR LOAN")]
FUNDS = ["NIPPON INDIA MF", "SBI BLUECHIP FUND", "MIRAE ASSET LARGE CAP", "AXIS ELSS TAX SAVER", "PARAG PARIKH FLEXI CAP",
         "MOTILAL OSWAL MIDCAP", "HDFC INDEX NIFTY 50", "ICICI PRU BALANCED ADV"]
BILLERS = ["BHARTI AIRTEL LTD", "RELIANCE JIO INFOCOM", "VODAFONE IDEA LTD", "BSES RAJDHANI POWER", "TATA POWER DDL",
           "MAHANAGAR GAS LTD", "BHARAT GAS CNG", "TATA PLAY DTH", "ACT FIBERNET", "BSNL", "MSEDCL ELECTRICITY",
           "DELHI JAL BOARD", "LIC PREMIUM"]
SCHEMES = ["PM KISAN SAMMAN NIDHI", "LPG SUBSIDY PAHAL", "PM UJJWALA YOJANA", "STATE PENSION SCHEME", "MGNREGA WAGES"]
FOREIGN = ["UPWORK GLOBAL INC", "FIVERR INTERNATIONAL", "STRIPE PAYMENTS", "WISE PAYMENTS LTD"]
CITIES = [("Delhi", "DELHI - 110005"), ("Mumbai", "MAHARASHTRA - 400014"), ("Bengaluru", "KARNATAKA - 560095"),
          ("Hyderabad", "TELANGANA - 500003"), ("Chennai", "TAMIL NADU - 600040"), ("Kolkata", "WEST BENGAL - 700034"),
          ("Pune", "MAHARASHTRA - 411001"), ("Ahmedabad", "GUJARAT - 380009"), ("Jaipur", "RAJASTHAN - 302001"),
          ("Lucknow", "UTTAR PRADESH - 226001"), ("Indore", "MADHYA PRADESH - 452010"), ("Patna", "BIHAR - 800001"),
          ("Kochi", "KERALA - 682024"), ("Guwahati", "ASSAM - 781006"), ("Chandigarh", "CHANDIGARH - 160017"),
          ("Surat", "GUJARAT - 395009"), ("Nagpur", "MAHARASHTRA - 440010"), ("Bhopal", "MADHYA PRADESH - 462011")]
BANK_SHORT = ["HDFC", "ICIC", "SBIN", "UTIB", "KKBK", "PUNB", "BARB", "FDRL", "INDB", "YESB"]
PURPOSES = ["Rent", "House Rent", "Fees", "Family", "Loan Repay", "Advance"]

PROFILES = {
    "salaried": dict(occupations=["Software Engineer", "Bank Clerk", "Government School Teacher", "Sales Executive",
                                  "Staff Nurse", "Retail Store Manager", "Factory Machine Operator"], weight=0.40),
    "self_employed": dict(occupations=["Kirana Store Owner", "Textile Trader", "Restaurant Owner",
                                       "Chartered Accountant (Practice)", "Consulting Physician", "Hardware Shop Owner"], weight=0.25),
    "gig": dict(occupations=["Food Delivery Partner", "Cab Driver", "Freelance Designer", "Bike Taxi Rider"], weight=0.12),
    "student": dict(occupations=["Student"], weight=0.08),
    "low_income": dict(occupations=["Farmer", "Homemaker", "Retired (PSU)", "Domestic Worker"], weight=0.15),
}


# ------------------------------------------------------------------ data model
@dataclass
class Txn:
    d: date
    ttype: str
    amount: float
    credit: bool
    counterparty: str
    narration: str = ""
    ref: str = ""
    balance: float = 0.0
    page: int = 0


@dataclass
class Statement:
    bank: str
    holder: str
    profile: str
    occupation: str
    city: str
    state_pin: str
    account_no: str
    account_type: str
    ifsc: str
    period_from: date
    period_to: date
    opening: float
    txns: list = field(default_factory=list)


def digits(rng, n):
    return "".join(rng.choice("0123456789") for _ in range(n))


def person(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def vpa_for(name, rng):
    if rng.random() < 0.35:
        return f"q{digits(rng, 10)}{rng.choice(VPA_SUFFIX)}"
    if not name.strip():
        return f"q{digits(rng, 10)}{rng.choice(VPA_SUFFIX)}"
    base = name.split()[0].lower().replace(" ", "")
    return f"{base}{rng.choice(['', digits(rng, 2), digits(rng, 4), '.' + rng.choice(LAST).lower()])}{rng.choice(VPA_SUFFIX)}"


def make_narration(bank, t: Txn, st: Statement, rng, card):
    grammar = N[bank].get(t.ttype) or GENERIC.get(t.ttype) or ["{PAYEE}"]
    tpl = rng.choice(grammar)
    cp = t.counterparty
    lender, product = (cp.split("|") + [""])[:2] if "|" in cp else (cp, "")
    city = st.city.upper()
    ctx = dict(r12=digits(rng, 12), n3=digits(rng, 3), n4=digits(rng, 4), n6=digits(rng, 6), n9=digits(rng, 9),
               n10=digits(rng, 10), n14=digits(rng, 14), card=card, CITY=city, HOLDER=st.holder.upper(),
               MERCH=cp.upper(), Merch=cp.title(), MERCH18=cp.upper()[:18], Name=cp, NAME=cp.upper(), vpa=vpa_for(cp, rng),
               EMPLOYER=cp.upper()[:26], PAYER=cp.upper(), PAYEE=cp.upper(), LENDER=lender, PRODUCT=product, FUND=cp,
               BILLER=cp, SCHEME=cp, FOREIGN=cp, BNK=rng.choice(BANK_SHORT), purpose=rng.choice(PURPOSES),
               ifsc=f"{rng.choice(BANK_SHORT)}000{digits(rng, 4)}", MON=t.d.strftime("%b").upper(), YY=t.d.strftime("%y"),
               dmy=t.d.strftime("%d%m%Y"))
    return tpl.format(**ctx), ctx["r12"]


# ------------------------------------------------------------------ cash-flow simulation
def simulate(st: Statement, rng: random.Random):
    """Create a realistic month-by-month list of transactions for the holder's profile."""
    p, days = st.profile, (st.period_to - st.period_from).days + 1
    ev = []

    def add(d, ttype, amt, credit, cp):
        ev.append(Txn(d, ttype, round(amt, 2), credit, cp))

    salary = rng.choice([18000, 25000, 32000, 45000, 62000, 85000, 120000]) * rng.uniform(0.95, 1.05)
    employer = rng.choice(EMPLOYERS)
    rent = rng.choice([0, 6000, 9000, 14000, 22000])
    landlord = person(rng)
    emis = [(rng.choice(LENDERS), rng.choice([2400, 4100, 8420, 14600, 22300])) for _ in range(rng.randint(0, 2))]
    sips = [(rng.choice(FUNDS), rng.choice([500, 1000, 2000, 5000])) for _ in range(rng.randint(0, 2))]
    friends = [person(rng) for _ in range(6)]
    biz_in, biz_out = rng.sample(PAYERS_BIZ, 4), rng.sample(PAYEES_BIZ, 4)
    gig_payer = rng.choice(GIG_PAYERS)
    scale = {"salaried": 1.0, "self_employed": 1.6, "gig": 0.5, "student": 0.25, "low_income": 0.35}[p]

    for k in range(days):
        d = st.period_from + timedelta(days=k)
        # monthly events
        if d.day == rng.choice([1, 1, 2]) or (d.day == 1):
            if p == "salaried" and not any(e.ttype == "salary" and e.d.month == d.month for e in ev):
                add(d, "salary", salary, True, employer)
            if p == "low_income" and st.occupation.startswith("Retired") and d.day == 1:
                add(d, "neft_in", salary * 0.6, True, "PENSION DISBURSING AUTH")
            if p == "student" and d.day == 1:
                add(d, "upi_p2p_in", rng.choice([3000, 5000, 8000]), True, rng.choice(friends[:2]))
        if d.day == 5:
            if rent and p in ("salaried", "gig", "self_employed"):
                add(d, "imps_out", rent, False, landlord)
            for (lender, prod), amt in emis:
                add(d, "nach_emi", amt, False, f"{lender}|{prod}")
        if d.day == 10:
            for fund, amt in sips:
                add(d, "sip", amt, False, fund)
        if d.day in (12, 20) and rng.random() < 0.7:
            add(d, "bill_pay", rng.uniform(199, 2400), False, rng.choice(BILLERS))
        if d.day == 28 and rng.random() < 0.6:
            add(d, "charge", rng.choice([17.70, 23.60, 29.50, 59.00, 118.00]), False, "")
        if d.day == 30 and d.month in (3, 6, 9, 12):
            add(d, "interest", rng.uniform(40, 900) * scale, True, "")
        # daily events
        for _ in range(rng.choices([0, 1, 2, 3, 4], [0.25, 0.35, 0.2, 0.12, 0.08])[0]):
            add(d, "upi_p2m", rng.choice([rng.uniform(20, 400), rng.uniform(100, 1500), rng.uniform(800, 4000)]) * scale,
                False, rng.choice(MERCHANTS))
        if rng.random() < 0.10:
            add(d, "upi_p2p_out", rng.choice([100, 200, 500, 1000, 2000]), False, rng.choice(friends))
        if rng.random() < 0.08:
            add(d, "upi_p2p_in", rng.choice([150, 300, 500, 1200, 2500]), True, rng.choice(friends))
        if rng.random() < 0.05:
            add(d, "atm_wdl", rng.choice([500, 1000, 2000, 3000, 5000]), False, "")
        if rng.random() < 0.04:
            add(d, "pos", rng.uniform(250, 6000) * scale, False, rng.choice(MERCHANTS))
        # profile-specific
        if p == "gig" and rng.random() < 0.6:
            add(d, "neft_in", rng.uniform(400, 2200), True, gig_payer)
        if p == "gig" and st.occupation.startswith("Freelance") and rng.random() < 0.04:
            add(d, "remit_in", rng.uniform(8000, 60000), True, rng.choice(FOREIGN))
        if p == "self_employed":
            if rng.random() < 0.35:
                add(d, rng.choice(["neft_in", "neft_in", "cheque_dep", "rtgs_in"]), rng.uniform(5000, 160000), True, rng.choice(biz_in))
            if rng.random() < 0.28:
                add(d, rng.choice(["neft_out", "neft_out", "cheque_paid"]), rng.uniform(4000, 140000), False, rng.choice(biz_out))
            if rng.random() < 0.15:
                add(d, "cash_dep", rng.choice([5000, 12000, 25000, 40000]), True, "")
            if rng.random() < 0.15:
                add(d, "upi_p2p_in", rng.uniform(100, 3000), True, rng.choice(friends))
        if p == "low_income":
            if st.occupation == "Farmer" and rng.random() < 0.02:
                add(d, "dbt", rng.choice([2000, 6000]), True, rng.choice(SCHEMES))
            if rng.random() < 0.04:
                add(d, "cash_dep", rng.choice([1000, 2000, 5000]), True, "")
            if st.occupation == "Homemaker" and rng.random() < 0.05:
                add(d, "upi_p2p_in", rng.choice([2000, 5000, 10000]), True, friends[0])
    ev.sort(key=lambda e: e.d)

    # apply to balance; drop debits that would overdraw the account
    bal, out = st.opening, []
    for e in ev:
        if not e.credit and e.amount > bal - 50:
            if e.ttype in ("upi_p2m", "pos", "upi_p2p_out", "atm_wdl", "bill_pay") or bal < 1000:
                continue
            e.amount = round(max(10.0, (bal - 50) * rng.uniform(0.3, 0.8)), 2)
        bal = round(bal + (e.amount if e.credit else -e.amount), 2)
        e.balance = bal
        out.append(e)
    return out


# ------------------------------------------------------------------ PDF rendering
PAGE_W, PAGE_H = A4
MARGIN = 26


def wrap_chars(text, font, size, width):
    """Wrap by characters (statements break mid-token, e.g. 'sharma@hdfcba|nk')."""
    lines, cur = [], ""
    for ch in text:
        if stringWidth(cur + ch, font, size) > width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines or [""]


def tint(hexcolor, f):
    c = HexColor(hexcolor)
    return HexColor("#%02x%02x%02x" % tuple(int(255 - (255 - v * 255) * f) for v in (c.red, c.green, c.blue)))


class Renderer:
    def __init__(self, st: Statement, rng: random.Random):
        self.st, self.rng, self.b = st, rng, BANKS[st.bank]
        self.fs = rng.choice([6.3, 6.6, 7.0])                   # font size jitter (domain randomisation)
        self.lead = self.fs + rng.choice([1.6, 2.0, 2.4])
        self.zebra = rng.random() < 0.5
        tw = PAGE_W - 2 * MARGIN
        total = sum(c[2] for c in self.b["cols"])
        self.cols = [(k, h, tw * w / total, a) for k, h, w, a in self.b["cols"]]

    def cells(self, t: Txn | None, idx):
        st, fmt = self.st, self.b["date_fmt"]
        if t is None:     # opening balance row
            ds = st.period_from.strftime(fmt)
            return {"date": ds, "value_date": ds, "narration": "OPENING BALANCE B/F", "balance": inr(st.opening),
                    "init": "1001" if st.bank == "AXIS" else ""}
        ds = t.d.strftime(fmt)
        ref = t.ref if self.st.bank not in ("ICICI", "BOB") or t.ttype.startswith("cheque") else ""
        return {"sr": str(idx), "date": ds, "value_date": ds, "narration": t.narration, "ref": ref,
                "debit": "" if t.credit else inr(t.amount), "credit": inr(t.amount) if t.credit else "",
                "balance": inr(t.balance), "init": "1001"}

    def row_lines(self, cells):
        nar_w = next(w for k, _, w, _ in self.cols if k == "narration") - 6
        return max(1, len(wrap_chars(cells.get("narration", ""), "Helvetica", self.fs, nar_w)),
                   *(len(wrap_chars(cells.get(k, ""), "Helvetica", self.fs, w - 6)) for k, _, w, _ in self.cols if k == "ref"))

    def layout(self):
        """Assign every transaction to a page (first pass)."""
        first_top = PAGE_H - MARGIN - 52 - 92 - 44 - 26
        cont_top = PAGE_H - MARGIN - 26 - 26
        bottom = MARGIN + 30
        page, y = 1, first_top
        rows = [(None, 0)] + [(t, i + 1) for i, t in enumerate(self.st.txns)]
        plan = []
        for t, idx in rows:
            h = self.row_lines(self.cells(t, idx)) * self.lead + 3
            if y - h < bottom:
                page += 1
                y = cont_top
            plan.append((t, idx, page, y, h))
            y -= h
            if t is not None:
                t.page = page
        return plan, page

    def draw(self, path):
        st, b, fs = self.st, self.b, self.fs
        plan, npages = self.layout()
        c = canvas.Canvas(str(path), pagesize=A4)
        c.setTitle(f"{b['name']} statement")
        color = HexColor(b["color"])
        tot_dr = sum(t.amount for t in st.txns if not t.credit)
        tot_cr = sum(t.amount for t in st.txns if t.credit)
        closing = st.txns[-1].balance if st.txns else st.opening
        cur_page = 0
        for t, idx, page, y, h in plan:
            if page != cur_page:
                if cur_page:
                    self.footer(c, cur_page, npages)
                    c.showPage()
                cur_page = page
                top = self.first_header(c, color, closing, tot_dr, tot_cr) if page == 1 else self.cont_header(c, color)
                self.table_header(c, top, color)
            cells = self.cells(t, idx)
            if self.zebra and idx % 2 == 0:
                c.setFillColor(tint(b["color"], 0.05))
                c.rect(MARGIN, y - h + 1, PAGE_W - 2 * MARGIN, h, stroke=0, fill=1)
            c.setFillColor(black)
            x = MARGIN
            for k, _, w, a in self.cols:
                lines = wrap_chars(cells.get(k, ""), "Helvetica", fs, w - 6) if k in ("narration", "ref") else [cells.get(k, "")]
                c.setFont("Helvetica", fs)
                for j, ln in enumerate(lines):
                    yy = y - fs - j * self.lead
                    if a == "R":
                        c.drawRightString(x + w - 3, yy, ln)
                    else:
                        c.drawString(x + 3, yy, ln)
                x += w
            c.setStrokeColor(tint("#888888", 0.35))
            c.setLineWidth(0.3)
            c.line(MARGIN, y - h + 1, PAGE_W - MARGIN, y - h + 1)
        self.footer(c, cur_page, npages)
        c.save()
        return npages

    def first_header(self, c, color, closing, tot_dr, tot_cr):
        st, b = self.st, self.b
        top = PAGE_H - MARGIN
        c.setFillColor(color)
        c.rect(MARGIN, top - 44, PAGE_W - 2 * MARGIN, 44, stroke=0, fill=1)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 15)
        c.drawString(MARGIN + 12, top - 22, b["name"])
        c.setFont("Helvetica", 6.5)
        c.drawString(MARGIN + 12, top - 34, b["tag"])
        c.setFont("Helvetica-Bold", 10)
        c.drawRightString(PAGE_W - MARGIN - 12, top - 20, b["title"])
        c.setFont("Helvetica", 6.5)
        c.drawRightString(PAGE_W - MARGIN - 12, top - 31, f"Generated on {st.period_to + timedelta(days=1):%d-%b-%Y} at 09:{self.rng.randint(10, 59)}:{self.rng.randint(10, 59)}")
        y = top - 60
        left = [("Account Holder", st.holder.upper()), ("Address", f"{self.rng.randint(1, 99)}, {self.rng.choice(['MG Road', 'Station Road', 'Civil Lines', 'Sector 12', 'Gandhi Nagar'])}"),
                ("", f"{st.city.upper()}, {st.state_pin}"), ("Mobile / Email", f"9XXXXX{digits(self.rng, 4)} | {st.holder.split()[0].lower()}{digits(self.rng, 2)}@gmail.com"),
                ("Customer ID", digits(self.rng, 9))]
        right = [("Account Number", st.account_no), ("Account Type", st.account_type), ("Branch", f"{st.city} Main Branch"),
                 ("IFSC", st.ifsc), ("Statement Period", f"{st.period_from:%d-%b-%Y} to {st.period_to:%d-%b-%Y}")]
        for i, ((l1, v1), (l2, v2)) in enumerate(zip(left, right)):
            yy = y - i * 11
            c.setFillColor(HexColor("#555555"))
            c.setFont("Helvetica", 6.5)
            c.drawString(MARGIN + 6, yy, l1)
            c.drawString(PAGE_W / 2 + 10, yy, l2)
            c.setFillColor(black)
            c.drawString(MARGIN + 80, yy, v1)
            c.drawString(PAGE_W / 2 + 90, yy, v2)
        y -= 70
        boxes = [("OPENING BALANCE", inr(st.opening)), (f"TOTAL WITHDRAWALS ({sum(1 for t in st.txns if not t.credit)})", inr(tot_dr)),
                 (f"TOTAL DEPOSITS ({sum(1 for t in st.txns if t.credit)})", inr(tot_cr)), ("CLOSING BALANCE", inr(closing))]
        bw = (PAGE_W - 2 * MARGIN - 18) / 4
        for i, (lab, val) in enumerate(boxes):
            x = MARGIN + i * (bw + 6)
            c.setStrokeColor(tint(b["color"], 0.4))
            c.setFillColor(tint(b["color"], 0.06))
            c.rect(x, y - 30, bw, 30, stroke=1, fill=1)
            c.setFillColor(HexColor("#555555"))
            c.setFont("Helvetica", 5.8)
            c.drawString(x + 6, y - 10, lab)
            c.setFillColor(color)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(x + 6, y - 24, val)
        return y - 44

    def cont_header(self, c, color):
        st, b = self.st, self.b
        top = PAGE_H - MARGIN
        c.setFillColor(color)
        c.rect(MARGIN, top - 22, PAGE_W - 2 * MARGIN, 22, stroke=0, fill=1)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(MARGIN + 10, top - 15, f"{b['name']}  ·  Statement of Account (contd.)")
        c.setFont("Helvetica", 6.5)
        c.drawRightString(PAGE_W - MARGIN - 10, top - 14, f"A/c {st.account_no}  |  {st.period_from:%d %b %Y} to {st.period_to:%d %b %Y}")
        return top - 26

    def table_header(self, c, top, color):
        c.setFillColor(tint(self.b["color"], 0.15))
        c.rect(MARGIN, top - 22, PAGE_W - 2 * MARGIN, 22, stroke=0, fill=1)
        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 6.3)
        x = MARGIN
        for k, h, w, a in self.cols:
            for j, part in enumerate(h.split("\n")):
                yy = top - 9 - j * 7
                (c.drawRightString(x + w - 3, yy, part) if a == "R" else c.drawString(x + 3, yy, part))
            x += w

    def footer(self, c, page, npages):
        c.setFillColor(HexColor("#777777"))
        c.setFont("Helvetica", 5.5)
        c.drawString(MARGIN, MARGIN + 6, f"This is a computer generated statement and does not require a signature. "
                                         f"{self.b['name']} - synthetic document for research use.")
        c.drawRightString(PAGE_W - MARGIN, MARGIN + 6, f"Page {page} of {npages}")


# ------------------------------------------------------------------ main
def make_statement(i, rng):
    bank = list(BANKS)[i % len(BANKS)]
    profile = rng.choices(list(PROFILES), [v["weight"] for v in PROFILES.values()])[0]
    occupation = rng.choice(PROFILES[profile]["occupations"])
    city, state_pin = rng.choice(CITIES)
    months = rng.choice([1, 1, 2, 2, 3])
    start = date(2026, rng.randint(1, 6), 1)
    end = date(start.year + (start.month - 1 + months) // 12, (start.month - 1 + months) % 12 + 1, 1) - timedelta(days=1)
    acct_type = "CURRENT" if profile == "self_employed" and rng.random() < 0.6 else "SAVINGS"
    opening = round(rng.choice([2500, 8000, 15000, 40000, 90000, 250000, 900000]) * rng.uniform(0.8, 1.2)
                    * (3 if profile == "self_employed" else 1), 2)
    st = Statement(bank=bank, holder=person(rng), profile=profile, occupation=occupation, city=city, state_pin=state_pin,
                   account_no=digits(rng, rng.choice([11, 12, 14, 15, 16])), account_type=acct_type,
                   ifsc=f"{BANKS[bank]['ifsc']}{digits(rng, 6)}", period_from=start, period_to=end, opening=opening)
    st.txns = simulate(st, rng)
    card = f"5XXXXXXXX{digits(rng, 4)}"
    for t in st.txns:
        t.narration, t.ref = make_narration(bank, t, st, rng, card)
        if t.ttype in ("interest", "charge", "atm_wdl", "cash_dep") and rng.random() < 0.5:
            t.ref = ""
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=140)
    ap.add_argument("--out", default="data/generated")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "pdf").mkdir(parents=True, exist_ok=True)
    gt, index = {}, []
    for i in range(args.n):
        st = make_statement(i, rng)
        fname = f"G{i + 1:03d}_{st.bank}_{st.holder.replace(' ', '-')}.pdf"
        pages = Renderer(st, rng).draw(out / "pdf" / fname)
        fmt = BANKS[st.bank]["date_fmt"]
        txns = [{"sr": k + 1, "date": t.d.isoformat(), "narration": t.narration, "reference": t.ref,
                 "debit": None if t.credit else t.amount, "credit": t.amount if t.credit else None, "balance": t.balance,
                 "txn_type": t.ttype, "counterparty": t.counterparty.split("|")[0], "page": t.page,
                 "printed": [t.d.strftime(fmt), t.narration, "" if t.credit else inr(t.amount),
                             inr(t.amount) if t.credit else "", inr(t.balance)]}
                for k, t in enumerate(st.txns)]
        closing = st.txns[-1].balance if st.txns else st.opening
        gt[fname] = {
            "account": {"bank": BANKS[st.bank]["name"], "bank_code": st.bank, "holder": st.holder, "occupation": st.occupation,
                        "profile": st.profile, "city": st.city, "account_no": st.account_no, "account_type": st.account_type,
                        "ifsc": st.ifsc, "period_from": st.period_from.isoformat(), "period_to": st.period_to.isoformat()},
            "summary": {"opening_balance": st.opening, "closing_balance": closing,
                        "total_debit": round(sum(t.amount for t in st.txns if not t.credit), 2),
                        "total_credit": round(sum(t.amount for t in st.txns if t.credit), 2),
                        "transaction_count": len(st.txns), "pages": pages},
            "header_printed": {"bank": BANKS[st.bank]["name"], "holder": st.holder.upper(), "account_no": st.account_no,
                               "period": f"{st.period_from:%d-%b-%Y} to {st.period_to:%d-%b-%Y}",
                               "opening_balance": inr(st.opening), "closing_balance": inr(closing)},
            "transactions": txns,
        }
        index.append(dict(file=fname, bank=st.bank, profile=st.profile, occupation=st.occupation, pages=pages,
                          txn_count=len(st.txns), opening=st.opening, closing=closing))
        print(f"{fname:48} pages={pages:2d} txns={len(st.txns)}")
    json.dump(gt, open(out / "ground_truth.json", "w"), indent=1, ensure_ascii=False)
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    print(f"\n{len(gt)} statements, {sum(r['pages'] for r in index)} pages, {sum(r['txn_count'] for r in index)} transactions")


if __name__ == "__main__":
    main()
