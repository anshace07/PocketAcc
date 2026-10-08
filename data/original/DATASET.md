# Synthetic Indian Bank Statement Dataset

**20 bank statements · 7 banks · 4774 transactions · 110 PDF pages**

Generated 2026-09-02.

---

## What this is (and is not)

Every statement here is **fully synthetic**. The account holders, account numbers,
IFSC/MICR codes, PAN numbers, phone numbers, email addresses, UPI VPAs, counterparties,
transaction references and balances are all generated. No real person's data was used,
scraped, or transformed. Merchant and biller names are real Indian brands used the way a
narration string would name them; every counterparty *individual* is fictional.

What is real is the **structure**: layouts, column sets, narration grammars, reference
number formats, charge conventions and cash-flow behaviour are modelled on how the
actual banks print statements. That is what makes it useful as training and test data.

---

## Contents

```
pdf/                    20 statement PDFs, one per account holder
ground_truth.json       every transaction, labelled — the answer key
index.json              per-statement metadata
index.csv               same index as a spreadsheet
```

### `ground_truth.json`

Keyed by PDF filename. Each entry:

```jsonc
{
  "account":  { bank, bank_code, holder, occupation, city, account_no,
                account_type, scheme, ifsc, micr, branch, branch_address,
                customer_id, cif, pan, mobile, email, nominee,
                period_from, period_to },
  "summary":  { opening_balance, closing_balance, total_debit, total_credit,
                debit_count, credit_count, transaction_count, pages },
  "transactions": [
    { "sr": 1,
      "date": "2026-06-01", "value_date": "2026-06-01",
      "narration": "UPI-SWIGGY-q8152042905@ybl-ICIC-685163100626-UPI PAYMENT",
      "reference": "685163100626",
      "debit": 363.0, "credit": null, "balance": 523569.35,
      "txn_type": "upi_p2m", "counterparty": "SWIGGY" }
  ]
}
```

`txn_type` is a clean categorical label — useful as a classification target:
`upi_p2m`, `upi_p2p_in`, `upi_p2p_out`, `salary`, `neft_in`, `neft_out`, `rtgs_in`,
`imps_out`, `nach_emi`, `sip`, `atm_wdl`, `pos`, `cash_dep`, `cheque_paid`,
`cheque_dep`, `bill_pay`, `interest`, `charge`, `remit_in`, `dbt`, `self`.

---

## The 20 accounts

| # | Bank | Holder | Occupation | City | Type | Period | Txns | Pages |
|---|------|--------|-----------|------|------|--------|------|-------|
| 01 | HDFC Bank | Arjun Deshpande | Software Engineer | Bengaluru | Savings | 2026-03 to 2026-08 | 534 | 13 |
| 02 | State Bank of India | Sunita Mishra | Government School Teacher | Lucknow | Savings | 2026-03 to 2026-08 | 198 | 6 |
| 03 | Punjab National Bank | Rakesh Agarwal | Kirana Store Owner | Indore | Current | 2026-06 to 2026-08 | 592 | 9 |
| 04 | Kotak Mahindra Bank | Imran Ansari | Food Delivery Partner | Delhi | Savings | 2026-06 to 2026-08 | 314 | 6 |
| 05 | ICICI Bank | Naveen Reddy | Cab Driver | Hyderabad | Savings | 2026-06 to 2026-08 | 224 | 5 |
| 06 | State Bank of India | Sneha Kulkarni | Student | Pune | Savings | 2026-06 to 2026-08 | 148 | 5 |
| 07 | State Bank of India | Ashok Chatterjee | Retired (PSU) | Kolkata | Savings | 2026-03 to 2026-08 | 161 | 5 |
| 08 | Bank of Baroda | Rekha Chauhan | Homemaker | Jaipur | Savings | 2026-06 to 2026-08 | 133 | 3 |
| 09 | Axis Bank | Kamlesh Patel | Textile Trader | Surat | Current | 2026-06 to 2026-08 | 205 | 4 |
| 10 | ICICI Bank | Divya Pillai | Staff Nurse | Kochi | Savings | 2026-03 to 2026-08 | 154 | 4 |
| 11 | HDFC Bank | Pankaj Mehta | Chartered Accountant (Practice) | Mumbai | Savings | 2026-06 to 2026-08 | 149 | 4 |
| 12 | Bank of Baroda | Dattatray Bora | Farmer | Nashik | Savings | 2026-03 to 2026-08 | 122 | 3 |
| 13 | Punjab National Bank | Harpreet Kaur | Bank Probationary Officer | Chandigarh | Savings | 2026-03 to 2026-08 | 206 | 4 |
| 14 | Kotak Mahindra Bank | Nikhil Bhatt | Freelance Product Designer | Goa | Savings | 2026-06 to 2026-08 | 128 | 3 |
| 15 | Axis Bank | Muthu Krishnan | Restaurant Owner | Chennai | Current | 2026-06 to 2026-08 | 445 | 9 |
| 16 | ICICI Bank | Aditya Sethi | Startup Founder | Gurgaon | Savings | 2026-06 to 2026-08 | 122 | 3 |
| 17 | Punjab National Bank | Gurpreet Singh | Factory Machine Operator | Ludhiana | Savings | 2026-06 to 2026-08 | 84 | 2 |
| 18 | HDFC Bank | Dr Shalini Thakur | Consulting Physician | Bhopal | Current | 2026-06 to 2026-08 | 631 | 16 |
| 19 | State Bank of India | Dinesh Prasad | Insurance Agent | Patna | Savings | 2026-03 to 2026-08 | 181 | 5 |
| 20 | Bank of Baroda | Jyoti Saikia | Retail Store Manager | Guwahati | Savings | 2026-08 to 2026-08 | 43 | 1 |

Statement periods vary deliberately: most are 3 months (the standard lender ask),
six are 6 months, one is a single month.

---

## Why each bank looks different

| Bank | Columns | Narration style | Body font |
|------|---------|-----------------|-----------|
| HDFC | Date · Narration · Chq./Ref.No. · Value Dt · Withdrawal · Deposit · Closing Balance | `UPI-SWIGGY-q123@ybl-HDFC-512345678901-PAYMENT ON CREDIT` | Helvetica |
| SBI | Txn Date · Value Date · Description · Ref No./Cheque No. · Debit · Credit · Balance | `TO TRANSFER-UPI/DR/512345678901/Swiggy/SBIN/q123@ybl/Payment` | Helvetica |
| ICICI | S No. · Value Date · Txn Date · Cheque Number · Transaction Remarks · Withdrawal (INR) · Deposit (INR) · Balance (INR) | `UPI/512345678901/Payment to Swiggy/q123@ybl/ICIC` | Helvetica |
| Axis | Tran Date · Chq No · Particulars · Debit · Credit · Balance · Init. Br | `UPI/P2M/512345678901/SWIGGY/q123@ybl` | Helvetica |
| Kotak | Date · Narration · Chq/Ref No · Withdrawal(Dr) · Deposit(Cr) · Balance | `UPI/512345678901/Payment to Swiggy` | Helvetica |
| PNB | Date · Instrument ID · Remarks · Amount(Dr) · Amount(Cr) · Balance | `UPI/DR/512345678901/SWIGGY/PUNB/q123@ybl` | **Courier** |
| BoB | Sr · Txn Date · Value Date · Description · Cheque No · Debit · Credit · Balance | `UPI/512345678901/Paid to SWIGGY/q123@ybl` | Helvetica |

Date formats differ too: `dd/mm/yy` (HDFC), `d Mon yyyy` (SBI), `dd-mm-yyyy`
(ICICI, Axis), `dd-Mon-yy` (Kotak), `dd/mm/yyyy` (PNB, BoB).

---

## Edge cases deliberately included

These are the things that break naive parsers and naive underwriting models:

- **Narration hard-wraps mid-token** across 2–3 lines, exactly as real bank PDFs do —
  a reference number can be split across a line break.
- **NACH/ECS debit returns.** When an EMI hits an account with insufficient funds the
  debit does not appear; instead an `ACH Debit Return Charges … Incl GST` entry does.
  See the gig worker (`04_KOTAK`).
- **Quarterly savings interest** credited on 31-Mar / 30-Jun, computed from average
  daily balance. Current accounts get none.
- **TDS on FD interest** u/s 194A alongside the interest payout (`07_SBI`).
- **Value date ≠ transaction date** on cheque clearing.
- **Foreign inward remittance** plus the bank's handling charge (`14_KOTAK`).
- **DBT / PM-KISAN government credits** (`12_BOB`).
- **Salary dates shift off Sundays**; EMI and SIP dates clamp to short months.
- **Seasonal cash flow** — the farmer's balance peaks after the rabi sale and drains
  through kharif sowing.
- **Cash-heavy, low-digital accounts** (`17_PNB`, `12_BOB`) next to accounts with
  600+ UPI transactions (`18_HDFC`, `03_PNB`).
- **Current accounts** with RTGS in lakhs, cheque runs and GST payments, next to a
  BSBDA account with a four-figure balance.
- **State-correct utility billers** (BESCOM in Bengaluru, MSEDCL in Nashik, APDCL in
  Guwahati) — a real geographic signal a model can pick up or overfit to.

---

## Verification

Every statement passes both checks:

1. Row-level: `balance[n] == balance[n-1] + credit[n] - debit[n]` for all rows.
2. Statement-level: `opening + total_credit - total_debit == closing`, and those totals
   match the summary box printed on page 1.

All 20 PDFs pass `qpdf --check` and are text-extractable with `pdftotext`
(they are vector PDFs, not scans — see the note below).

---

## If you need harder inputs

The PDFs here are digital-native, so text extraction is clean. Real-world intake is
usually messier. To stress an OCR pipeline, degrade them first:

```bash
# scan simulation: rasterise, add noise, re-wrap as PDF
pdftoppm -r 150 -jpeg statement.pdf page
convert page-*.jpg -rotate 0.4 -attenuate 0.35 +noise Gaussian scanned.pdf
```

Password-protected statements (banks send them locked with a DOB/PAN-derived password)
are another common intake case:

```bash
qpdf --encrypt AKUM1985 owner 128 -- statement.pdf locked.pdf
```
