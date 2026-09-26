"""Product seed: 10 sample knowledge sources run through the REAL ingest pipeline."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app import models
from app.ingest import ingest_text
from app.models import utcnow

SEED_DOCS: list[tuple[str, str, str | None]] = [
    # (name, kind, connector_type) -> text
    (
        "Acme Corp - Master Services Agreement",
        "document",
        None,
        """MASTER SERVICES AGREEMENT - Acme Corp and ClusterX.

This agreement was signed on March 15, 2025 and is currently ACTIVE. The contract
was renewed on January 10, 2026 for a further twelve months. The total contract
value is $240,000 per year, billed quarterly at $60,000 per quarter.

The agreement expires on December 31, 2027 unless renewed. Either party may
terminate with 90 days written notice. The service level agreement guarantees
99.9% platform uptime, with service credits of 10% of the monthly fee for each
full percentage point below the target. Acme Corp is our largest enterprise
client, accounting for 18% of annual recurring revenue. The account owner is
Priya Nair. Payment terms are Net 30. Late payments accrue interest at 1.5% per
month. The contract includes a most-favored-customer pricing clause."""
        .replace("  ", " "),
    ),
    (
        "Q3 2026 Revenue by Client",
        "document",
        None,
        """Q3 2026 REVENUE BY CLIENT (all figures in USD).

Acme Corp: $60,000. Globex Inc: $42,500. Initech: $31,200. Umbrella Health:
$28,900. Hooli Systems: $24,750. Stark Manufacturing: $19,300. Wayne Logistics:
$15,600. Massive Dynamic: $12,400. Other clients: $18,850.

Total Q3 2026 revenue: $253,500, up 14% quarter over quarter. Top three clients
(Acme Corp, Globex Inc, Initech) account for 53% of quarterly revenue. Acme Corp
remains the single largest account at $60,000 per quarter under the renewed
Master Services Agreement. Churn was zero in Q3; net revenue retention was 118%
driven by upsells to Globex Inc and Umbrella Health."""
        .replace("  ", " "),
    ),
    (
        "Approved Vendor List 2026",
        "document",
        None,
        """APPROVED VENDOR LIST - 2026.

Cloud infrastructure: CloudServe Hosting (primary, $4,200/month), Northwind
Cloud (backup). Payment processing: PayFlow Inc (2.9% + $0.30 per transaction).
Office supplies: Staples Business (account #88412). Travel: Wanderlust Travel
(preferred corporate rates). Legal: Hart & Associates (retainer $3,000/month).
Accounting: LedgerLine LLP. SaaS tools: Figma, Notion, Linear, 1Password.

All vendors were reviewed and approved by Finance in January 2026. New vendors
require a security questionnaire and approval from the CFO for spend above
$10,000 per year. CloudServe Hosting is our largest vendor by spend at
approximately $50,400 per year. PayFlow Inc processes all customer invoices."""
        .replace("  ", " "),
    ),
    (
        "Invoice and Expense Policy",
        "document",
        None,
        """INVOICE AND EXPENSE POLICY (effective June 1, 2026).

All customer invoices are issued Net 30 and processed through PayFlow Inc.
Expenses under $500 may be approved by a direct manager. Expenses between $500
and $5,000 require department head approval. Expenses above $5,000 require CFO
approval. Travel expenses must be submitted within 14 days of the trip with
itemized receipts. The per-diem rate for domestic travel is $85 per day and
$140 per day for international travel. Software subscriptions over $200 per
month need IT approval before purchase. Reimbursements are paid on the 15th of
each month. Missing receipts above $25 will delay reimbursement until resolved."""
        .replace("  ", " "),
    ),
    (
        "Data Retention Policy",
        "document",
        None,
        """DATA RETENTION POLICY (effective April 2026).

Customer support tickets are retained for 3 years. Financial records and
invoices are retained for 7 years to comply with tax regulations. Employee
records are retained for 5 years after termination. Marketing email lists
honor unsubscribe requests within 48 hours and purge suppressed addresses
after 2 years. Database backups are kept for 90 days on a rolling basis. Log
files are retained for 1 year. Contract documents are retained for 7 years
after expiry. Data deletion requests from customers are completed within 30
days of verification. All deletions follow NIST 800-88 media sanitization
guidelines for production storage."""
        .replace("  ", " "),
    ),
    (
        "Globex Inc - Statement of Work",
        "document",
        None,
        """STATEMENT OF WORK - Globex Inc. Project: analytics dashboard rollout.

Signed August 2, 2026. Status: IN PROGRESS. Project value: $85,000 fixed fee.
Milestone 1 (discovery) completed August 30, 2026, $25,000 paid. Milestone 2
(beta delivery) due October 15, 2026, $35,000. Milestone 3 (go-live) due
November 30, 2026, $25,000. The Globex Inc account owner is Marcus Chen.
Scope includes five dashboards, SSO integration, and training for 20 users.
Out of scope: mobile app development and data migration from legacy systems.
Weekly status calls are held every Tuesday. Change requests are billed at
$180 per hour with written approval required."""
        .replace("  ", " "),
    ),
    (
        "Remote Work Security Policy",
        "document",
        None,
        """REMOTE WORK SECURITY POLICY (IT, effective May 2026).

All employees working remotely must use a company-managed device with full-disk
encryption. VPN is required when accessing internal systems from public
networks. Passwords must be at least 16 characters and stored in 1Password;
password reuse across services is prohibited. Multi-factor authentication is
mandatory for email, code repositories, and production infrastructure. Screens
must lock after 5 minutes of inactivity. Public Wi-Fi may only be used with
the corporate VPN active. Security incidents must be reported to IT within 4
hours of discovery. Annual security training is required for all staff."""
        .replace("  ", " "),
    ),
    (
        "CloudServe Hosting - Vendor Contract",
        "document",
        None,
        """VENDOR CONTRACT - CloudServe Hosting.

Signed February 1, 2026. Status: ACTIVE. Monthly spend: $4,200 for the
production cluster, billed annually at $50,400 with a 5% prepay discount. Term:
24 months, expiring January 31, 2028. Includes 99.95% uptime SLA, 24/7 support
with 1-hour response for critical incidents, and 10 TB of included bandwidth
per month. Overage bandwidth is billed at $0.08 per GB. The contract auto-
renews for 12-month terms unless cancelled 60 days before expiry. Early
termination fee is 50% of the remaining contract value. Account manager: Dana
Whitfield."""
        .replace("  ", " "),
    ),
    (
        "Employee Leave Policy",
        "document",
        None,
        """EMPLOYEE LEAVE POLICY (HR, effective January 2026).

Full-time employees receive 20 days of paid annual leave per year, accrued
monthly. Unused leave up to 5 days may be carried over to the next year.
Sick leave: 10 days per year, no carryover. Parental leave: 12 weeks paid for
the primary caregiver, 4 weeks for the secondary caregiver. Leave requests
longer than 3 days require manager approval at least 2 weeks in advance.
Public holidays follow the Philippines national calendar plus December 24 and
December 31. Unpaid leave may be granted at manager discretion for up to 30
days per year."""
        .replace("  ", " "),
    ),
    (
        "Q3 2026 Sales Pipeline Summary",
        "document",
        None,
        """Q3 2026 SALES PIPELINE SUMMARY.

Total pipeline value: $1.2M across 34 open opportunities. Stage breakdown:
Discovery $380K (12 deals), Evaluation $410K (11 deals), Negotiation $290K
(7 deals), Contract $120K (4 deals). Top opportunities: Cyberdyne Systems
$150K (Negotiation), Tyrell Corp $120K (Evaluation), Soylent Corp $95K
(Discovery). Average deal size: $35,300. Win rate in Q3: 31%. Average sales
cycle: 47 days. The pipeline covers 4.7x the Q4 quota of $255,000. Two deals
over $100K slipped from Q3 into Q4: Cyberdyne Systems and Tyrell Corp."""
        .replace("  ", " "),
    ),
]

# Connector records seeded as real Source rows (kind=connector) with staged docs.
SEED_CONNECTORS: list[tuple[str, str, str]] = [
    # (name, connector_type, text) - text goes into data/staging/<slug>/ and is synced
    (
        "Salesforce CRM",
        "crm",
        """CRM EXPORT - key accounts (synced September 26, 2026).

Acme Corp: status Customer, ARR $240,000, health score 92, last contact Sep 20.
Globex Inc: status Customer, ARR $170,000, health score 81, last contact Sep 24.
Initech: status Customer, ARR $124,800, health score 77, last contact Sep 18.
Cyberdyne Systems: status Prospect, pipeline $150,000, stage Negotiation.
Tyrell Corp: status Prospect, pipeline $120,000, stage Evaluation.""",
    ),
    (
        "NetSuite ERP",
        "erp",
        """ERP EXPORT - open invoices (synced September 26, 2026).

INV-2026-0912 Acme Corp $60,000 due Oct 15, 2026 - SENT.
INV-2026-0918 Globex Inc $35,000 due Oct 30, 2026 - SENT (milestone 2).
INV-2026-0921 Initech $31,200 due Oct 12, 2026 - OVERDUE by 15 days, follow up.
Total accounts receivable: $126,200. Overdue: $31,200 (Initech).""",
    ),
    (
        "Google Drive",
        "gdrive",
        """GOOGLE DRIVE INDEX (synced September 26, 2026).

/Contracts/Acme Corp MSA signed.pdf - 24 pages, signed Mar 15 2025.
/Contracts/CloudServe vendor agreement.pdf - 11 pages.
/Finance/Q3 2026 board deck.pptx - 32 slides.
/Product/Roadmap 2026-2027.md - last edited Sep 10 2026.
/HR/Offer letter templates/ - 4 documents.""",
    ),
]


def _slug(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def seed_product(db: Session) -> dict:
    """Idempotent seed: 10 documents + 3 connectors with staged files, all ingested."""
    existing = {s.name for s in db.query(models.Source).all()}
    counts = {"sources": 0, "chunks": 0}

    for name, kind, connector_type, text in SEED_DOCS:
        if name in existing:
            continue
        src = models.Source(name=name, kind=kind, connector_type=connector_type,
                            status="ready", last_sync_at=utcnow())
        db.add(src)
        db.flush()
        counts["chunks"] += ingest_text(db, src, text)
        counts["sources"] += 1

    # Connectors: stage a file under data/staging/<slug>/ then sync it for real.
    import os

    staging_root = os.environ.get("EBRAIN_STAGING", "data/staging")
    for name, ctype, text in SEED_CONNECTORS:
        if name in existing:
            continue
        slug = _slug(name)
        sdir = os.path.join(staging_root, slug)
        os.makedirs(sdir, exist_ok=True)
        fname = f"{slug}-export.txt"
        with open(os.path.join(sdir, fname), "w", encoding="utf-8") as f:
            f.write(text)
        src = models.Source(name=name, kind="connector", connector_type=ctype,
                            status="ready", last_sync_at=utcnow())
        db.add(src)
        db.flush()
        from app.product import sync_connector  # local import: product defines router

        counts["chunks"] += sync_connector(db, src)
        counts["sources"] += 1

    db.commit()
    return counts
