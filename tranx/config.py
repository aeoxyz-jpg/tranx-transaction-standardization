from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

HF_DATASET = "mitulshah/transaction-categorization"
SAMPLE_SIZE = 100_000
N_CUSTOMERS = 1_500
MCC_COVERAGE = 0.60
MCC_NOISE = 0.15  # fraction of present MCCs drawn from a wrong (neighboring) category

CATEGORIES = [
    "Income",
    "Food & Dining",
    "Healthcare & Medical",
    "Shopping & Retail",
    "Utilities & Services",
    "Entertainment & Recreation",
    "Financial Services",
    "Charity & Donations",
    "Transportation",
    "Government & Legal",
]

COUNTRIES = ["USA", "UK", "CANADA", "AUSTRALIA", "INDIA"]

PAYMENT_METHODS = [
    "credit_card", "debit_card", "ach", "wire", "check", "cash_app", "transfer",
]

# Model routes
EMBED_MODEL = "all-MiniLM-L6-v2"
SLM_MODEL = "qwen2.5:3b-instruct"
OLLAMA_URL = "http://127.0.0.1:11434"
MLX_BASE = "mlx-community/Qwen2.5-3B-Instruct-4bit"  # base for the LoRA route

# Noise vocabulary used to derive canonical merchant names (see synth/canonical.py)
FORMAT_WORDS = [
    # multi-word phrases first: the regex alternation matches in list order
    "Shopping Center", "Business District",
    "Online", "Store", "Branch", "Center", "Station",
    "Strip Mall", "Mall", "Downtown", "Residential", "Airport", "Campus",
]
TIME_PHRASES = [
    "Night", "Evening", "Afternoon", "Morning", "Rush Hour",
    "Lunch Time", "Dinner Time", "Weekend", "Weekday", "Holiday",
]
# "Hospital" is both a location suffix in the source templates ("Donation Hospital",
# "Burlington Store Branch Hospital") and part of real merchant names. It is
# stripped only as a trailing word after another token, except for these names.
HOSPITAL_MERCHANTS = {"Children's Hospital"}
# Source labels whose name contains a noise word (FORMAT_WORDS / TIME_PHRASES); they
# are protected from stripping ("Online Bank" is not "Bank", "Senior Center" is not
# "Senior"). In the source these words occur in 100% of the label's rows.
PROTECTED_NAMES = ["Community Center", "Senior Center", "Convention Center",
                   "Online Bank", "Holiday Pay"]
# Gold-label merges: distinct source labels that are the same business.
LABEL_MERGES = {"Cane's": "Raising Cane's"}
# One source label that is two businesses, told apart by the source category.
CATEGORY_SPLIT_LABELS = {
    "Frontier": {"Utilities & Services": "Frontier Communications",
                 "Transportation": "Frontier Airlines"},
}

# Source labels that name a kind of money movement rather than a counterparty
# business (pay types, transfers, loans, donations, fees). Their gold merchant is
# null and they are excluded from merchant metrics; category is still scored.
# Unnamed *service providers* ("Bank", "Cable Company", "Pharmacy") are kept:
# those transactions do have a merchant, the descriptor just does not name it.
TRANSACTION_TYPE_LABELS = {
    # Income
    "Annuity", "Bonus", "Capital Gains", "Cashback", "Commission", "Consulting",
    "Contract Work", "Credit", "Deposit", "Direct Deposit", "Disability", "Dividend",
    "Franchise", "Freelance", "Full-time", "Gig Work", "Income", "Interest", "Licensing",
    "Maternity Leave", "Overtime", "Part-time", "Paternity Leave", "Holiday Pay", "Payroll",
    "Pension", "Refund", "Reimbursement", "Rental Income", "Reward", "Royalty", "Salary",
    "Sick Pay", "Side Hustle", "Social Security", "Unemployment", "Vacation Pay", "Wage",
    # Financial flows
    "ATM", "Transfer", "Loan", "Mortgage", "Investment", "Stock", "Mutual Fund", "Retirement",
    # Donations
    "Donation", "Offering", "Gift", "Fundraising",
    # Government fees
    "Tax", "Permit", "License", "Registration", "Visa", "Passport",
}
# Source labels that name what was bought (a procedure, service product, activity or
# fee item) rather than who was paid. Also merchant-less (user ruling 2026-09-27).
# Unnamed providers and venues stay merchants: Hospital, Lab, Urgent Care,
# Cardiologist, Pharmacy, Cable Company, Gym, Cinema, Zoo, Bank, Church.
PURCHASE_ITEM_LABELS = {
    # medical procedures and care types
    "Blood Test", "MRI", "X-Ray", "Surgery", "Dental", "Eye Care", "Physical Therapy",
    "Occupational Therapy", "Speech Therapy", "Primary Care",
    # transport services and fees
    "Toll", "Parking", "Bus", "Train", "Taxi",
    # events and activities
    "Concert", "Festival", "Fair", "Carnival", "Go-Kart", "Laser Tag", "Paintball",
    "Mini Golf", "Rock Climbing", "Pilates", "Fitness",
    # connectivity products
    "Broadband", "DSL", "Fiber", "WiFi", "Mobile", "Cable Internet", "Satellite Internet",
    "Mobile Hotspot", "Public WiFi", "Business WiFi", "WiFi Hotspot",
    # financial and professional services
    "Insurance", "Banking", "Accounting", "Bookkeeping", "Tax Preparation",
    "Wealth Management", "Retirement Planning",
    # generic services
    "Legal", "Service",
}
# Every label with no merchant; the gold keeps it in the txn_type column.
NON_MERCHANT_LABELS = TRANSACTION_TYPE_LABELS | PURCHASE_ITEM_LABELS
# Hard-mode descriptor synthesis: mimic the dirt real card-network descriptors
# carry (aggregator prefixes, embedded geo/store ids, truncation, uppercasing).
HARD_PREFIXES = [
    "SQ *", "TST* ", "PP*", "PAYPAL *", "SP * ", "POS DEBIT ",
    "PURCHASE ", "DEBIT CARD PURCHASE ", "ACH ",
]
HARD_CITIES = [
    "ATLANTA", "NEW YORK", "SAN JOSE", "CHICAGO", "AUSTIN",
    "SEATTLE", "MIAMI", "DENVER", "BOSTON", "PHOENIX",
]
HARD_REGIONS = ["GA", "NY", "CA", "IL", "TX", "WA", "FL", "CO", "MA", "AZ"]

# Maps parenthetical payment hints in descriptions to canonical payment methods
PAREN_PAYMENT_HINTS = {
    "Cash": "cash_app",
    "App": "cash_app",
    "Digital Wallet": "credit_card",
    "Contactless": "credit_card",
    "ACH": "ach",
    "Bank Transfer": "transfer",
}

# Synthetic realism (spec 2026-09-28): fictional local merchants, abbreviation noise,
# repeat structure, and the two evaluation views.
LOCAL_MERCHANTS_N = 800
LOCAL_MERCHANT_SHARE = 0.25   # stable-hash share of merchant rows reassigned to a local
ABBREV_P = 0.15               # per-descriptor chance of dropping interior vowels
LOCATIONS_ROWS_PER = 3        # a merchant gets ceil(rows / this) locations ...
LOCATIONS_CAP = 200           # ... capped here
LOCATIONS_ZIPF_S = 1.5        # row -> location draw is Zipf-weighted (tail of one-offs)
# High enough that no model-view descriptor is sampled away (random 1,496, unseen 2,833).
EVAL_CAPS = {"random": 5000, "unseen": 5000}
BOOTSTRAP_N = 4000
RUN_DIR = REPORTS_DIR / "run"
PREDS_DIR = REPORTS_DIR / "preds"
# Gold columns produced by synth.feed.build_feed; origin is "source" / "local" (null
# for merchant-less rows); noise flags are False outside hard mode.
GOLD_COLUMNS = ["txn_id", "category", "canonical_merchant", "txn_type", "direction",
                "origin", "noise_abbrev", "noise_trunc"]
