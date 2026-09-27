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
