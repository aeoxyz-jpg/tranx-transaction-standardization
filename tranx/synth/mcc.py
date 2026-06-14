import random

# Real ISO 18245 merchant category codes grouped by our category labels.
# Some codes intentionally appear under multiple categories: real MCCs are
# many-to-many (5411 supermarkets sell food and general merchandise, 5912
# pharmacies sell retail goods, 6011 ATMs serve both income and finance). This
# overlap means MCC -> category is a noisy prior, not a clean oracle.
CATEGORY_MCC = {
    "Income": [6011, 6012],                      # ATM / financial institution credits
    "Food & Dining": [5812, 5814, 5499, 5411],   # restaurants, fast food, grocery
    "Healthcare & Medical": [8011, 8062, 5912],  # doctors, hospitals, pharmacy
    "Shopping & Retail": [5311, 5651, 5732, 5411, 5912, 5999],  # dept, apparel, grocery, pharmacy, misc
    "Utilities & Services": [4900, 4814, 4899],  # utilities, telecom, cable
    "Entertainment & Recreation": [7832, 7997, 7941, 5999],  # cinema, clubs, sports, misc retail
    "Financial Services": [6012, 6051, 6300, 6011],  # financial inst, quasi-cash, insurance, ATM
    "Charity & Donations": [8398, 8661],         # charity, religious orgs
    "Transportation": [4111, 4121, 4511, 4814],  # transit, taxi, airlines, telecom/telematics
    "Government & Legal": [9399, 9211, 9222],    # government, court, fines
}


def mcc_for(category: str, rng: random.Random) -> int | None:
    """Pick a plausible MCC for a category, or None if the category is unknown."""
    options = CATEGORY_MCC.get(category)
    if not options:
        return None
    return rng.choice(options)
