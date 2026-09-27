"""Fictional local merchants: one-off businesses no model saw in pretraining, so the
eval can test recovery of an unknown name from a noisy descriptor."""
import random
from tranx import config
from tranx.synth.families import _tokens

# Word lists avoid descriptor noise words (FORMAT_WORDS, TIME_PHRASES, countries) so
# the string cleaner cannot strip part of a local name; any remaining clash with a
# source label is rejected at generation time.
_SURNAMES = [
    "Ashford", "Brennan", "Calloway", "Delacroix", "Fenwick", "Galloway", "Hargrove",
    "Ingram", "Jessop", "Kessler", "Lindqvist", "Marlowe", "Northcott", "Okafor",
    "Pembrook", "Quimby", "Rourke", "Sandoval", "Thackeray", "Underhill", "Vasquez",
    "Whitlock", "Yardley", "Zeller", "Abernathy", "Birkett", "Castellano", "Dunmore",
    "Ellery", "Fairbairn", "Gorski", "Holloway", "Iverson", "Kowalski", "Lachance",
    "Moreau", "Nakamura", "Oyelaran", "Petrakis", "Rasmussen", "Szabo", "Tremblay",
    "Varga", "Winslow", "Achebe", "Bellamy", "Cardenas", "Donnelly",
]
_PLACES = [
    "Millbrook", "Ashgrove", "Fernhill", "Oakridge", "Willowdale", "Brookhaven",
    "Stonebridge", "Cedarvale", "Maplewood", "Riverton", "Elmstead", "Pinecrest",
    "Harborview", "Lakeshore", "Foxhollow", "Glenmore", "Hollowell", "Kingsbury",
    "Larkfield", "Moorland", "Northfield", "Redcliff", "Sandhurst", "Thornbury",
    "Westbrook", "Ambleside", "Bramley", "Crestwood",
]
_ADJECTIVES = [
    "Blue", "Rustic", "Copper", "Little", "Humble", "Crimson", "Silver", "Hidden",
    "Sunny", "Wild", "Golden", "Velvet",
]
_ANIMALS = [
    "Heron", "Fox", "Badger", "Otter", "Sparrow", "Magpie", "Hare", "Wren", "Stag",
    "Kestrel",
]

_NOUNS = {
    "Food & Dining": ["Bakery", "Kitchen", "Bistro", "Diner", "Deli", "Tavern",
                      "Cantina", "Eatery", "Smokehouse", "Creperie", "Tearoom",
                      "Brasserie", "Patisserie", "Trattoria"],
    "Healthcare & Medical": ["Orthodontics", "Podiatry", "Optometry", "Physiotherapy",
                             "Wellness", "Apothecary", "Dentistry", "Audiology",
                             "Chiropractic", "Dermatology"],
    "Shopping & Retail": ["Hardware", "Boutique", "Florist", "Bookshop", "Outfitters",
                          "Mercantile", "Haberdashery", "Emporium", "Jewelers",
                          "Tailors", "Toyshop", "Antiques"],
    "Utilities & Services": ["Plumbing", "Roofing", "Landscaping", "Pest Control",
                             "Locksmiths", "Laundromat", "Dry Cleaners", "Propane",
                             "Septic", "Heating & Cooling", "Waste Removal", "Glaziers"],
    "Entertainment & Recreation": ["Playhouse", "Billiards", "Karaoke Lounge",
                                   "Skate Rink", "Cineplex", "Axe Throwing",
                                   "Riding Stables", "Marina", "Dance Academy",
                                   "Bouldering", "Speedway", "Puppet Theatre"],
    "Charity & Donations": ["Foundation", "Hospice", "Rescue Mission", "Literacy Trust",
                            "Memorial Fund", "Soup Kitchen", "Arts Council",
                            "Scholarship Fund", "Relief Fund", "Wildlife Trust"],
    "Transportation": ["Cab Co", "Auto Repair", "Tire & Lube", "Towing", "Ferry",
                       "Shuttle", "Garage", "Motors", "Coachworks", "Rickshaw"],
    "Government & Legal": ["Solicitors", "Law Group", "Attorneys", "Township",
                           "Borough Council", "Recorder of Deeds", "Magistrate",
                           "Paralegal", "Conveyancing", "Chambers"],
}

# Templates per category; {n} is a category noun.
_TEMPLATES = {
    "Food & Dining": ["{s}'s {n}", "{p} {a} {n}", "{a} {x} {n}"],
    "Healthcare & Medical": ["{s} {n}", "{p} {n}", "{s} & {s2} {n}"],
    "Shopping & Retail": ["{s}'s {n}", "{p} {n}", "{a} {x} {n}"],
    "Utilities & Services": ["{s} {n}", "{p} {n}", "{s} & Sons {n}"],
    "Entertainment & Recreation": ["{p} {n}", "{a} {x} {n}", "{s}'s {n}"],
    "Charity & Donations": ["{s} {n}", "{p} {n}", "Friends of {p} {n}"],
    "Transportation": ["{s} {n}", "{p} {n}", "{s} & Sons {n}"],
    "Government & Legal": ["{s} & {s2} {n}", "{p} {n}", "{s} {n}"],
}

LOCAL_CATEGORIES = [c for c in config.CATEGORIES if c in _NOUNS]


def local_merchants(source_labels: list[str], seed: int, n: int) -> list[tuple[str, str]]:
    """n fictional (name, category) pairs, split evenly over LOCAL_CATEGORIES. A name
    sharing any token with a source label is rejected, so locals never join a source
    brand family."""
    banned = set().union(*[_tokens(s) for s in source_labels]) if source_labels else set()
    rng = random.Random(seed)
    k = len(LOCAL_CATEGORIES)
    out, seen = [], set()
    for ci, cat in enumerate(LOCAL_CATEGORIES):
        quota = n // k + (1 if ci < n % k else 0)
        got, tries = 0, 0
        while got < quota:
            tries += 1
            if tries > 200 * quota + 1000:
                raise RuntimeError(f"cannot generate {quota} local names for {cat}")
            s, s2 = rng.sample(_SURNAMES, 2)
            name = rng.choice(_TEMPLATES[cat]).format(
                s=s, s2=s2, p=rng.choice(_PLACES), a=rng.choice(_ADJECTIVES),
                x=rng.choice(_ANIMALS), n=rng.choice(_NOUNS[cat]))
            if name in seen or _tokens(name) & banned:
                continue
            seen.add(name)
            out.append((name, cat))
            got += 1
    return out
