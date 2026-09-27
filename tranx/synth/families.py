"""Brand-family grouping and stable hash assignment shared by the synth and the splits."""
_STOP = {"the", "of", "and", "s", "for"}


def _tokens(name: str) -> frozenset:
    import re
    return frozenset(t for t in re.findall(r"[a-z0-9]+", name.lower()) if t not in _STOP)


# Sibling banners whose names share no token subset.
_FAMILY_LINKS = [("Saks Fifth Avenue", "Saks Off 5th")]


def brand_families(merchants: list[str]) -> dict[str, str]:
    """Group merchant labels that name the same brand family: one label's tokens are
    a subset of the other's ("Walmart" / "Walmart Pharmacy", "Nordstrom" /
    "Nordstrom Rack"), plus explicit _FAMILY_LINKS. A one-word generic label that
    occurs inside >= 3 other labels ("Pharmacy", "Fitness", "Bank") is a hub, not a
    brand: it does not link, otherwise it would chain Kroger, Safeway and Walmart
    into one family. Returns merchant -> family key (smallest name in the group)."""
    parent = {m: m for m in merchants}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    toks = {m: _tokens(m) for m in merchants}
    hubs = {m for m in merchants if len(toks[m]) == 1
            and sum(toks[m] <= toks[o] for o in merchants if o != m) >= 3}
    for i, a in enumerate(merchants):
        for b in merchants[i + 1:]:
            ta, tb = toks[a], toks[b]
            if a in hubs or b in hubs or not ta or not tb:
                continue
            if ta <= tb or tb <= ta:
                parent[find(a)] = find(b)
    for a, b in _FAMILY_LINKS:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for m in merchants:
        groups.setdefault(find(m), []).append(m)
    return {m: min(g) for g in groups.values() for m in g}


def _in_eval(key: str, frac: float) -> bool:
    """Stable hash assignment: a key's side of the split never depends on which other
    keys exist, so label edits elsewhere do not reshuffle the held-out set."""
    import hashlib
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % 1000 < frac * 1000
