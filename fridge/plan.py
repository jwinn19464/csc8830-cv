"""Fridge Oracle step 5 - urgency and recipe ranking.

urgency(days_left):  expired -> 0 (flag it, don't cook it), today/tomorrow -> 1.0, then decays
                     as exp(-days / 4), so food lasting weeks barely counts.
score(recipe) = sum of urgency of the inventory items it uses
              + 0.15 * (fraction of the recipe you already have)
              - 0.35 * (number of missing ingredients)
Pantry staples (salt, oil, water...) are assumed to be in the kitchen.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import re

RECIPES = json.loads((pathlib.Path(__file__).with_name("recipes.json")).read_text())
STAPLES = {"salt", "pepper", "oil", "olive oil", "water", "sugar", "flour", "vinegar", "soy sauce",
           "garlic powder", "chili flakes", "butter"}


def days_left(expiry: dt.date | None, today: dt.date | None = None):
    if expiry is None:
        return None
    return (expiry - (today or dt.date.today())).days


def urgency(d):
    import math
    if d is None:
        return 0.15
    if d < 0:
        return 0.0
    return 1.0 if d <= 1 else math.exp(-(d - 1) / 4)


def status(d):
    if d is None:
        return "no date"
    if d < 0:
        return "expired"
    if d <= 2:
        return "use now"
    if d <= 6:
        return "use soon"
    return "fine"


def _norm(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z ]", "", s)
    words = []
    for w in s.split():
        if w.endswith("oes") or w.endswith("ches"):
            w = w[:-2]
        elif w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            w = w[:-1]
        words.append(w)
    return " ".join(words)


def matches(item_name, ingredient):
    a, b = _norm(item_name), _norm(ingredient)
    return bool(a) and (a == b or a in b.split() or b in a.split() or (len(b) > 3 and b in a))


def rank(inventory, top=6, today=None):
    """inventory: list of {'name', 'expiry': date|None}. Expired items are not used."""
    inv = []
    for it in inventory:
        d = days_left(it.get("expiry"), today)
        if d is not None and d < 0:
            continue
        if it.get("name"):
            inv.append({**it, "days": d, "u": urgency(d)})
    out = []
    for r in RECIPES:
        used, missing = [], []
        for ing in r["ingredients"]:
            hit = [it for it in inv if matches(it["name"], ing)]
            if hit:
                used.append((ing, max(hit, key=lambda x: x["u"])))
            elif _norm(ing) not in {_norm(s) for s in STAPLES}:
                missing.append(ing)
        if not used:
            continue
        have_frac = len(used) / len(r["ingredients"])
        score = sum(h["u"] for _, h in used) + 0.15 * have_frac - 0.35 * len(missing)
        out.append({"recipe": r["name"], "score": round(score, 3), "minutes": r["minutes"],
                    "uses": [f"{h['name']} ({status(h['days'])})" for _, h in used],
                    "buy": missing, "steps": r["steps"]})
    return sorted(out, key=lambda x: -x["score"])[:top]
