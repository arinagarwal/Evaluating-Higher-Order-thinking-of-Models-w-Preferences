from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'final'))
from dishes import DISHES
from config import BANNED_INGREDIENTS, CUISINE_RANGES, CUISINE_RISK_MAP

# The tail of dishes.py (indices 900–999) is an unlabelled "combined" section
# with dishes from all cuisines mixed together. The index-based CUISINE_RANGES
# cannot handle this, so we maintain an explicit lookup for those dishes.
_COMBINED_CUISINE: dict[str, str] = {
    "Bagna Cauda": "Italian", "Pandoro": "Italian", "Panettone": "Italian",
    "Zabaglione": "Italian", "Stracciatella Soup": "Italian",
    "Tonkotsu Ramen": "Japanese", "Chicken Katsu Curry": "Japanese",
    "Sushi Nigiri Platter": "Japanese", "Tempura Udon": "Japanese",
    "Miso Soup": "Japanese",
    "Tacos al Pastor": "Mexican", "Chicken Enchiladas": "Mexican",
    "Guacamole": "Mexican", "Mole Poblano with Turkey": "Mexican",
    "Chicken Tikka Masala": "Indian", "Palak Paneer": "Indian",
    "Butter Chicken": "Indian", "Biryani": "Indian",
    "Pad Thai": "Thai", "Green Curry with Chicken": "Thai",
    "Tom Yum Goong": "Thai", "Massaman Curry": "Thai", "Som Tum": "Thai",
    "Kung Pao Chicken": "Chinese", "Mapo Tofu": "Chinese",
    "Peking Duck": "Chinese", "Dim Sum Platter": "Chinese",
    "Char Siu Pork": "Chinese",
    "Coq au Vin": "French", "Beef Bourguignon": "French",
    "Ratatouille": "French", "Croque Monsieur": "French",
    "French Onion Soup": "French",
    "Bibimbap": "Korean", "Kimchi Jjigae": "Korean", "Bulgogi": "Korean",
    "Japchae": "Korean", "Tteokbokki": "Korean", "Samgyeopsal": "Korean",
    "Doro Wat": "Ethiopian", "Injera with Misir Wat": "Ethiopian",
    "Kitfo": "Ethiopian", "Tibs": "Ethiopian", "Shiro Wat": "Ethiopian",
    "Gomen": "Ethiopian",
    "Chicken Tagine with Preserved Lemons": "Moroccan",
    "Lamb Tagine with Prunes": "Moroccan", "Couscous Royale": "Moroccan",
    "Ceviche de Pescado": "Peruvian", "Lomo Saltado": "Peruvian",
    "Aji de Gallina": "Peruvian", "Anticuchos": "Peruvian",
    "Causa Limeña": "Peruvian",
    "Iskender Kebab": "Turkish", "Lahmacun": "Turkish", "Pide": "Turkish",
    "Manti": "Turkish", "Imam Bayildi": "Turkish", "Karniyarik": "Turkish",
    "Pho Bo": "Vietnamese", "Banh Mi": "Vietnamese", "Bun Cha": "Vietnamese",
    "Goi Cuon": "Vietnamese", "Com Tam": "Vietnamese",
    "Bun Bo Hue": "Vietnamese", "Cao Lau": "Vietnamese",
    "Moussaka": "Greek", "Souvlaki": "Greek", "Spanakopita": "Greek",
    "Tzatziki": "Greek", "Pastitsio": "Greek", "Dolmades": "Greek",
    "Feijoada": "Brazilian", "Pão de Queijo": "Brazilian",
    "Coxinha": "Brazilian", "Moqueca de Peixe": "Brazilian",
    "Picanha": "Brazilian", "Brigadeiro": "Brazilian",
    "Swedish Meatballs": "Scandinavian", "Gravlax": "Scandinavian",
    "Smørrebrød": "Scandinavian", "Janssons Frestelse": "Scandinavian",
    "Pierogi Ruskie": "Polish", "Bigos": "Polish",
    "Żurek": "Polish", "Barszcz Czerwony": "Polish",
    "Jerk Chicken": "Caribbean", "Ackee and Saltfish": "Caribbean",
    "Rice and Peas": "Caribbean", "Oxtail Stew": "Caribbean",
    "Curry Goat": "Caribbean", "Roti with Curry Chicken": "Caribbean",
    "Nasi Goreng": "Indonesian", "Rendang": "Indonesian",
    "Satay Ayam": "Indonesian", "Gado-Gado": "Indonesian",
    "Soto Ayam": "Indonesian", "Bakso": "Indonesian",
    "Schnitzel": "German", "Bratwurst with Sauerkraut": "German",
    "Sauerbraten": "German", "Spätzle": "German", "Kartoffelpuffer": "German",
}


def infer_cuisine(dish: str) -> str:
    """
    Map a dish to its cuisine.
    First checks the explicit combined-section lookup (dishes 900-999 from
    dishes.py which have no cuisine markers in the file). Falls back to the
    index-based CUISINE_RANGES for the main labeled blocks (0-899).
    """
    if dish in _COMBINED_CUISINE:
        return _COMBINED_CUISINE[dish]
    try:
        idx = DISHES.index(dish)
    except ValueError:
        return "Unknown"
    for start, end, cuisine in CUISINE_RANGES:
        if start <= idx < end:
            return cuisine
    return "Unknown"


class Receptor:
    """
    Converts a raw dish request into a structured schema consumed by the
    Unconsciousness module. Analogous to the CoCoMo receptor module that
    processes sensor input into workspace representations.
    """

    def process(self, dish: str, past_substitutions=None) -> dict:
        cuisine = infer_cuisine(dish)
        risk_score = CUISINE_RISK_MAP.get(cuisine, CUISINE_RISK_MAP["Unknown"])

        return {
            "dish": dish,
            "cuisine": cuisine,
            "constraints": BANNED_INGREDIENTS,
            "risk_score": risk_score,
            "past_substitutions": past_substitutions or [],
        }
