"""
Static structural features per country.

These are slow-moving country-level signals that don't change from query to
query. They provide the model with a prior before it reads any news — the
equivalent of "what's the baseline here?" without relying on live data.

Three features per country:

  conflict_baserate  — fraction of years 2000-2023 with active armed conflict
                       (>=25 battle deaths/year), sourced from UCDP ACD 2023.
                       Range: 0.0 (no recorded conflict) to 1.0 (conflict every year).

  polity_norm        — Polity5 score (2018, most recent stable year) normalized
                       from [-10, +10] to [-1.0, +1.0].
                       -1.0 = full autocracy, +1.0 = full democracy.
                       Source: Center for Systemic Peace, Polity5 dataset.

  mil_spending_norm  — Military spending as % of GDP (SIPRI 2022),
                       normalized by dividing by 10 (so 5% GDP → 0.5).
                       Capped at 1.0.

Default for unknown countries: baserate=0.15, polity=0.0, mil_spending=0.15
(world medians — conservative neutral prior).

IMPORTANT: These are approximate values for modeling purposes. Do not use
for policy analysis. Update when Polity6/UCDP refreshes.
"""
from __future__ import annotations

# ── Country data table ────────────────────────────────────────────────────────
# Format: "Country": (conflict_baserate, polity_norm, mil_spending_norm)
# polity_norm  = polity_score / 10
# mil_spending_norm = mil_pct_gdp / 10  (so 5% → 0.5)

_COUNTRY_DATA: dict[str, tuple[float, float, float]] = {
    # ── Active/recent conflict zones ──────────────────────────────────────────
    "Afghanistan":     (0.96, -0.7,  0.10),   # UCDP every year 2000-2021; Polity=-7
    "Syria":           (0.70, -0.9,  0.40),   # conflict 2011-; Polity=-9
    "Yemen":           (0.65, -0.2,  0.40),   # conflict 2015-; Polity=-2
    "Iraq":            (0.91, 0.0,   0.30),   # UCDP most years; Polity=0
    "Somalia":         (1.00, -0.9,  0.06),   # continuous conflict
    "Sudan":           (0.74, -0.7,  0.20),   # Darfur + civil war cycles
    "South Sudan":     (0.75, -0.7,  0.30),
    "Libya":           (0.50, -0.8,  0.30),   # post-2011 instability
    "Mali":            (0.48, 0.0,   0.20),   # Sahel insurgency
    "Central African Republic": (0.57, -0.6, 0.10),
    "Democratic Republic of the Congo": (0.87, -0.6, 0.10),
    "Mozambique":      (0.35, 0.6,   0.10),
    "Ethiopia":        (0.61, -0.3,  0.10),   # Tigray conflict 2020+
    "Nigeria":         (0.35, 0.5,   0.04),   # Boko Haram, Niger Delta
    "Cameroon":        (0.30, -0.3,  0.12),
    "Myanmar":         (0.70, -0.6,  0.30),   # decades of civil war
    "Colombia":        (0.96, 0.8,   0.30),   # FARC conflict (mostly resolved 2016)
    "Philippines":     (0.57, 0.6,   0.11),   # NPA insurgency + Mindanao
    "Pakistan":        (0.70, 0.4,   0.40),   # FATA/TTP insurgency
    "Ukraine":         (0.35, 0.6,   0.50),   # Donbas 2014+, invasion 2022
    "Russia":          (0.26, -0.7,  0.41),   # Chechnya cycles; Polity=-7
    "Israel":          (0.87, 1.0,   0.56),   # Polity=+10; high mil spending
    "Palestine":       (0.80, -0.6,  0.10),
    "Gaza":            (0.90, -0.6,  0.10),   # alias
    "Lebanon":         (0.35, 0.4,   0.45),
    "Iran":            (0.22, -0.7,  0.24),   # Polity=-7; proxy wars
    "Turkey":          (0.35, 0.2,   0.17),   # PKK conflict; Syria ops
    "Azerbaijan":      (0.35, -0.7,  0.56),   # Nagorno-Karabakh
    "Armenia":         (0.26, 0.4,   0.45),
    "Georgia":         (0.30, 0.7,   0.22),   # 2008 war + Abkhazia
    # ── Medium-risk: internal tensions or regional rivalries ──────────────────
    "India":           (0.22, 0.9,   0.25),   # Kashmir, Maoist insurgency
    "China":           (0.04, -0.7,  0.17),   # Xinjiang; Polity=-7
    "North Korea":     (0.04, -1.0,  2.00),   # Polity=-10; mil spending ~20% GDP → cap 1.0
    "South Korea":     (0.09, 0.8,   0.27),
    "Taiwan":          (0.00, 0.8,   0.22),
    "Venezuela":       (0.09, -0.3,  0.16),
    "Bolivia":         (0.00, 0.7,   0.14),
    "Indonesia":       (0.22, 0.8,   0.08),   # Papua + Aceh (historical)
    "Thailand":        (0.22, 0.3,   0.13),   # Southern insurgency + coups
    "Bangladesh":      (0.04, 0.2,   0.13),
    "Sri Lanka":       (0.39, 0.7,   0.20),   # Tamil war ended 2009
    "Nepal":           (0.22, 0.7,   0.14),   # Maoist insurgency (ended 2006)
    "Cambodia":        (0.09, -0.5,  0.20),
    "Saudi Arabia":    (0.13, -0.8,  0.60),   # Yemen intervention
    "United Arab Emirates": (0.04, -0.7, 0.56),
    "Egypt":           (0.13, -0.6,  0.18),   # Sinai insurgency
    "Algeria":         (0.04, -0.5,  0.60),
    "Morocco":         (0.04, -0.3,  0.31),
    "Tunisia":         (0.09, 0.4,   0.22),
    "Burkina Faso":    (0.35, 0.0,   0.18),   # Sahel jihadist expansion
    "Niger":           (0.26, -0.5,  0.19),
    "Chad":            (0.52, -0.7,  0.18),
    "Senegal":         (0.13, 0.7,   0.17),   # Casamance
    "Uganda":          (0.26, -0.2,  0.26),
    "Kenya":           (0.22, 0.6,   0.12),
    "Tanzania":        (0.09, 0.3,   0.10),
    "Zimbabwe":        (0.00, -0.3,  0.19),
    "Zambia":          (0.00, 0.5,   0.13),
    "Angola":          (0.52, -0.2,  0.18),   # civil war ended 2002; post-conflict
    "Mexico":          (0.09, 0.6,   0.05),   # cartel violence ≠ armed conflict in UCDP
    "Guatemala":       (0.04, 0.5,   0.04),
    "Haiti":           (0.13, -0.3,  0.07),
    "Cuba":            (0.00, -0.9,  0.00),   # Polity=-9
    "Brazil":          (0.00, 0.8,   0.14),
    "Argentina":       (0.00, 0.8,   0.07),
    "Chile":           (0.00, 0.9,   0.18),
    "Peru":            (0.04, 0.7,   0.11),
    # ── Low-risk: stable democracies ──────────────────────────────────────────
    "United States":   (0.09, 1.0,   0.35),   # Polity=+10; global deployments
    "United Kingdom":  (0.04, 1.0,   0.22),
    "France":          (0.04, 0.9,   0.21),
    "Germany":         (0.00, 1.0,   0.14),
    "Italy":           (0.00, 1.0,   0.15),
    "Spain":           (0.00, 1.0,   0.10),
    "Canada":          (0.00, 1.0,   0.14),
    "Australia":       (0.00, 1.0,   0.19),
    "New Zealand":     (0.00, 1.0,   0.13),
    "Japan":           (0.00, 1.0,   0.10),
    "Sweden":          (0.00, 1.0,   0.11),
    "Norway":          (0.00, 1.0,   0.17),
    "Denmark":         (0.00, 1.0,   0.13),
    "Finland":         (0.00, 1.0,   0.19),
    "Netherlands":     (0.00, 1.0,   0.14),
    "Belgium":         (0.00, 1.0,   0.11),
    "Switzerland":     (0.00, 1.0,   0.08),
    "Austria":         (0.00, 1.0,   0.08),
    "Portugal":        (0.00, 1.0,   0.18),
    "Poland":          (0.00, 1.0,   0.24),
    "Czech Republic":  (0.00, 1.0,   0.15),
    "Hungary":         (0.00, 0.5,   0.16),
    "Romania":         (0.00, 0.9,   0.18),
    "Greece":          (0.00, 1.0,   0.25),
    "Ireland":         (0.00, 1.0,   0.03),
}

# Normalize North Korea's military spending (capped at 1.0)
_COUNTRY_DATA["North Korea"] = (
    _COUNTRY_DATA["North Korea"][0],
    _COUNTRY_DATA["North Korea"][1],
    min(1.0, _COUNTRY_DATA["North Korea"][2]),
)

# World medians — used for unknown countries
_DEFAULT = (0.15, 0.0, 0.15)


def get_country_features(country: str) -> dict[str, float]:
    """
    Return structural features for a country as a flat dict.

    Keys:
      country_conflict_baserate  — 0.0 to 1.0  (UCDP conflict frequency)
      country_polity_norm        — -1.0 to +1.0 (democracy score, normalized)
      country_mil_spending_norm  — 0.0 to 1.0  (military % of GDP / 10)

    Falls back to world medians for unknown countries.
    """
    # Try exact match, then title-case, then strip whitespace
    entry = (
        _COUNTRY_DATA.get(country)
        or _COUNTRY_DATA.get(country.strip())
        or _COUNTRY_DATA.get(country.title())
    )
    if entry is None:
        # Partial match (e.g. "Republic of Ireland" → "Ireland")
        country_lower = country.lower()
        for key, val in _COUNTRY_DATA.items():
            if key.lower() in country_lower or country_lower in key.lower():
                entry = val
                break

    if entry is None:
        entry = _DEFAULT

    return {
        "country_conflict_baserate": entry[0],
        "country_polity_norm": entry[1],
        "country_mil_spending_norm": entry[2],
    }
