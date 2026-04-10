"""
Question parser: converts a free-text binary question into a canonical
structured representation that the rest of the pipeline can act on.

Design principles:
  - Zero LLM dependency. Pure rule-based + regex. Fast, deterministic, offline.
  - Returns a ParsedQuestion even when parsing is uncertain; confidence field
    signals how much to trust the structure.
  - Subject type and event family drive OOD detection and feature routing.
  - Resolution rule is a human-readable string derived heuristically; it is
    shown to the user so they can correct it before trusting the prediction.

Usage:
    from question.parser import parse_question
    pq = parse_question("Will Pedro Sánchez resign before 6 April 2026?")
    print(pq.event_family)   # "political"
    print(pq.predicate)      # "resign"
    print(pq.deadline)       # date(2026, 4, 6)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Literal, Optional

# ── Type aliases ──────────────────────────────────────────────────────────────

SubjectType  = Literal["person", "country", "organization", "artist", "other"]
EventFamily  = Literal["conflict", "political", "legal", "entertainment", "economic", "other"]


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class ParsedQuestion:
    """Canonical representation of a binary forecasting question.

    All fields are best-effort; check `parse_confidence` before trusting them.
    """
    # The original free-text question
    raw: str

    # Who/what is the question about?
    subject: str                      # e.g. "Pedro Sánchez", "Ukraine", "Bad Bunny"
    subject_type: SubjectType         # drives which features are relevant

    # What happens to the subject?
    predicate: str                    # normalized verb, e.g. "resign", "arrested", "perform"
    predicate_raw: str                # original verb phrase from question

    # When must it happen?
    deadline: Optional[date]          # None if not parseable
    deadline_raw: str                 # original deadline string

    # Domain classification
    event_family: EventFamily         # drives OOD detection + feature routing

    # Geographic scope
    jurisdiction: Optional[str]       # e.g. "Spain", "Gaza", None

    # Human-readable resolution rule (heuristic; show to user)
    resolution_rule: str

    # How much to trust this parse (0.0–1.0)
    parse_confidence: float

    # Flags
    is_negated: bool = False          # "Will X NOT happen?"
    has_deadline: bool = True

    # Raw keyword matches for debugging
    matched_predicates: list[str] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)


# ── Keyword tables ────────────────────────────────────────────────────────────
# Each entry: (regex_pattern, normalized_predicate, event_family)
# Patterns matched case-insensitively against the full question.

_PREDICATE_TABLE: list[tuple[str, str, EventFamily]] = [
    # ── Conflict / escalation ─────────────────────────────────────────────────
    (r"\bmilitary (action|operation|strike|offensive|attack|escalat)", "military_escalation", "conflict"),
    (r"\b(escalat|invad|invasion|offensive|siege|bomb|airstrike|drone strike)", "military_escalation", "conflict"),
    (r"\b(war|warfare|combat|fighting|troops|ground invasion)", "military_escalation", "conflict"),
    (r"\b(coup|overthrow|junta)\b", "coup", "conflict"),
    (r"\bceasefire\b", "ceasefire", "conflict"),
    (r"\bnuclear (test|weapon|strike|detonation)\b", "nuclear_event", "conflict"),
    (r"\b(missile|rocket) (launch|test|strike)\b", "military_escalation", "conflict"),

    # ── Political ────────────────────────────────────────────────────────────
    (r"\b(resign\w*|step(s)? down|quit(s)? office|leave(s)? office)\b", "resign", "political"),
    (r"\b(election|referendum|vote|ballot|poll)\b", "election_event", "political"),
    (r"\b(impeach\w*|removal from office|ousted)\b", "impeach", "political"),
    (r"\b(appoint\w*|nominat\w*|confirmed as|sworn in)\b", "appointment", "political"),
    (r"\b(sanction\w*|embargo\w*)\b", "sanction", "political"),
    (r"\b(legislation|law|bill|act)\b.{0,20}\b(pass|sign|veto|approve|reject)\b", "legislation", "political"),
    (r"\b(pass|sign|veto|approve|reject)\b.{0,20}\b(legislation|law|bill|act)\b", "legislation", "political"),
    (r"\b(summit|meeting|talks|negotiat\w*|diplomacy|diplomatic)\b", "diplomatic_event", "political"),
    (r"\b(coalition|parliament|congress|senate)\b.{0,30}\b(collapse|fall|dissolv\w*)\b", "government_collapse", "political"),
    (r"\b(dissolv\w*|snap election|early election)\b", "government_collapse", "political"),

    # ── Legal / judicial ─────────────────────────────────────────────────────
    # Use stem + \w* to match inflected forms: arrest/arrested, capture/captured, etc.
    (r"\b(arrest\w*|detain\w*|taken into custody|apprehend\w*)\b", "arrested", "legal"),
    (r"\b(captur\w*|extrad\w*|handover|handed over)\b", "captured", "legal"),
    (r"\b(indict\w*|charged with|prosecut\w*)\b", "indicted", "legal"),
    (r"\b(convict\w*|sentenced|guilty verdict|found guilty)\b", "convicted", "legal"),
    (r"\b(released|freed|acquit\w*)\b", "released", "legal"),
    (r"\b(trial|hearing|verdict|ruling)\b", "trial_event", "legal"),
    (r"\b(asset freeze|asset seizure)\b", "legal_sanction", "legal"),

    # ── Entertainment / cultural ──────────────────────────────────────────────
    (r"\b(tour|concert|perform|show|gig|festival)\b", "perform", "entertainment"),
    (r"\b(visit|appear in|come to|travel to)\b.{0,30}\b(spain|europe|france|uk|germany|us|usa)\b", "visit", "entertainment"),
    (r"\b(album|single|release|drop)\b.{0,20}\b(music|song|track)\b", "music_release", "entertainment"),
    (r"\b(film|movie|series|season)\b.{0,20}\b(release|premiere|debut)\b", "media_release", "entertainment"),
    (r"\b(award|oscar|grammy|golden globe|prize)\b", "award_event", "entertainment"),
    (r"\b(retire|retirement|farewell tour)\b", "retire", "entertainment"),

    # ── Economic / financial ──────────────────────────────────────────────────
    (r"\b(rate cut|rate hike|interest rate|fed|ecb|central bank)\b", "interest_rate", "economic"),
    (r"\b(recession|gdp|growth rate|economic contraction)\b", "economic_indicator", "economic"),
    (r"\b(bankrupt|default|insolvency|debt crisis)\b", "default", "economic"),
    (r"\b(ipo|merger|acquisition|takeover)\b", "corporate_event", "economic"),
    (r"\b(oil price|gas price|energy crisis|opec)\b", "energy_market", "economic"),
    (r"\b(inflation|deflation|cpi|pce)\b", "economic_indicator", "economic"),
]

# ── Subject-type hints ────────────────────────────────────────────────────────
# Regex patterns that suggest a specific subject type.

_SUBJECT_TYPE_HINTS: list[tuple[str, SubjectType]] = [
    # Countries and regions (incomplete; just common ones)
    (r"\b(ukraine|russia|china|usa|us|iran|israel|gaza|syria|north korea|taiwan|"
     r"france|germany|spain|uk|italy|turkey|india|pakistan|afghanistan|yemen|"
     r"iraq|libya|somalia|ethiopia|nigeria|myanmar|venezuela|cuba|sudan)\b",
     "country"),
    # Political roles that imply a person
    (r"\b(president|prime minister|chancellor|minister|senator|governor|"
     r"secretary|congressman|mp|pm|ceo|cfo)\b", "person"),
    # Entertainment roles
    (r"\b(singer|rapper|band|musician|artist|actor|director|producer|athlete)\b",
     "artist"),
    # Organizations
    (r"\b(nato|un|eu|imf|who|opec|fed|ecb|congress|parliament|supreme court|"
     r"pentagon|kremlin|whitehouse)\b", "organization"),
]

# ── Known artists (subset; triggers artist subject type) ─────────────────────
_KNOWN_ARTISTS = {
    "bad bunny", "taylor swift", "beyoncé", "beyonce", "drake", "rihanna",
    "adele", "coldplay", "madonna", "bruce springsteen", "metallica",
    "the weeknd", "harry styles", "billie eilish", "j balvin", "maluma",
    "rosalía", "rosalia", "karol g", "ozuna", "rauw alejandro", "anuel aa",
    "shakira", "enrique iglesias", "alejandro sanz",
}

# ── Known politicians (subset) ────────────────────────────────────────────────
_KNOWN_POLITICIANS = {
    "pedro sánchez", "pedro sanchez", "donald trump", "joe biden",
    "kamala harris", "vladimir putin", "xi jinping", "emmanuel macron",
    "olaf scholz", "giorgia meloni", "rishi sunak", "keir starmer",
    "benjamin netanyahu", "volodymyr zelensky", "nicolás maduro",
    "nicolas maduro", "narendra modi", "javier milei", "lula", "lula da silva",
    "ursula von der leyen", "kim jong un", "kim jong-un",
    "recep tayyip erdogan", "erdogan",
}

# ── Deadline parsing ──────────────────────────────────────────────────────────

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
    "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}

def _parse_deadline(text: str) -> tuple[Optional[date], str]:
    """Extract deadline from question text. Returns (date_or_None, raw_string)."""
    # Pattern: before/by/until + date
    # "before 6 April 2026", "by April 6 2026", "before April 6th", "in 2026"
    patterns = [
        # "before/by/until 6 April 2026" or "before/by/until April 6, 2026"
        r"\b(?:before|by|until|prior to|no later than)\s+"
        r"(\d{1,2})(?:st|nd|rd|th)?\s+(\w+)\s+(\d{4})",
        r"\b(?:before|by|until|prior to|no later than)\s+"
        r"(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})",
        # "before 6 April" (no year) → infer current/next year
        r"\b(?:before|by|until|prior to)\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\w+)(?:\s+(\d{4}))?",
        r"\b(?:before|by|until|prior to)\s+(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{4}))?",
        # "in 2026" or "in Q2 2026"
        r"\bin\s+(20\d{2})\b",
        # "this year"
        r"\bthis year\b",
        # ISO date
        r"\b(20\d{2})[-/](\d{2})[-/](\d{2})\b",
    ]

    today = date.today()

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if not m:
            continue
        raw = m.group(0)
        groups = [g for g in m.groups() if g is not None]

        # ISO date
        if re.match(r"20\d{2}[-/]\d{2}[-/]\d{2}", raw):
            try:
                return date.fromisoformat(raw.replace("/", "-")), raw
            except ValueError:
                continue

        # "this year"
        if "this year" in raw.lower():
            return date(today.year, 12, 31), "this year"

        # "in YYYY"
        if re.match(r"in\s+20\d{2}", raw, re.IGNORECASE):
            year = int(groups[0])
            return date(year, 12, 31), raw

        # Try to extract day, month, year from groups
        day = month = year = None
        for g in groups:
            if g.isdigit():
                val = int(g)
                if val > 31:
                    year = val
                elif val > 12:
                    # Too big to be a month number, must be a day
                    if day is None:
                        day = val
                elif month is None:
                    # Could be month number (1-12) if not already named — skip;
                    # we prefer named months for clarity. Treat as day if month set.
                    if day is None:
                        day = val   # ambiguous: treat as day
                else:
                    # month already set by name → this digit is the day
                    if day is None:
                        day = val
            elif g.lower() in _MONTHS:
                month = _MONTHS[g.lower()]

        if month is None:
            continue
        if year is None:
            year = today.year
            if date(year, month, day or 28) < today:
                year += 1
        if day is None:
            # last day of month
            import calendar
            day = calendar.monthrange(year, month)[1]

        try:
            return date(year, month, day), raw
        except ValueError:
            continue

    return None, ""


# ── Subject extraction ────────────────────────────────────────────────────────

def _extract_subject(text: str) -> tuple[str, SubjectType, float]:
    """
    Returns (subject_string, subject_type, confidence).

    Strategy:
    1. Known artists → artist
    2. Known politicians → person
    3. Regex type hints
    4. Capitalize first proper noun after "Will"
    """
    lower = text.lower()

    # Known artist exact match
    for artist in _KNOWN_ARTISTS:
        if artist in lower:
            # Find canonical capitalization from original text
            pattern = re.compile(re.escape(artist), re.IGNORECASE)
            m = pattern.search(text)
            return (m.group(0) if m else artist.title()), "artist", 0.95

    # Known politician exact match
    for pol in _KNOWN_POLITICIANS:
        if pol in lower:
            pattern = re.compile(re.escape(pol), re.IGNORECASE)
            m = pattern.search(text)
            return (m.group(0) if m else pol.title()), "person", 0.92

    # Type hint regex
    for pat, stype in _SUBJECT_TYPE_HINTS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(0).strip(), stype, 0.70

    # Fallback: first capitalized phrase after "Will" / "¿"
    m = re.search(r"\bwill\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})", text)
    if m:
        return m.group(1).strip(), "other", 0.40

    return "unknown", "other", 0.20


# ── Jurisdiction extraction ───────────────────────────────────────────────────

_COUNTRY_NAMES = {
    "afghanistan", "albania", "algeria", "angola", "argentina", "armenia",
    "australia", "austria", "azerbaijan", "bahrain", "bangladesh", "belarus",
    "belgium", "bolivia", "brazil", "bulgaria", "burkina faso", "cameroon",
    "canada", "chad", "chile", "china", "colombia", "congo", "croatia", "cuba",
    "czech republic", "denmark", "egypt", "ethiopia", "finland", "france",
    "gaza", "georgia", "germany", "ghana", "greece", "guatemala", "haiti",
    "hungary", "india", "indonesia", "iran", "iraq", "ireland", "israel",
    "italy", "japan", "jordan", "kazakhstan", "kenya", "kosovo", "kuwait",
    "latvia", "lebanon", "libya", "lithuania", "malaysia", "mali", "mexico",
    "moldova", "mongolia", "morocco", "mozambique", "myanmar", "nepal",
    "netherlands", "new zealand", "nicaragua", "nigeria", "north korea",
    "norway", "pakistan", "palestine", "panama", "peru", "philippines",
    "poland", "portugal", "qatar", "romania", "russia", "rwanda",
    "saudi arabia", "senegal", "serbia", "sierra leone", "somalia",
    "south africa", "south korea", "south sudan", "spain", "sri lanka",
    "sudan", "sweden", "switzerland", "syria", "taiwan", "tajikistan",
    "tanzania", "thailand", "tunisia", "turkey", "ukraine", "united arab emirates",
    "united kingdom", "united states", "usa", "us", "uk", "uruguay",
    "uzbekistan", "venezuela", "vietnam", "yemen", "zambia", "zimbabwe",
}

def _extract_jurisdiction(text: str) -> Optional[str]:
    """Find first mentioned country/region in the question."""
    lower = text.lower()
    # Longest match first to avoid "us" matching inside "russia"
    for country in sorted(_COUNTRY_NAMES, key=len, reverse=True):
        pattern = r"\b" + re.escape(country) + r"\b"
        if re.search(pattern, lower):
            return country.title()
    return None


# ── Predicate extraction ──────────────────────────────────────────────────────

def _extract_predicate(text: str) -> tuple[str, str, EventFamily, list[str]]:
    """
    Returns (normalized_predicate, raw_phrase, event_family, matched_patterns).
    Falls back to ("unknown", raw_verb, "other", []) if no match.
    """
    matches: list[tuple[str, str, EventFamily]] = []

    for pat, pred, family in _PREDICATE_TABLE:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            matches.append((pred, m.group(0), family))

    if not matches:
        # Last resort: extract main verb after subject
        m = re.search(r"\bwill\s+\S+\s+(\w+)", text, re.IGNORECASE)
        raw = m.group(1) if m else "unknown"
        return "unknown", raw, "other", []

    # If multiple matches, prefer the most specific (longer pattern match)
    matches.sort(key=lambda x: len(x[1]), reverse=True)
    pred, raw, family = matches[0]
    matched = [m[1] for m in matches]
    return pred, raw, family, matched


# ── Negation detection ────────────────────────────────────────────────────────

def _is_negated(text: str) -> bool:
    return bool(re.search(r"\bnot\b|\bnever\b|\bno longer\b|\bfail(s)? to\b", text, re.IGNORECASE))


# ── Resolution rule generator ─────────────────────────────────────────────────

def _resolution_rule(predicate: str, subject: str, deadline: Optional[date]) -> str:
    deadline_str = f"before {deadline.isoformat()}" if deadline else "by the deadline"
    rules = {
        "resign": f"Formal resignation or official removal of {subject} from office {deadline_str}.",
        "arrested": f"Official arrest or detention of {subject} confirmed by credible authority {deadline_str}.",
        "captured": f"Physical capture or confirmed detention of {subject} by state actor {deadline_str}.",
        "indicted": f"Formal criminal indictment or charges filed against {subject} {deadline_str}.",
        "convicted": f"Criminal conviction verdict for {subject} {deadline_str}.",
        "perform": f"Public performance or concert by {subject} in the specified location {deadline_str}.",
        "visit": f"Verified public appearance or official visit by {subject} {deadline_str}.",
        "military_escalation": f"Armed conflict or military escalation event involving {subject} {deadline_str}.",
        "coup": f"Successful or attempted coup against government of {subject} {deadline_str}.",
        "ceasefire": f"Formal ceasefire agreement signed and publicly announced involving {subject} {deadline_str}.",
        "election_event": f"Official election or referendum held as scheduled involving {subject} {deadline_str}.",
        "impeach": f"Formal impeachment vote or removal procedure initiated against {subject} {deadline_str}.",
        "interest_rate": f"Official central bank rate decision announced {deadline_str}.",
        "default": f"Formal default declaration or missed debt payment by {subject} {deadline_str}.",
        "sanction": f"Official sanction or embargo imposed on {subject} {deadline_str}.",
        "nuclear_event": f"Confirmed nuclear test or nuclear weapon use by {subject} {deadline_str}.",
        "music_release": f"Official public release of music by {subject} {deadline_str}.",
        "government_collapse": f"Government collapse, dissolution of parliament, or snap election triggered in {subject} {deadline_str}.",
        "diplomatic_event": f"Official diplomatic summit or meeting involving {subject} confirmed {deadline_str}.",
    }
    return rules.get(predicate, f"Event '{predicate}' occurs involving {subject} {deadline_str}.")


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_question(question: str) -> ParsedQuestion:
    """
    Parse a free-text binary question into a canonical ParsedQuestion.

    This is entirely rule-based — no LLM, no network calls.
    parse_confidence indicates how reliable the parse is.
    """
    text = question.strip()

    # Remove leading "Will" variations and question marks for cleaner matching
    deadline, deadline_raw = _parse_deadline(text)
    subject, subject_type, subj_conf = _extract_subject(text)
    predicate, predicate_raw, event_family, matched = _extract_predicate(text)
    jurisdiction = _extract_jurisdiction(text)
    negated = _is_negated(text)
    resolution = _resolution_rule(predicate, subject, deadline)

    # Compute parse confidence
    conf = subj_conf
    if predicate != "unknown":
        conf = min(1.0, conf + 0.2)
    if deadline is not None:
        conf = min(1.0, conf + 0.1)
    if event_family != "other":
        conf = min(1.0, conf + 0.1)

    return ParsedQuestion(
        raw=text,
        subject=subject,
        subject_type=subject_type,
        predicate=predicate,
        predicate_raw=predicate_raw,
        deadline=deadline,
        deadline_raw=deadline_raw,
        event_family=event_family,
        jurisdiction=jurisdiction,
        resolution_rule=resolution,
        parse_confidence=round(conf, 2),
        is_negated=negated,
        has_deadline=deadline is not None,
        matched_predicates=matched,
        matched_entities=[subject] if subject != "unknown" else [],
    )
