"""
Relation extractor — extracts temporal relations from headlines using Ollama.

Runs non-fatally during daily world state update and per-question prediction.
Falls back to rule-based patterns when Ollama is unavailable.

Output schema (list of dicts):
  {
    subject:       str,    # canonical name
    subject_type:  str,    # 'person' | 'country' | 'organization'
    relation_type: str,    # 'holds_office' | 'conflicts_with' | 'under_investigation' | ...
    object:        str,
    object_type:   str,
    confidence:    float,
    action:        str,    # 'assert' | 'retract'
    valid_from:    str,    # ISO date, best-effort from context
    attributes:    dict,
  }
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_VALID_RELATION_TYPES = {
    "holds_office", "governs", "member_of", "allied_with",
    "under_investigation", "sanctioned_by", "conflicts_with",
    "negotiates_with", "disputes_with", "at_war_with",
    "indicted_by", "arrested_by", "resigned_from",
}

_VALID_ACTIONS = {"assert", "retract"}

_EXTRACTION_PROMPT = """You are a geopolitical knowledge graph extractor.
Given the headline below, extract any TEMPORAL RELATIONS mentioned.

Return ONLY valid JSON with this schema:
{{"relations": [
  {{
    "subject": "canonical entity name",
    "subject_type": "person|country|organization",
    "relation_type": "one of: holds_office|governs|member_of|allied_with|under_investigation|sanctioned_by|conflicts_with|negotiates_with|disputes_with|at_war_with|indicted_by|arrested_by|resigned_from",
    "object": "canonical entity name",
    "object_type": "person|country|organization",
    "confidence": 0.0-1.0,
    "action": "assert or retract",
    "attributes": {{}}
  }}
]}}

Rules:
- Only extract relations explicitly stated or very strongly implied.
- Use canonical names (e.g. "United States" not "the US", "Vladimir Putin" not "Putin").
- action="retract" only if the headline says someone RESIGNED, was REMOVED, LOST OFFICE, etc.
- Return {{"relations": []}} if no relations are mentioned.
- Return valid JSON only, no commentary.

Headline: {headline}
Entity context: {entity}
"""


def _llm_extract(headline: str, entity: str) -> list[dict]:
    """Try Ollama extraction. Returns [] on any failure."""
    try:
        from llm.client import OllamaClient
        client = OllamaClient()
        if not client.is_available():
            return []

        prompt = _EXTRACTION_PROMPT.format(
            headline=headline[:300],
            entity=entity,
        )
        raw = client.generate(prompt, max_tokens=400, temperature=0.0)
        if not raw:
            return []

        # Extract JSON from response (model may add prose)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return []

        data = json.loads(match.group(0))
        return data.get("relations", [])

    except Exception as e:
        logger.debug("llm relation extraction failed: %s", e)
        return []


# Rule-based patterns as fallback / augmentation
_OFFICE_PATTERNS = [
    (r"(?P<person>[A-Z][a-z]+ [A-Z][a-z]+)\s+(?:elected|appointed|sworn in|inaugurated|becomes?|named)\s+(?:as\s+)?(?:president|prime minister|chancellor|premier|pm|leader)\s+(?:of\s+)?(?P<country>[A-Z][a-zA-Z\s]+)", "holds_office", "assert"),
    (r"(?P<person>[A-Z][a-z]+ [A-Z][a-z]+)\s+(?:resigns?|steps? down|quits?|ousted|removed)\s+(?:as|from)?\s+(?:president|prime minister|chancellor)", "holds_office", "retract"),
    (r"(?P<person>[A-Z][a-z]+ [A-Z][a-z]+)\s+(?:indicted|charged|arrested|investigated)", "under_investigation", "assert"),
]

_CONFLICT_PATTERNS = [
    (r"(?P<a>[A-Z][a-zA-Z\s]+)\s+(?:attacks?|invades?|strikes?|bombs?|launches? (?:offensive|attack|airstrikes?))\s+(?P<b>[A-Z][a-zA-Z\s]+)", "conflicts_with", "assert"),
    (r"(?P<a>[A-Z][a-zA-Z\s]+)\s+(?:and|with)\s+(?P<b>[A-Z][a-zA-Z\s]+)\s+(?:ceasefire|peace deal|truce)", "conflicts_with", "retract"),
]


def _rule_extract(headline: str, entity: str) -> list[dict]:
    """Fast rule-based relation extraction."""
    results = []
    now_str = datetime.now(timezone.utc).date().isoformat()

    for pattern, rel_type, action in _OFFICE_PATTERNS:
        m = re.search(pattern, headline, re.IGNORECASE)
        if m:
            gd = m.groupdict()
            subj = gd.get("person", entity)
            obj  = gd.get("country", "")
            if subj and obj:
                results.append({
                    "subject":      subj.strip(),
                    "subject_type": "person",
                    "relation_type": rel_type,
                    "object":       obj.strip(),
                    "object_type":  "country",
                    "confidence":   0.65,
                    "action":       action,
                    "valid_from":   now_str,
                    "attributes":   {},
                })

    for pattern, rel_type, action in _CONFLICT_PATTERNS:
        m = re.search(pattern, headline, re.IGNORECASE)
        if m:
            gd = m.groupdict()
            a = gd.get("a", "").strip()
            b = gd.get("b", "").strip()
            if a and b and len(a) > 3 and len(b) > 3:
                results.append({
                    "subject":      a,
                    "subject_type": "country",
                    "relation_type": rel_type,
                    "object":       b,
                    "object_type":  "country",
                    "confidence":   0.60,
                    "action":       action,
                    "valid_from":   now_str,
                    "attributes":   {},
                })

    return results


def _validate(rel: dict) -> bool:
    """Basic validation of extracted relation dict."""
    if not rel.get("subject") or not rel.get("object"):
        return False
    if rel.get("relation_type") not in _VALID_RELATION_TYPES:
        return False
    if rel.get("action", "assert") not in _VALID_ACTIONS:
        return False
    if not (0.0 <= float(rel.get("confidence", 0)) <= 1.0):
        return False
    return True


def extract_relations(
    headline: str,
    entity: str,
    use_llm: bool = True,
    min_confidence: float = 0.5,
) -> list[dict]:
    """
    Extract relations from a single headline.
    Tries LLM first (if available), then rule-based.
    Returns validated, deduplicated list.
    """
    relations: list[dict] = []

    if use_llm:
        llm_results = _llm_extract(headline, entity)
        relations.extend(llm_results)

    rule_results = _rule_extract(headline, entity)

    # Merge: only add rule results not already covered by LLM
    existing_keys = {(r.get("subject",""), r.get("relation_type",""), r.get("object","")) for r in relations}
    for r in rule_results:
        key = (r.get("subject",""), r.get("relation_type",""), r.get("object",""))
        if key not in existing_keys:
            relations.append(r)
            existing_keys.add(key)

    # Validate and filter
    valid = [r for r in relations if _validate(r) and float(r.get("confidence", 0)) >= min_confidence]
    return valid


def extract_and_write(
    headlines: list[str],
    entity: str,
    use_llm: bool = True,
    min_confidence: float = 0.55,
) -> int:
    """
    Extract relations from a list of headlines and write them to the DB.
    Returns number of relations written.
    """
    from data_layer.relation_writer import write_relation

    now = datetime.now(timezone.utc)
    n_written = 0

    for headline in headlines[:20]:   # cap at 20 to avoid excessive LLM calls
        if not headline or len(headline.strip()) < 10:
            continue

        rels = extract_relations(headline, entity, use_llm=use_llm, min_confidence=min_confidence)
        for rel in rels:
            try:
                vf_str = rel.get("valid_from")
                valid_from = (
                    datetime.fromisoformat(vf_str).replace(tzinfo=timezone.utc)
                    if vf_str else now
                )
                action = rel.get("action", "assert")

                if action == "retract":
                    from data_layer.relation_writer import retract_relation
                    retract_relation(rel["subject"], rel["object"], rel["relation_type"], now)
                else:
                    rid = write_relation(
                        subject_name=rel["subject"],
                        subject_type=rel.get("subject_type", "person"),
                        object_name=rel["object"],
                        object_type=rel.get("object_type", "country"),
                        relation_type=rel["relation_type"],
                        valid_from=valid_from,
                        confidence=float(rel.get("confidence", 0.6)),
                        attributes=rel.get("attributes", {}),
                        extractor_version="rule_v1" if float(rel.get("confidence",0)) < 0.7 else "llm_v1",
                    )
                    if rid:
                        n_written += 1

            except Exception as e:
                logger.debug("extract_and_write: failed for %s: %s", rel, e)

    return n_written
