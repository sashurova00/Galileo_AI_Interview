"""
Naming and labeling for identified intents and topics.
"""

# intent_namer.py
from __future__ import annotations
from typing import List, Dict, Any
import os
import json
import re

def _fallback_name(keywords: List[str]) -> str:
    return " / ".join(keywords[:3]).title() if keywords else "General"

def name_intents_with_llm(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Turn cluster summaries into intent names/descriptions for customer support.
    Uses OpenAI if OPENAI_API_KEY is present; otherwise falls back to keyword-based names.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # fallback: naive names
        named = []
        for c in clusters:
            named.append({
                "level": c.get("level","L1"),
                "intent_id": f"{c.get('level','L1')}-{c['topic_id']}",
                "name": _fallback_name(c.get("keywords", [])),
                "description": f"Auto-named from keywords: {', '.join(c.get('keywords', []))}",
                "size": c.get("size", 0),
                "keywords": c.get("keywords", []),
                "includes": [c.get("topic_id")]
            })
        return named

    # OpenAI path (compact JSON schema prompt)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = {
            "role": "user",
            "content": (
                "You are designing a customer-support intent taxonomy for an enterprise LLM-evaluation startup so the issues will be specific to this industry."
                "Return compact JSON with fields: name, description, includes (topic_ids), level, size, keywords. "
                "Keep <= 20 total intents; merge near-duplicates.\n\n"
                f"Clusters:\n{json.dumps(clusters, ensure_ascii=False)}"
            )
        }
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You return only JSON."}, prompt],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        return data.get("intents", [])
    except Exception:
        # fallback if API hiccups
        return [
            {
                "level": c.get("level","L1"),
                "intent_id": f"{c.get('level','L1')}-{c['topic_id']}",
                "name": _fallback_name(c.get("keywords", [])),
                "description": f"Auto-named from keywords: {', '.join(c.get('keywords', []))}",
                "size": c.get("size", 0),
                "keywords": c.get("keywords", []),
                "includes": [c.get("topic_id")]
            } for c in clusters
        ]

def _compress_text(s: str, max_chars: int = 220) -> str:
    s = " ".join((s or "").split())
    return s[:max_chars]

def build_llm_discovery_payload(records: List[Dict[str, Any]], max_chars: int = 220) -> List[Dict[str, str]]:
    """
    records: [{ "record_id": "...", "source": "...", "text": "..." }, ...]
    returns compact items for prompt: [{id, source, text}]
    """
    items = []
    for r in records:
        items.append({
            "id": str(r["record_id"]),
            "source": str(r.get("source", "")),
            "text": _compress_text(str(r.get("text", "")), max_chars=max_chars)
        })
    return items

def _extract_json_block(s: str) -> str:
    # try to pull the largest {...} block
    m1 = re.search(r"\{.*\}", s, re.DOTALL)
    if m1:
        return m1.group(0)
    return s

def llm_discover_topics_from_items(items: List[Dict[str, str]], max_intents: int = 12, allow_l2: bool = True, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    LLM-only topic discovery focused on natural business patterns for Galileo.AI support.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot run LLM discovery.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    context = """
You are analyzing customer support queries for Galileo.AI, an enterprise AI evaluation platform.

Analyze the actual customer issues to discover natural, recurring patterns. Group similar problems together based on:
- What customers are actually asking about
- What type of help or resolution they need
- What business function this relates to

Create intent categories that emerge from the data itself. Focus on business-operational groupings that would help a support team route and resolve issues efficiently.

Some examples of intents:
- Authentication & access issues
- Billing & subscription management  
- Platform technical issues (outages, performance)
- Integration problems (APIs, third-party tools)
- Product functionality questions (RAG evaluation, LLM metrics)
- Data management (uploads, datasets, formats)
- Sales inquiries (quotes, pricing, discounts)
- Account management (users, permissions, settings)

Goal: Discover the natural support patterns in this specific dataset, not impose predetermined categories.
"""

    schema = f"""
Return ONLY this JSON structure:

{{
  "intents": [
    {{
      "name": "Clear Intent Name",
      "description": "What this covers and why customers contact support",
      "business_impact": "high|medium|low",
      "volume_expectation": "high|medium|low",
      "includes": ["record_id1", "record_id2", ...]
    }}
  ],
  "patterns": "Key insights about recurring customer issues and business needs"
}}

Requirements:
- Maximum {max_intents} intents total
- Every record must belong to exactly one intent
- Focus on business operations, not technical implementation details
- Names should be clear to anyone in the company
- Cover the breadth of customer needs without being overly specific
"""

    prompt = {
        "role": "user", 
        "content": (
            context + "\n\n" + schema + "\n\n" +
            "Customer support records to analyze:\n" +
            json.dumps(items, ensure_ascii=False)
        )
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Identify natural business patterns in customer support data. Return only JSON."},
            prompt
        ],
        temperature=0.2,
    )
    
    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(_extract_json_block(raw))
    except Exception:
        # Retry with JSON fix
        fix = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Fix the JSON formatting only."},
                {"role": "user", "content": f"Make this valid JSON only, no commentary:\n{raw}"}
            ],
            temperature=0.0,
        )
        fixed = fix.choices[0].message.content.strip()
        data = json.loads(_extract_json_block(fixed))

    # Build assignments map
    intents = data.get("intents", [])
    assignments = {}
    for intent in intents:
        name = intent.get("name", "")
        for rid in intent.get("includes", []):
            assignments[str(rid)] = name

    return {
        "intents": intents,
        "assignments": assignments,
        "raw": data
    }