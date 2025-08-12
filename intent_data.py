# intent_data.py
# Utilities to normalize multi-source support data (Slack, Email, Shortcut, Customer Calls)
# into a single DataFrame: [record_id, source, timestamp, text, metadata]
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import pandas as pd
import re


# Map common variants to a canonical source name
SOURCE_MAP: Dict[str, str] = {
    "slack": "slack",
    "email": "email",
    "shortcut": "shortcut",           # Shortcut (formerly Clubhouse)
    "clubhouse": "shortcut",
    "customer_call": "customer_call",
    "customer-call": "customer_call",
    "customerconversation": "customer_call",
    "customer_conversation": "customer_call",
    "conversation": "customer_call",
    "call": "customer_call",
}


def _norm_source(raw: Any) -> str:
    s = (str(raw) if raw is not None else "").strip().lower()
    s = s.replace("-", "_").replace(" ", "")
    return SOURCE_MAP.get(s, s or "unknown")


def _join_nonempty(parts: List[str], sep: str = "\n") -> str:
    return sep.join([p for p in parts if p and str(p).strip()])[:10000].strip()


def normalize_records(
    dataset_json: Dict[str, Any],
    *,
    include_skipped_in_attrs: bool = True,
) -> pd.DataFrame:
    """
    Normalize a multi-source JSON payload into a tabular DataFrame.

    Input schema (loosely expected):
      {
        "dataset_name": "...",
        "created_utc": "...",
        "records": [
          { "source": "slack" | "email" | "shortcut" | "customer_call" | ...,
            "payload": {...},
            "ingested_at": "..." }
        ]
      }

    Output columns:
      - record_id: str
      - source: str   (canonical)
      - timestamp: str (best-effort; raw ISO, Slack ts, etc.)
      - text: str     (content to embed/classify)
      - metadata: dict (source-specific fields useful downstream)

    Notes:
      - Any records we can't extract text from are skipped.
      - A list of skipped records (index, source, reason) is stored in df.attrs["skipped"].
    """
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    records = dataset_json.get("records", []) or []
    for i, rec in enumerate(records):
        src_raw = rec.get("source", "unknown")
        src = _norm_source(src_raw)
        payload: Dict[str, Any] = rec.get("payload", {}) or {}
        ts = rec.get("ingested_at") or payload.get("created_at") or payload.get("Date")

        text = ""
        meta: Dict[str, Any] = {}

        if src == "slack":
            ev = payload.get("event", {}) or {}
            # Core text
            text = (ev.get("text") or "").strip()
            # Timestamp
            ts = ts or ev.get("ts")
            # Metadata
            meta = {
                "channel": payload.get("channel_name"),
                "thread_ts": ev.get("thread_ts"),
                "user": (payload.get("user") or {}).get("id") or ev.get("user"),
                "team": payload.get("team"),
            }

        elif src == "email":
            hdr = payload.get("headers", {}) or {}
            subj = hdr.get("Subject", "") or ""
            body = payload.get("body", "") or ""
            text = _join_nonempty([subj, body], sep="\n")
            ts = ts or hdr.get("Date")
            meta = {
                "from": hdr.get("From"),
                "to": hdr.get("To"),
                "message_id": hdr.get("Message-ID"),
            }

        elif src == "shortcut":
            # Support slight schema variants
            name = payload.get("name") or payload.get("title") or ""
            desc = (
                payload.get("description")
                or payload.get("body")
                or payload.get("text")
                or ""
            )
            text = _join_nonempty([name, desc], sep="\n")
            ts = ts or payload.get("created_at") or payload.get("updated_at")
            meta = {
                "id": payload.get("id"),
                "story_type": payload.get("story_type"),
                "labels": [l.get("name") for l in payload.get("labels", []) if isinstance(l, dict)],
                "workflow_state_id": payload.get("workflow_state_id"),
                "project_id": payload.get("project_id"),
                "app_url": payload.get("app_url"),
            }

        elif src == "customer_call":
            # Prefer transcript turns; fall back to summary, then topic
            transcript = payload.get("transcript", [])
            if isinstance(transcript, list) and transcript and isinstance(transcript[0], dict):
                turns = [f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in transcript]
                text = _join_nonempty(turns, sep="\n")
            else:
                text = (payload.get("summary") or payload.get("notes") or "").strip()
            if not text:
                text = (payload.get("topic") or "").strip()

            ts = ts or payload.get("started_at")
            meta = {
                "call_id": payload.get("call_id"),
                "topic": payload.get("topic"),
                "participants": payload.get("participants"),
                "duration_sec": payload.get("duration_sec"),
            }

        else:
            # Generic fallback: try common fields before giving up
            name = payload.get("name") or payload.get("title") or ""
            desc = payload.get("description") or payload.get("body") or payload.get("text") or ""
            text = _join_nonempty([name, desc], sep="\n") or str(payload)[:2000]

        text = (text or "").strip()
        if not text:
            skipped.append({
                "index": i,
                "source": src_raw,
                "reason": "empty_text_after_parse",
            })
            continue

        rows.append({
            "record_id": rec.get("id") or f"{src}-{i}",
            "source": src,
            "timestamp": ts,
            "text": text,
            "metadata": meta,
        })

    df = pd.DataFrame(rows, columns=["record_id", "source", "timestamp", "text", "metadata"])
    if include_skipped_in_attrs:
        df.attrs["skipped"] = skipped
    return df


def debug_summary(dataset_json: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return a small summary dict that you can print or show in Streamlit,
    to understand what was parsed vs. skipped.
    """
    raw_sources = [ (rec.get("source") or "unknown") for rec in dataset_json.get("records", []) ]
    raw_counts = pd.Series(raw_sources).str.lower().replace("-", "_").value_counts().to_dict()
    parsed_counts = df["source"].value_counts().to_dict() if not df.empty else {}

    skipped = getattr(df, "attrs", {}).get("skipped", [])
    return {
        "raw_source_counts": raw_counts,
        "parsed_source_counts": parsed_counts,
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:5],
    }

def load_training_df(df_or_path) -> pd.DataFrame:
    """
    Load the labeled training dataset with columns:
    record_id, source, timestamp, text, L1_Intent
    """
    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()
    else:
        path = str(df_or_path)
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        elif path.lower().endswith(".json"):
            df = pd.read_json(path)
        else:
            raise ValueError("Unsupported training file format. Use CSV or JSON.")

    required = {"record_id", "source", "timestamp", "text", "L1_Intent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training dataset missing columns: {missing}")

    df = df.dropna(subset=["text", "L1_Intent"]).reset_index(drop=True)

    df["record_id"] = df["record_id"].astype(str)
    df["L1_Intent"] = df["L1_Intent"].astype(str).str.strip()

    return df

def load_intent_catalog(csv_path: str) -> pd.DataFrame:
    """
    Load your intents list (Level, Name, Size, Description) and enrich it.
    Returns columns: intent_id, name, level, size, description, kind
    """
    cat = pd.read_csv(csv_path)

    # Drop accidental index column
    if "Unnamed: 0" in cat.columns:
        cat = cat.drop(columns=["Unnamed: 0"])

    cat = cat.rename(columns={
        "Level": "level",
        "Name": "name",
        "Size": "size",
        "Description": "description",
    })

    # Make a stable ID
    def slugify(x):
        return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")
    cat["intent_id"] = cat["name"].apply(slugify)

    # Rough type inference for routing (bug / feature / question / other)
    def infer_kind(row):
        txt = f"{row.get('name','')} {row.get('description','')}".lower()
        if any(k in txt for k in ["feature", "request", "enhancement", "improvement", "roadmap"]):
            return "feature"
        if any(k in txt for k in ["bug", "error", "crash", "exception", "fail", "broken"]):
            return "bug"
        if any(k in txt for k in ["how do i", "how to", "question", "help", "docs", "documentation"]):
            return "question"
        return "other"

    cat["kind"] = cat.apply(infer_kind, axis=1)
    return cat[["intent_id", "name", "level", "size", "description", "kind"]]