# app_clean.py
import os
import json
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from enum import Enum
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Optional

from api_clients import PlainClient, ShortcutClient
from intent_data import normalize_records, debug_summary, load_training_df, load_intent_catalog
from intent_model import load_sbert, embed_texts

# =========================
# App config & constants
# =========================

# Load environment variables
load_dotenv()

# Matching defaults 
DEFAULT_K = int(os.getenv("NN_K", 8))                   # Top-K neighbors
DEFAULT_MIN_SIM = float(os.getenv("NN_MIN_SIM", 0.35))  # Min cosine sim
DEFAULT_POWER = float(os.getenv("NN_POWER", 3.0))       # Neighbor weighting power

FEATURE_HINTS = ["feature", "request", "enhancement", "improvement", "roadmap", "rfe", "fr:"]


def get_matching_params(training_size: int) -> tuple[int, float, float]:
    """Resolve auto-matching params with sensible bounds given training set size."""
    k = max(3, min(DEFAULT_K, training_size))
    return k, DEFAULT_MIN_SIM, DEFAULT_POWER


def _is_ready() -> bool:
    """Minimum ready state to run predictions."""
    return (
        "training_df" in st.session_state
        and "nn" in st.session_state
        and st.session_state.get("training_df") is not None
        and st.session_state.get("nn") is not None
    )


def render_how_to_use():
    with st.expander("📘 How this tool works (overview & setup)", expanded=False):
        st.markdown(
            """
        ### What it does

        - **Classify** support messages into intents using k-nearest neighbors over SBERT embeddings.
        - **No labels? No problem.** Use **🧪 Intent Lab** to discover + name intents from your historical data, review them, and one-click **generate a training set** for classification.
        - **Auto-route:** After prediction, create a **Plain** ticket — and if the intent is a **bug** or **feature**, also create a **Shortcut** story.

        ---

        ### Setup paths
        **A) You do not have labeled data yet (use 🧪 Intent Lab)**
        1. Open **🧪 Intent Lab** and upload raw support messages/emails/conversations.
        2. Run **LLM-only intent discovery** to define high-level intents (data-driven).
        3. Review the “Issues with Assigned Intent” table (this becomes your provisional training set).
        4. Click **Use Intent Lab results as training data** in the sidebar, then **Build / Refresh intent index**.

        **B) You already have labeled training data**
        1. In **Intent Matching Setup**, upload a CSV/JSON or provide a file path.
        2. Click **Build / Refresh intent index** (embeds + builds kNN).

        ---

        ### Running it
        1. In the sidebar (**🔑 API Keys**), add keys if you want to create real tickets.
        2. Go to **Process Message**.
        3. Paste a support message or select a pre-defined example and click **Process Message**.
        4. Results are shown in this order:
        - **Predicted Intent**
        - **Plain Ticket** details
        - **Shortcut Issue** (only for bug/feature)

        ---

        ### Training data format
        - **Required columns:** `text`, `L1_Intent`
        - **Optional:** `record_id`
        - CSV or JSON is supported; JSON should be a list of records.
        """
        )


# =========================
# Core classes / clients
# =========================

class Service(Enum):
    SLACK = "slack"
    PLAIN = "plain"
    SHORTCUT = "shortcut"


class SupportProcessor:
    def __init__(self):
        # Toggle mock behavior per service
        self.mock_slack = True   
        self.mock_plain = False
        self.mock_shortcut = False

        # Example messages for quick testing
        self.mock_messages = [
            "I can't log in to my account via SSO. It says 'invalid credentials'.",
            "The dashboard is loading very slowly today, I think it's a bug.",
            "I'd like to request a new enhancement for the reporting module.",
            "I'm getting a 500 error when submitting the contact form.",
            "How do I reset my password?",
        ]

        # API keys
        self.plain_api_key = os.getenv("PLAIN_API_KEY")
        self.shortcut_api_key = os.getenv("SHORTCUT_API_KEY")

        # Captured mock responses (for Recent Messages tab)
        self.mock_responses: List[Dict] = []

    def create_plain_ticket(self, message: str, intent_kind: str, summary: str) -> Optional[Dict]:
        """Create a ticket in Plain (mocked if no API key)."""
        if self.mock_plain or not self.plain_api_key:
            ticket_id = f"TKT-{random.randint(1000, 9999)}"
            mock_response = {
                "id": ticket_id,
                "title": f"[{intent_kind.upper()}] {summary[:50]}",
                "description": message,
                "status": "open",
                "created_at": datetime.now().isoformat(),
                "mock": True,
                "note": "Mock response" + (" (no Plain API key)" if not self.plain_api_key else ""),
            }
            self.mock_responses.append({
                "type": "plain_ticket",
                "data": mock_response,
                "timestamp": datetime.now().isoformat(),
            })
            return mock_response

        try:
            client = PlainClient(self.plain_api_key)
            customer_email = "customer@example.com"  

            debug_result = client.debug_create_thread(
                title=f"[{intent_kind.upper()}] {summary[:50]}",
                description=message,
                customer_email=customer_email,
                priority="HIGH" if intent_kind == "bug" else "MEDIUM",
            )

            if debug_result.get("success"):
                result = debug_result.get("result", {})
                thread_data = result.get("data", {}).get("createThread", {})
                thread = thread_data.get("thread")
                if thread:
                    self.mock_responses.append({
                        "type": "plain_ticket",
                        "data": thread,
                        "timestamp": datetime.now().isoformat(),
                    })
                    st.success(f"Plain thread created successfully! ID: {thread.get('id', 'unknown')}")
                    return thread
                st.error("No thread data returned from Plain.")
                return None

            st.error(f"Plain debug call failed: {debug_result.get('error')}")
            return None
        except Exception as e:
            st.error(f"Error creating Plain ticket: {str(e)}")
            return None

    def create_shortcut_issue(self, message: str, intent_kind: str, summary: str) -> Optional[Dict]:
        """Create a story in Shortcut."""
        if self.mock_shortcut or not self.shortcut_api_key:
            issue_id = random.randint(1000, 9999)
            mock_response = {
                "id": issue_id,
                "name": f"[{intent_kind.upper()}] {summary}",
                "description": message,
                "story_type": "bug" if intent_kind == "bug" else "feature",
                "app_url": f"https://app.shortcut.com/story/{issue_id}",
                "created_at": datetime.now().isoformat(),
                "mock": True,
                "note": "Mock response" + (" (no Shortcut API key)" if not self.shortcut_api_key else ""),
            }
            self.mock_responses.append({
                "type": "shortcut_issue",
                "data": mock_response,
                "timestamp": datetime.now().isoformat(),
            })
            return mock_response

        try:
            client = ShortcutClient(self.shortcut_api_key)
            story = client.create_story(
                name=f"[{intent_kind.upper()}] {summary}",
                description=message,
                story_type="bug" if intent_kind == "bug" else "feature",
            )
            if story:
                self.mock_responses.append({
                    "type": "shortcut_issue",
                    "data": story,
                    "timestamp": datetime.now().isoformat(),
                })
                st.success(f"Shortcut story created successfully! ID: {story.get('id', 'unknown')}")
            return story
        except Exception as e:
            st.error(f"Error creating Shortcut issue: {str(e)}")
            return None


# =========================
# Cached resources (embeddings & kNN)
# =========================

@st.cache_resource(show_spinner=False)
def _load_model_cached(name: str):
    return load_sbert(name)


@st.cache_data(show_spinner=False)
def _embed_training_cached(model_name: str, texts: List[str]):
    model = _load_model_cached(model_name)
    return embed_texts(model, texts, batch_size=128, show_progress_bar=True)


@st.cache_resource(show_spinner=False)
def _build_nn_index(embeddings: np.ndarray):
    # metric='cosine' -> distances are 1 - cosine_sim
    n_neighbors = max(25, DEFAULT_K)  # ensure enough neighbors for defaults
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(embeddings)
    return nn


def _embed_one(model_name: str, text: str) -> np.ndarray:
    """Embed a single text and return (1,d) normalized row."""
    model = _load_model_cached(model_name)
    vec = embed_texts(model, [text], batch_size=1, show_progress_bar=False)[0]
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec.reshape(1, -1)


def predict_intent_from_neighbors(
    text: str,
    model_name: str,
    training_df: pd.DataFrame,
    nn,
    k: int = 8,
    min_sim: float = 0.35,
    power: float = 3.0,
) -> dict:
    """
    Returns dict:
      predicted_intent: str
      confidence: float (top cosine similarity)
      neighbors: List[dict] with record_id, L1_Intent, similarity, text
      inferred_kind: Optional[str]
    """
    q = _embed_one(model_name, text)
    n_nbrs = min(k, len(training_df))
    dists, idxs = nn.kneighbors(q, n_neighbors=n_nbrs)
    sims = 1.0 - dists[0]
    idxs = idxs[0]

    kept = [(int(i), float(sims[j])) for j, i in enumerate(idxs) if sims[j] >= min_sim]
    neighbors = []

    if not kept:
        inferred = "feature" if any(h in text.lower() for h in FEATURE_HINTS) else "other"
        return {
            "predicted_intent": "Unknown",
            "confidence": 0.0,
            "neighbors": [],
            "inferred_kind": inferred,
        }

    weights = defaultdict(float)
    for i, sim in kept:
        rec = training_df.iloc[i]
        lbl = str(rec["L1_Intent"]).strip()
        w = sim ** power
        weights[lbl] += w
        neighbors.append({
            "record_id": rec["record_id"],
            "L1_Intent": lbl,
            "similarity": round(sim, 4),
            "text": rec["text"],
        })

    predicted = max(weights.items(), key=lambda x: x[1])[0]
    top_sim = max(s for _, s in kept)

    return {
        "predicted_intent": predicted,
        "confidence": float(top_sim),
        "neighbors": neighbors,
        "inferred_kind": None,
    }


# =========================
# Main app
# =========================

def main():
    st.title("CX Intent Detection Tool")
    
    # Show help dropdown
    render_how_to_use()

    # Session state init
    if "processor" not in st.session_state:
        st.session_state.processor = SupportProcessor()

    # ---- Sidebar: API configuration
    st.sidebar.header("API Configuration")
    with st.sidebar.expander("🔑 API Keys", expanded=False):
        st.text_input("OpenAI API Key", type="password", key="openai_key", value=os.getenv("OPENAI_API_KEY", ""))
        st.text_input("Plain API Key", type="password", key="plain_key", value=os.getenv("PLAIN_API_KEY", ""))
        st.text_input("Shortcut API Key", type="password", key="shortcut_key", value=os.getenv("SHORTCUT_API_KEY", ""))
        if st.button("Save API Keys"):
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
            os.environ["PLAIN_API_KEY"] = st.session_state.plain_key
            os.environ["SHORTCUT_API_KEY"] = st.session_state.shortcut_key
            st.success("API Keys saved to environment for this session.")

    # ---- Sidebar: quick test messages
    with st.sidebar.expander("🧪 Test Tools", expanded=True):
        if st.button("Generate Random Test Message"):
            st.session_state.test_message = random.choice(st.session_state.processor.mock_messages)
            st.rerun()
        st.markdown("**Quick Test Messages:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("General Issue", use_container_width=True):
                st.session_state.test_message = "I am getting overcharged on my subscription. Please fix it."
                st.rerun()
        with col2:
            if st.button("Feature Request", use_container_width=True):
                st.session_state.test_message = (
                    "Can you add a new feature to the reporting module? "
                    "I'd like dark mode to be enabled; it's easier on the eyes for night shifts."
                )
                st.rerun()

    # ---- Sidebar: intent matching setup
    st.sidebar.markdown("---")
    st.sidebar.header("Intent Matching Setup")
    model_name_infer = st.sidebar.selectbox(
        "Embedding model",
        ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
        index=0,
        help="Used for both training and query embeddings.",
    )
    train_upload = st.sidebar.file_uploader("Training data (CSV/JSON)", type=["csv", "json"])
    train_path = st.sidebar.text_input("…or path to training data", value="training_data.csv")
    catalog_path = st.sidebar.text_input("Intent catalog CSV", value="/mnt/data/intent_list.csv")
    build_btn = st.sidebar.button("Build / Refresh intent index", use_container_width=True)

    if build_btn:
        try:
            # Catalog
            st.session_state.intent_catalog = load_intent_catalog(catalog_path)

            # Training set
            if train_upload is not None:
                if train_upload.name.lower().endswith(".csv"):
                    train_df_raw = pd.read_csv(train_upload)
                else:
                    train_df_raw = pd.read_json(train_upload)
                st.session_state.training_df = load_training_df(train_df_raw)
            else:
                st.session_state.training_df = load_training_df(train_path)

            with st.spinner("Embedding training set…"):
                tr_texts = st.session_state.training_df["text"].tolist()
                tr_embs = _embed_training_cached(model_name_infer, tr_texts)

            st.session_state.training_embeddings = tr_embs
            st.session_state.nn = _build_nn_index(tr_embs)

            st.success(f"Index ready ✅  Rows: {len(st.session_state.training_df)}   Dim: {tr_embs.shape[1]}")

            with st.sidebar.expander("Preview loaded data", expanded=False):
                st.write("Catalog (first 5):")
                st.dataframe(st.session_state.intent_catalog.head(5), use_container_width=True)
                st.write("Training label distribution:")
                st.write(st.session_state.training_df["L1_Intent"].value_counts().head(15))

        except Exception as e:
            st.sidebar.error(f"Failed to build index: {e}")

    # =========================
    # Tabs
    # =========================
    tab1, tab2, tab3 = st.tabs(["Process Message", "View Recent Messages", "🧪 Intent Lab"])

    # ---- Tab 1: Process Message
    with tab1:
        st.subheader("Process Support Message")
        message = st.text_area("Enter support message:", height=200, value=st.session_state.get("test_message", ""))

        if st.button("Process Message"):
            if not message:
                st.warning("Please enter a message to process.")
            elif not _is_ready():
                st.error("Please load training data and build the intent index from the sidebar first.")
            else:
                # Auto matching params (no UI)
                train_df = st.session_state.training_df
                k, min_sim, power = get_matching_params(len(train_df))

                with st.spinner("Matching intent via nearest neighbors..."):
                    pred = predict_intent_from_neighbors(
                        message,
                        model_name=model_name_infer,
                        training_df=train_df,
                        nn=st.session_state.nn,
                        k=int(k),
                        min_sim=float(min_sim),
                        power=float(power),
                    )

                predicted_intent = pred["predicted_intent"]
                confidence = pred["confidence"]

                # Map predicted intent -> kind via catalog
                kind = "other"
                catalog = st.session_state.get("intent_catalog")
                if (
                    catalog is not None
                    and isinstance(predicted_intent, str)
                    and predicted_intent.lower() != "unknown"
                ):
                    row = catalog.loc[catalog["name"].str.lower() == predicted_intent.lower()]
                    if not row.empty:
                        kind = row.iloc[0]["kind"]

                # Fallback kind for unknown/low-sim
                if predicted_intent == "Unknown" and pred.get("inferred_kind"):
                    kind = pred["inferred_kind"]

                summary = (message[:160] + "…") if len(message) > 160 else message

                # UI: Show prediction
                st.markdown("### Predicted Intent")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Intent", predicted_intent)
                with c2:
                    st.metric("Confidence (top cosine sim)", f"{confidence:.3f}")
                st.caption(f"Auto-matching params → k={k}, min_sim={min_sim:.2f}, power={power:.1f}")

                with st.expander("🔎 Prediction details", expanded=True):
                    st.markdown(f"**Kind (routing):** `{kind}`")
                    if pred["neighbors"]:
                        nbr_df = pd.DataFrame(pred["neighbors"])
                        st.dataframe(
                            nbr_df[["record_id", "L1_Intent", "similarity", "text"]],
                            use_container_width=True,
                        )

                # Create Plain ticket
                ticket = st.session_state.processor.create_plain_ticket(message, kind, summary)
                if ticket:
                    st.markdown("### Plain Ticket")
                    st.json(ticket)

                # Create Shortcut issue if applicable
                issue = None
                if kind in ["feature", "bug"]:
                    issue = st.session_state.processor.create_shortcut_issue(message, kind, summary)
                if issue:
                    st.markdown("### Shortcut Issue")
                    st.json(issue)

                st.success("Message processed!")

    # ---- Tab 2: Recent Messages
    with tab2:
        st.subheader("Recent Support Messages")
        responses = getattr(st.session_state.processor, "mock_responses", [])
        if responses:
            for response in reversed(responses):
                with st.expander(f"{response['type'].replace('_', ' ').title()} - {response['timestamp']}"):
                    data = response["data"]
                    if data.get("mock"):
                        note = f" ({data.get('note', '')})" if data.get("note") else ""
                        st.warning("This is a mock response" + note)
                    st.json(data)
        else:
            st.info("No messages processed yet. Use the 'Process Message' tab to create some.")

    # ---- Tab 3: Intent Lab — LLM-only flow
    with tab3:
        st.subheader("🧪 Intent Lab (LLM-only): Load Conversations → Generate Intent List")

        # 1) Load JSON
        uploaded = st.file_uploader("Upload JSON", type=["json"])
        if uploaded:
            try:
                st.session_state.intent_raw_dataset = json.loads(uploaded.getvalue().decode("utf-8"))
                st.success(f"Loaded {len(st.session_state.intent_raw_dataset.get('records', []))} records.")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

        dataset = st.session_state.get("intent_raw_dataset")

        # 2) Normalize → DataFrame + parsing summary
        if dataset:
            df = normalize_records(dataset)
            st.session_state.intent_df = df
            st.write(f"Normalized rows: {len(df)}")
            st.dataframe(df.head(20), use_container_width=True)

            summary = debug_summary(dataset, df)
            with st.expander("Parsing summary", expanded=False):
                st.json(summary)

        # 3) LLM-only topic discovery (no embeddings/clustering)
        df = st.session_state.get("intent_df")
        if df is not None:
            st.markdown("### 🔎 Generate Intent List with LLM")

            # Auto-select a reasonable cap based on dataset size (read-only caption)
            n = len(df)
            max_intents = int(np.clip(np.sqrt(n), 8, 25)) if n > 0 else 12
            st.caption(f"Auto-selected max L1 intents: {max_intents} (based on {n} records)")

            model_choice = st.selectbox("LLM model", ["gpt-4o-mini", "gpt-4o"], index=0)

            if st.button("Run LLM Discovery on all records"):
                from intent_namer import build_llm_discovery_payload, llm_discover_topics_from_items
                items = build_llm_discovery_payload(
                    df[["record_id", "source", "text"]].to_dict(orient="records"),
                    max_chars=220,
                )
                with st.spinner("Sending compact set to LLM..."):
                    try:
                        res = llm_discover_topics_from_items(
                            items,
                            max_intents=max_intents,
                            allow_l2=True,
                            model=model_choice,
                        )
                        st.session_state.llm_discovery = res
                        st.success("LLM discovery complete.")
                    except Exception as e:
                        st.error(f"LLM discovery failed: {e}")

        # 4) Results (catalog + labeled rows) + downloads
        if st.session_state.get("llm_discovery"):
            res = st.session_state.llm_discovery
            intents = res["intents"]
            assignments = res["assignments"]

            # Catalog table
            cat_rows = []
            for it in intents:
                cat_rows.append({
                    "Level": it.get("level", "L1"),
                    "Name": it.get("name", ""),
                    "Size": len(it.get("includes", [])),
                    "Description": it.get("description", ""),
                })
            catalog_llm = pd.DataFrame(cat_rows).sort_values(["Level", "Size"], ascending=[True, False])
            st.subheader("Intent Catalog (LLM)")
            st.dataframe(catalog_llm, use_container_width=True)

            # Labeled issues table
            df2 = st.session_state.intent_df.copy()
            df2["L1_Intent"] = df2["record_id"].astype(str).map(assignments).fillna("Unknown")
            st.subheader("Issues with Assigned L1 Intent (LLM)")
            st.dataframe(df2[["record_id", "source", "timestamp", "text", "L1_Intent"]], use_container_width=True)

            # Downloads
            st.download_button(
                "Download LLM Intent Catalog (CSV)",
                catalog_llm.to_csv(index=False).encode("utf-8"),
                file_name="intent_catalog_llm.csv",
            )
            st.download_button(
                "Download LLM-Labeled Records (CSV)",
                df2.to_csv(index=False).encode("utf-8"),
                file_name="labeled_issues_llm.csv",
            )


if __name__ == "__main__":
    main()
