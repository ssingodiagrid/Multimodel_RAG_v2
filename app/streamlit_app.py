import streamlit as st
from dotenv import load_dotenv
import sys
import uuid
from pathlib import Path
import pandas as pd

# Add project root to sys.path to allow imports from src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

load_dotenv()

from configs.settings import settings
from src.retrieval.vector_store import VectorStoreManager, get_qdrant_client
from src.retrieval.bm25_store import BM25Manager
from src.retrieval.reranker import get_reranker
from src.generation.rag_chain import RAGPipeline
from src.retrieval.semantic_cache import SemanticCacheManager

# ── Cached resource loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FAISS index...")
def _load_faiss():
    return VectorStoreManager().load_faiss_store()

@st.cache_resource(show_spinner="Connecting to Qdrant...")
def _get_cached_qdrant_client():
    return get_qdrant_client()

@st.cache_resource(show_spinner="Loading Qdrant index...")
def _load_qdrant():
    client = _get_cached_qdrant_client()
    return VectorStoreManager().load_qdrant_store(client=client)

@st.cache_resource(show_spinner="Loading BM25 index...")
def _load_bm25():
    return BM25Manager().load_bm25_store()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="IFC Annual Report 2024 QA", page_icon="📊", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_store" not in st.session_state:
    st.session_state.active_store = None
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "cache_namespace" not in st.session_state:
    st.session_state.cache_namespace = uuid.uuid4().hex

SemanticCacheManager.cleanup_old_namespaces(
    settings.indices_dir / "semantic_cache",
    active_namespace=st.session_state.cache_namespace,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Retrieval Settings")
    st.markdown("---")

    qdrant_badge = "🐳 Docker Server" if settings.qdrant_mode == "server" else "💾 Local File"
    st.info(f"Qdrant mode: **{qdrant_badge}**")
    st.markdown("---")

    st.subheader("🗄️ Vector Store")
    store_choice = st.radio(
        label="Select retrieval backend:",
        options=["FAISS", "Qdrant", "BM25 (Sparse)", "Hybrid (Qdrant + BM25)"],
        index=3,
        help=(
            "**FAISS** — Fast in-memory flat index.\n\n"
            "**Qdrant** — HNSW dense vector index.\n\n"
            "**BM25** — Sparse keyword-based index.\n\n"
            "**Hybrid** — Merges Qdrant and BM25 using EnsembleRetriever."
        ),
    )
    st.markdown("---")

    st.subheader("🔍 Advanced Retrieval")
    use_reranker = st.toggle("Enable Cross-Encoder Re-ranking", value=False, help="Reranks top-N results to improve relevance. Slower but more accurate.")
    use_multihop = st.toggle("Enable Multi-hop Retrieval", value=False, help="Decomposes complex queries into multiple sub-queries. Excellent for multi-part questions.")
    
    st.markdown("---")
    st.subheader("🏷️ Metadata Filters (Qdrant Only)")
    
    filter_page_range = st.slider("Page Range:", min_value=1, max_value=200, value=(1, 200))
    filter_content = st.multiselect("Content Type:", ["text", "table", "image"], default=["text", "table", "image"])
    
    st.markdown("---")

    st.subheader("🔢 Top-K Chunks")
    top_k = st.slider("Number of chunks to retrieve:", min_value=1, max_value=15, value=5, step=1)
    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.cache_namespace = uuid.uuid4().hex
        st.session_state.pipeline = None
        st.session_state.active_store = None
        st.rerun()

    st.markdown("---")
    
    st.subheader("🧭 Navigation")
    page = st.radio("Go to:", ["💬 Chat", "📊 Evaluation"])

    st.markdown("---")
    st.caption("Phase 3 · Advanced RAG · IFC 2024")

# ── Build retriever when store or top-k changes ───────────────────────────────
pipeline_key = f"{store_choice}|{top_k}|{use_reranker}|{use_multihop}|{filter_page_range}|{filter_content}"

if st.session_state.active_store != pipeline_key:
    try:
        # Build Metadata Filter for Qdrant
        qdrant_filter = None
        if store_choice in ["Qdrant", "Hybrid (Qdrant + BM25)"]:
            from qdrant_client.http import models as rest
            
            must_conditions = []
            
            # Page range filter
            must_conditions.append(
                rest.FieldCondition(
                    key="metadata.page_number",
                    range=rest.Range(gte=filter_page_range[0], lte=filter_page_range[1])
                )
            )
            
            # Content type filter
            if filter_content:
                must_conditions.append(
                    rest.FieldCondition(
                        key="metadata.content_type",
                        match=rest.MatchAny(any=filter_content)
                    )
                )
            
            qdrant_filter = rest.Filter(must=must_conditions)

        # Base retrieval K needs to be higher if we are reranking
        fetch_k = top_k * 3 if use_reranker else top_k

        if store_choice == "FAISS":
            base_retriever = _load_faiss().as_retriever(search_kwargs={"k": fetch_k})
        elif store_choice == "Qdrant":
            base_retriever = _load_qdrant().as_retriever(search_kwargs={"k": fetch_k, "filter": qdrant_filter})
        elif store_choice == "BM25 (Sparse)":
            bm25 = _load_bm25()
            bm25.k = fetch_k
            base_retriever = bm25
        else: # Hybrid
            from langchain_classic.retrievers import EnsembleRetriever
            qdrant_retriever = _load_qdrant().as_retriever(search_kwargs={"k": fetch_k, "filter": qdrant_filter})
            bm25 = _load_bm25()
            bm25.k = fetch_k
            base_retriever = EnsembleRetriever(
                retrievers=[qdrant_retriever, bm25],
                weights=[0.5, 0.5]
            )

        # Apply Multi-hop if enabled
        if use_multihop:
            from src.generation.query_transformer import get_multi_query_retriever
            from langchain_google_vertexai import ChatVertexAI
            llm = ChatVertexAI(
                model_name=settings.gemini_model,
                project=settings.gcp_project,
                location=settings.gcp_location,
                temperature=0.0
            )
            base_retriever = get_multi_query_retriever(base_retriever, llm)

        # Apply reranker if enabled
        if use_reranker:
            final_retriever = get_reranker(base_retriever, top_k=top_k)
        else:
            final_retriever = base_retriever

        st.session_state.pipeline = RAGPipeline(
            final_retriever,
            cache_namespace=st.session_state.cache_namespace,
        )
        st.session_state.active_store = pipeline_key
    except Exception as e:
        import traceback
        st.error(f"⚠️ Failed to initialize retriever: {e}")
        st.error(traceback.format_exc())
        st.session_state.pipeline = None
        st.session_state.active_store = None

pipeline = st.session_state.pipeline

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Chat
# ═══════════════════════════════════════════════════════════════════════════════
if page == "💬 Chat":
    st.title("📊 IFC Annual Report 2024 — Q&A")
    badge_color = {"FAISS": "🟦", "Qdrant": "🟩", "Both (Merged)": "🟪"}
    st.caption(
        f"{badge_color.get(store_choice, '⬜')} Active retriever: **{store_choice}** · "
        f"Top-K: **{top_k}** · Qdrant: **{qdrant_badge}**"
    )
    st.markdown("---")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("retrieved_docs"):
                with st.expander(
                    f"Retrieved Chunks ({len(message['retrieved_docs'])})",
                    expanded=False,
                ):
                    for idx, doc_info in enumerate(message["retrieved_docs"], start=1):
                        st.markdown(
                            f"**Chunk {idx}** · Page **{doc_info['page_number']}** · Type **{doc_info['content_type']}**"
                        )
                        st.code(doc_info["page_content"], language="text")

    if prompt := st.chat_input("Ask a question about the IFC Annual Report..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if pipeline:
            import time
            start_time = time.time()
            retrieved_docs = pipeline.get_retrieved_docs(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in pipeline.generate_stream(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                latency = time.time() - start_time
                full_response += f"\n\n⏱️ *Response Time: {latency:.2f}s*"
                message_placeholder.markdown(full_response)

                with st.expander(f"Retrieved Chunks ({len(retrieved_docs)})", expanded=False):
                    for idx, doc in enumerate(retrieved_docs[:top_k], start=1):
                        metadata = doc.metadata or {}
                        page_number = metadata.get("page_number", "?")
                        content_type = metadata.get("content_type", "unknown")
                        st.markdown(
                            f"**Chunk {idx}** · Page **{page_number}** · Type **{content_type}**"
                        )
                        st.code(doc.page_content[:1200], language="text")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "retrieved_docs": [
                        {
                            "page_number": (doc.metadata or {}).get("page_number", "?"),
                            "content_type": (doc.metadata or {}).get("content_type", "unknown"),
                            "page_content": doc.page_content[:1200],
                        }
                        for doc in retrieved_docs[:top_k]
                    ],
                }
            )
        else:
            st.warning("Pipeline not initialized. Check your vector store setup.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Evaluation Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Evaluation":
    st.title("📊 Phase 2 — Evaluation Dashboard")
    st.caption("Run `python evaluate.py` first to generate results.")
    st.markdown("---")

    eval_dir = project_root / "data" / "evaluation"
    ragas_path = eval_dir / "ragas_results.csv"
    judge_path = eval_dir / "llm_judge_results.csv"

    ragas_df = pd.read_csv(ragas_path) if ragas_path.exists() else None
    judge_df = pd.read_csv(judge_path) if judge_path.exists() else None

    if ragas_df is None and judge_df is None:
        st.info(
            "No evaluation results found yet.\n\n"
            "Run the evaluation pipeline:\n```bash\npython evaluate.py\n```"
        )
    else:
        # ── Filters ───────────────────────────────────────────────────────────
        content_types = ["All"]
        for df in [ragas_df, judge_df]:
            if df is not None and "context_content_type" in df.columns:
                content_types += [t for t in df["context_content_type"].dropna().unique() if t not in content_types]

        selected_type = st.selectbox("Filter by content type:", content_types)

        def filter_df(df):
            if df is None:
                return None
            if selected_type != "All" and "context_content_type" in df.columns:
                return df[df["context_content_type"] == selected_type].reset_index(drop=True)
            return df

        ragas_filtered = filter_df(ragas_df)
        judge_filtered = filter_df(judge_df)

        col1, col2 = st.columns(2)

        # ── RAGAS Metrics ─────────────────────────────────────────────────────
        with col1:
            st.subheader("🔬 RAGAS Metrics")
            if ragas_filtered is not None:
                ragas_metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                available = [m for m in ragas_metrics if m in ragas_filtered.columns]
                if available:
                    means = ragas_filtered[available].mean()
                    st.bar_chart(means, use_container_width=True)

                    st.markdown("**Mean Scores:**")
                    for m in available:
                        score = means[m]
                        color = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
                        st.write(f"{color} `{m}`: **{score:.3f}**")
                else:
                    st.warning("RAGAS metric columns not found in results CSV.")
            else:
                st.info("Run `python evaluate.py` to generate RAGAS results.")

        # ── LLM Judge Scores ──────────────────────────────────────────────────
        with col2:
            st.subheader("⚖️ LLM-as-a-Judge")
            if judge_filtered is not None:
                judge_metrics = ["correctness", "completeness", "clarity"]
                available = [m for m in judge_metrics if m in judge_filtered.columns]
                if available:
                    means = judge_filtered[available].mean()
                    st.bar_chart(means, use_container_width=True)

                    st.markdown("**Mean Scores (out of 5):**")
                    for m in available:
                        score = means[m]
                        color = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
                        st.write(f"{color} `{m}`: **{score:.2f} / 5**")

                    if "average_score" in judge_filtered.columns:
                        avg = judge_filtered["average_score"].mean()
                        st.metric("Overall Average", f"{avg:.2f} / 5")
                else:
                    st.warning("Judge score columns not found in results CSV.")
            else:
                st.info("Run `python evaluate.py` to generate LLM Judge results.")

        st.markdown("---")

        # ── Per-question breakdown ─────────────────────────────────────────────
        st.subheader("🔎 Per-Question Breakdown")
        breakdown_tabs = st.tabs(["RAGAS", "LLM Judge"])

        with breakdown_tabs[0]:
            if ragas_filtered is not None:
                # Handle RAGAS column naming changes in newer versions
                if "user_input" in ragas_filtered.columns and "question" not in ragas_filtered.columns:
                    ragas_filtered = ragas_filtered.rename(columns={"user_input": "question", "response": "answer"})
                
                ragas_metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                show_cols = ["question"] + [m for m in ragas_metrics if m in ragas_filtered.columns]
                if "context_content_type" in ragas_filtered.columns:
                    show_cols.append("context_content_type")
                st.dataframe(ragas_filtered[show_cols], use_container_width=True)
            else:
                st.info("No RAGAS results yet.")

        with breakdown_tabs[1]:
            if judge_filtered is not None:
                show_cols = ["question", "generated_answer"] + \
                            [m for m in ["correctness", "completeness", "clarity", "average_score"] if m in judge_filtered.columns]
                if "context_content_type" in judge_filtered.columns:
                    show_cols.append("context_content_type")
                st.dataframe(judge_filtered[show_cols], use_container_width=True)
            else:
                st.info("No LLM Judge results yet.")
