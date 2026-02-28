import streamlit as st
import requests
import time

# ── Configuration ────────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/upload"
QUERY_ENDPOINT = f"{API_BASE_URL}/query"

ALLOWED_EXTENSIONS = ["pdf", "txt", "docx"]


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assist",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Global */
        .block-container { max-width: 820px; padding-top: 2rem; }

        /* Header */
        .app-header {
            text-align: center;
            padding: 1.5rem 0 0.5rem;
        }
        .app-header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .app-header p {
            font-size: 1rem;
            opacity: 0.7;
            margin-top: 0;
        }

        /* Divider */
        .section-divider {
            border: none;
            border-top: 1px solid rgba(128,128,128,0.25);
            margin: 1.5rem 0;
        }

        /* Cards */
        .result-card {
            background: linear-gradient(135deg, #f8f9fc 0%, #eef1f8 100%);
            border-radius: 12px;
            padding: 1.5rem 1.75rem;
            margin-top: 1rem;
            border-left: 4px solid #4f8bf9;
            color: #1a1a2e !important;   /* 👈 ADD THIS */
}
        .result-card .answer-text {
            font-size: 1.05rem;
            line-height: 1.75;
            color: #1a1a2e;
        }
        .result-card .source-link {
            display: inline-block;
            margin-top: 0.75rem;
            font-size: 0.9rem;
            color: #4f8bf9;
        }

        .upload-result-card {
            background: linear-gradient(135deg, #f0faf4 0%, #e4f5ec 100%);
            border-radius: 12px;
            padding: 1.5rem 1.75rem;
            margin-top: 1rem;
            border-left: 4px solid #2ecc71;
            color: #1a1a2e !important;   /* 👈 ADD THIS */


        .error-card {
            background: linear-gradient(135deg, #fef4f4 0%, #fde8e8 100%);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-top: 1rem;
            border-left: 4px solid #e74c3c;
            color: #7a1a1a;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        .sidebar-section-title {
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            opacity: 0.6;
            margin-bottom: 0.25rem;
        }

        /* Chat history items */
        .history-item {
            padding: 0.6rem 0.75rem;
            border-radius: 8px;
            margin-bottom: 0.35rem;
            font-size: 0.88rem;
            cursor: default;
            transition: background 0.15s;
        }
        .history-item:hover { background: rgba(79,139,249,0.08); }
        .history-q { font-weight: 600; color: #1a1a2e; }
        .history-a { opacity: 0.72; margin-top: 2px; }

        /* Hide Streamlit branding */
        #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session State ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "upload_result" not in st.session_state:
    st.session_state.upload_result = None


# ── Helper Functions ─────────────────────────────────────────────────────────
def upload_files(files) -> dict:
    """Send files to the FastAPI /upload endpoint."""
    multipart = [("uploaded_files", (f.name, f.getvalue(), f.type)) for f in files]
    response = requests.post(UPLOAD_ENDPOINT, files=multipart, timeout=300)
    response.raise_for_status()
    return response.json()


def query_rag(query: str, top_k: int, min_score: float) -> dict:
    """Send a query to the FastAPI /query endpoint."""
    payload = {"query": query, "top_k": top_k, "min_score": min_score}
    response = requests.post(QUERY_ENDPOINT, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def truncate(text: str, length: int = 80) -> str:
    return text[:length] + "…" if len(text) > length else text


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️  Settings")

    top_k = st.slider(
        "Top K results",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of document chunks to retrieve.",
    )
    min_score = st.slider(
        "Min relevance score",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum similarity score to include a chunk.",
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Chat History ─────────────────────────────────────────────────────────
    st.markdown("### 💬  Chat History")

    if st.session_state.chat_history:
        if st.button("Clear history", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        for idx, item in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(
                f"""
                <div class='history-item'>
                    <div class='history-q'>Q: {truncate(item["query"], 60)}</div>
                    <div class='history-a'>{truncate(item["answer"], 90)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("No queries yet. Ask a question to get started.")


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class='app-header'>
        <h1>🔍 RAG Assist</h1>
        <p>Upload documents and ask questions — powered by Groq &amp; Gemini LLMs</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_upload, tab_query = st.tabs(["📂  Upload Documents", "💡  Ask a Question"])

# ── Upload Tab ───────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("##### Upload your documents for processing")
    st.caption("Supported formats: PDF, TXT, DOCX")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)}** file(s) selected:")
        for f in uploaded_files:
            size_kb = len(f.getvalue()) / 1024
            st.text(f"  • {f.name}  ({size_kb:.1f} KB)")

    col_l, col_r = st.columns([3, 1])
    with col_r:
        submit_upload = st.button(
            "Upload & Process",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        )

    if submit_upload and uploaded_files:
        with st.spinner("Processing documents — this may take a moment…"):
            try:
                result = upload_files(uploaded_files)
                st.session_state.upload_result = result
            except requests.exceptions.ConnectionError:
                st.session_state.upload_result = {
                    "error": "Could not connect to the backend. Make sure the FastAPI server is running on port 8000."
                }
            except requests.exceptions.HTTPError as exc:
                st.session_state.upload_result = {
                    "error": f"Server returned an error: {exc.response.status_code} — {exc.response.text}"
                }
            except Exception as exc:
                st.session_state.upload_result = {"error": str(exc)}

    # Display upload result
    if st.session_state.upload_result:
        res = st.session_state.upload_result
        if "error" in res:
            st.markdown(
                f"<div class='error-card'>⚠️ {res['error']}</div>",
                unsafe_allow_html=True,
            )
        else:
            docs = res.get("documents_processed", 0)
            vecs = res.get("vectors_stored", 0)
            st.markdown(
                f"""
                <div class='upload-result-card'>
                    <strong>✅  Upload Successful</strong><br>
                    <span style='font-size:0.95rem'>
                        Documents processed: <strong>{docs}</strong><br>
                        Vectors stored: <strong>{vecs}</strong>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Query Tab ────────────────────────────────────────────────────────────────
with tab_query:
    st.markdown("##### Ask a question about your documents")

    query_input = st.text_area(
        "Your question",
        placeholder="e.g. What services does the company provide?",
        height=100,
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns([3, 1])
    with col_b:
        submit_query = st.button(
            "Submit",
            type="primary",
            use_container_width=True,
            disabled=not query_input.strip(),
        )

    if submit_query and query_input.strip():
        with st.spinner("Searching documents and generating answer…"):
            try:
                result = query_rag(query_input.strip(), top_k, min_score)

                answer_text = result.get("answer", "No answer returned.")
                source_url = None

                # Separate source URL if appended to the answer
                if "\nSource: http" in answer_text:
                    parts = answer_text.rsplit("\nSource: ", 1)
                    answer_text = parts[0]
                    source_url = parts[1].strip()

                # Build card HTML
                source_html = ""
                if source_url:
                    source_html = (
                        f"<a class='source-link' href='{source_url}' target='_blank'>"
                        f"📎 Source: {truncate(source_url, 70)}</a>"
                    )

                st.markdown(
                    f"""
                    <div class='result-card'>
                        <div class='answer-text'>{answer_text}</div>
                        {source_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Save to history
                st.session_state.chat_history.append(
                    {"query": query_input.strip(), "answer": answer_text}
                )

            except requests.exceptions.ConnectionError:
                st.markdown(
                    "<div class='error-card'>⚠️ Could not reach the backend. "
                    "Ensure the FastAPI server is running on port 8000.</div>",
                    unsafe_allow_html=True,
                )
            except requests.exceptions.HTTPError as exc:
                st.markdown(
                    f"<div class='error-card'>⚠️ Server error: {exc.response.status_code}</div>",
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.markdown(
                    f"<div class='error-card'>⚠️ {exc}</div>",
                    unsafe_allow_html=True,
                )
