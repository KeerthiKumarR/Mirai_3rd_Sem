import streamlit as st
import time
from rag import answer, CulinaryRAGSystem
from rag.config import COLLECTION_NAME, DEFAULT_PDF_PATH

# Streamlit Page Config
st.set_page_config(
    page_title="Culinary RAG Assistant - Le Grand Bistro",
    page_icon="👨‍🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .source-badge {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .not-found-badge {
        display: inline-block;
        background-color: #fee2e2;
        color: #b91c1c;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG System in Cache
@st.cache_resource(show_spinner="Connecting to ChromaDB and initializing RAG engine...")
def load_rag():
    return CulinaryRAGSystem.get_instance()

rag_engine = load_rag()

# Header
st.markdown('<div class="main-title">👨‍🍳 Culinary Operations & Allergen RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Interactive Q&A powered by ChromaDB Vector Database & Strictly Grounded Inference for <i>Le Grand Bistro Manual</i></div>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("📖 System Overview")
    st.markdown("""
    - **Corpus:** `grand_bistro_culinary_recipe_manual_50p.pdf`
    - **Pages:** 50 Pages
    - **Vector DB:** ChromaDB (Persistent)
    - **Embeddings:** `all-MiniLM-L6-v2` (Local ONNX)
    - **Strict Grounding:** Enabled (`[not found]` fallback)
    """)
    
    st.divider()
    st.subheader("⚡ Quick Assessment Questions")
    
    presets = [
        ("Q1: Truffle Ganache Quantities", "What exact quantities, in grams, of dark couverture chocolate and heavy cream are required for a standard 20-portion batch of Signature Truffle Ganache?"),
        ("Q2: Autumn Tasting Menu Allergens", "According to the Allergen Matrix, which dishes on the Autumn Tasting Menu contain tree nuts or sesame?"),
        ("Q3: Eiffel Tower Closing Time (Negative Test)", "At what time does the Eiffel Tower in Paris close to visitors during the winter season?"),
        ("Q4: Sous-Vide Duck Breast Safety", "What is the mandatory minimum internal cooking temperature, in degrees Celsius, and the required holding time for sous-vide duck breast, as specified in the Food Safety section?"),
        ("Q5: Noma Michelin Rating (Negative Test)", "What is the current Michelin-star rating of Noma in Copenhagen?")
    ]
    
    selected_preset = None
    for label, q_text in presets:
        if st.button(label, use_container_width=True):
            selected_preset = q_text

# Session State for Query History
if "history" not in st.session_state:
    st.session_state.history = []

# Main Input Section
default_query = selected_preset if selected_preset else ""
query = st.text_input(
    "Ask any culinary question, recipe detail, allergen rule, or food safety parameter:",
    value=default_query,
    placeholder="e.g., What are the ingredients for Signature Truffle Ganache?"
)

col_submit, col_clear = st.columns([1, 6])
with col_submit:
    submit_button = st.button("Search & Answer", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear Chat"):
        st.session_state.history = []
        st.rerun()

# Execution
if (submit_button or selected_preset) and query.strip():
    with st.spinner("Retrieving relevant passages and synthesizing grounded answer..."):
        start_time = time.time()
        
        # 1. Retrieve
        retrieval_res = rag_engine.retriever.retrieve(query.strip())
        
        # 2. Answer
        result = rag_engine.generator.generate(query.strip(), retrieval_res)
        latency = time.time() - start_time
        
        # Save to history
        st.session_state.history.insert(0, {
            "query": query.strip(),
            "answer": result["answer"],
            "sources": result["sources"],
            "latency": latency,
            "retrieval": retrieval_res
        })

# Render Results
if st.session_state.history:
    st.subheader("💡 Results")
    for idx, item in enumerate(st.session_state.history):
        with st.container():
            st.markdown(f"### **Q: {item['query']}**")
            
            # Answer Box
            if item["answer"] == "[not found]":
                st.error("❌ **Answer:** `[not found]`")
                st.caption("ℹ️ The information requested is not present in the 50-page manual. Strict grounding prevented hallucination.")
            else:
                st.success(f"**Answer:**\n\n{item['answer']}")
                
            # Sources
            col_src, col_lat = st.columns([4, 1])
            with col_src:
                if item["sources"]:
                    st.markdown("**Source Citations:**")
                    badges_html = " ".join([f'<span class="source-badge">📄 {s}</span>' for s in item["sources"]])
                    st.markdown(badges_html, unsafe_allow_html=True)
                else:
                    st.markdown('<span class="not-found-badge">No Source Reference (Out of Corpus)</span>', unsafe_allow_html=True)
            with col_lat:
                st.caption(f"⚡ Latency: {item['latency']:.2f}s")
                
            # Evidence inspection drawer
            chunks = item["retrieval"].get("chunks", [])
            if chunks:
                with st.expander(f"🔍 Inspect Retrieved Vector Chunks ({len(chunks)} chunks)"):
                    for c_idx, c in enumerate(chunks, 1):
                        st.markdown(f"**Chunk {c_idx}** | Page: `{c.get('page')}` | ID: `{c.get('id')}` | Distance: `{c.get('distance', 0):.4f}`")
                        st.code(c["text"], language="markdown")
                        st.divider()
                        
            st.divider()
