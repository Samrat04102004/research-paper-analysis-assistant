import streamlit as st
import os
import re
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# Import your custom modules
from core.parser import SectionAwareParser
from core.retriever import AdvancedRetriever
from core.evaluator import RagasEvaluator

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Configuration & Setup
load_dotenv()
st.set_page_config(page_title="Research Assistant Pro", layout="wide")

def deduplicate_docs(docs):
    """Filters out chunks with identical content to stop repetitive citations."""
    unique_texts = set()
    deduped = []
    for doc in docs:
        content_hash = hash(doc.page_content.strip())
        if content_hash not in unique_texts:
            unique_texts.add(content_hash)
            deduped.append(doc)
    return deduped

# --- UI ENHANCEMENTS (CSS) ---
st.markdown("""
    <style>
    .main-title { font-size: 3rem !important; font-weight: 800; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-title { font-size: 1.2rem; color: #64748B; margin-bottom: 2rem; }
    div[data-testid="stMarkdownContainer"] p { font-size: 1.1rem !important; line-height: 1.6; }
    div[data-testid="stMetricValue"] { font-size: 2.5rem !important; font-weight: 700; color: #2563EB; }
    .evidence-card {
        background-color: #F8FAFC;
        border-left: 5px solid #3B82F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def clean_pdf_text(text):
    text = re.sub(r'<(EOS|pad|pad|unk)>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Only uploads folder needed now
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Initialize Session States
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "retriever" not in st.session_state: st.session_state.retriever = None
if "last_response" not in st.session_state: st.session_state.last_response = None
if "query" not in st.session_state: st.session_state.query = ""

# --- SIDEBAR: Document Ingestion ---
with st.sidebar:
    st.header("📁 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type="pdf")

    if uploaded_file:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Index Document", width="stretch"):
            with st.status("Processing PDF...", expanded=True) as status:

                # Fresh in-memory Qdrant client (no disk, no reset needed)
                client = QdrantClient(":memory:")

                # Create clean collection
                client.create_collection(
                    collection_name="research_papers",
                    vectors_config=rest.VectorParams(
                        size=1536,
                        distance=rest.Distance.COSINE
                    )
                )

                st.write("Parsing sections and metadata...")
                parser = SectionAwareParser()
                docs = parser.parse(file_path)
                
                st.write("Deduplicating chunks...")
                for doc in docs:
                    doc.page_content = clean_pdf_text(doc.page_content)
                docs = deduplicate_docs(docs)
                
                st.write("Generating embeddings and indexing...")
                embeddings = OpenAIEmbeddings()
                
                st.session_state.vectorstore = QdrantVectorStore(
                    client=client,
                    collection_name="research_papers",
                    embedding=embeddings
                )
                
                st.session_state.vectorstore.add_documents(docs)
                
                st.write("Configuring Ensemble Retriever...")
                st.session_state.retriever = AdvancedRetriever.get_ensemble_retriever(
                    docs, embeddings, st.session_state.vectorstore
                )

                status.update(label="Indexing Complete!", state="complete", expanded=False)

            st.success(f"Indexed {len(docs)} unique chunks.")

# --- MAIN UI: Analysis & QA ---
st.markdown('<h1 class="main-title">🔬 Research Assistant Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Citation-Grounded Paper Analysis</p>', unsafe_allow_html=True)

with st.container():
    col_input, col_btn = st.columns([0.8, 0.2])
    with col_input:
        user_query = st.text_input("Technical Question:", placeholder="e.g. Describe the self-attention mechanism...", label_visibility="collapsed")
    with col_btn:
        submit_button = st.button("Analyze Paper", width="stretch", type="primary")

# Logic for Generating Answer
if submit_button and user_query:
    if not st.session_state.retriever:
        st.error("Please upload and index a document first.")
    else:
        with st.spinner("Analyzing and generating citations..."):
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            prompt = ChatPromptTemplate.from_template("""
            You are an expert Scientific Research Assistant specialized in analyzing academic papers.

            Your task is to answer the question STRICTLY using the provided context.

            ======================
            GROUNDING RULES
            ======================
            1. Use ONLY the information present in the context.
            2. Do NOT use prior knowledge, assumptions, or external facts.
            3. If the answer is not explicitly supported, reply exactly:
               "The provided document does not contain this information."
            4. If multiple sources support a claim, include all in a single bracket.
            5. Never fabricate numbers, results, or explanations.
            6. Do NOT copy long text verbatim; summarize faithfully.

            ======================
            ANSWER STYLE
            ======================
            • Write in clear academic tone.
            • Be concise but complete.
            • Prefer short evidence-backed paragraphs.
            • Avoid repetition and filler language.

            ======================
            OUTPUT FORMAT
            ======================
            Provide:

            
            <well-structured, citation-grounded explanation>

            ======================
            CONTEXT
            ======================
            {context}

            ======================
            QUESTION
            ======================
            {input}
            """)
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(st.session_state.retriever, combine_docs_chain)
            
            response = rag_chain.invoke({"input": user_query})
            st.session_state.last_response = response
            st.session_state.query = user_query

# Persistent Display of Results
if st.session_state.last_response:
    st.subheader("Analysis Results")
    st.markdown(st.session_state.last_response["answer"])
    
    with st.expander("🔍 View Source Evidence & Metadata"):
        displayed_evidence = deduplicate_docs(st.session_state.last_response["context"])
        for doc in displayed_evidence:
            st.markdown(f"""
            <div class="evidence-card">
                <strong>Section:</strong> {doc.metadata.get('section')} | <strong>Page:</strong> {doc.metadata.get('page')}<br><br>
                {doc.page_content}
            </div>
            """, unsafe_allow_html=True)

    # --- EVALUATION DASHBOARD ---
    st.divider()
    st.header("📊 System Performance Audit")
    
    if st.button("Run RAGAS Evaluation", type="secondary"):
        with st.spinner("Evaluating Faithfulness and Relevancy..."):
            evaluator = RagasEvaluator()
            context_list = [doc.page_content for doc in st.session_state.last_response["context"]]
            
            eval_results = evaluator.run_evaluation(
                question=st.session_state.query,
                answer=st.session_state.last_response["answer"],
                contexts=context_list
            )
            
            col1, col2 = st.columns(2)
            f_score = eval_results["faithfulness"].iloc[0]
            r_score = eval_results["answer_relevancy"].iloc[0]
            
            col1.metric("Faithfulness", f"{f_score:.2f}")
            col2.metric("Answer Relevancy", f"{r_score:.2f}")
            
            st.subheader("📝 Audit Summary")
            audit_df = pd.DataFrame({
                "Metric": ["Faithfulness", "Answer Relevancy"],
                "Score": [f"{f_score:.2f}", f"{r_score:.2f}"],
                "Threshold": [">= 0.80", ">= 0.80"],
                "Status": ["✅ PASS" if f_score >= 0.8 else "❌ FAIL", 
                           "✅ PASS" if r_score >= 0.8 else "❌ FAIL"]
            })
            st.table(audit_df)

            if f_score > 0.80 and r_score > 0.80:
                st.success("High Quality: Answer is grounded and highly relevant.")
            else:
                st.warning("Needs Review: Low grounding or relevancy detected.")
            
            with st.expander("🛠️ View Detailed Trace"):
                st.data_editor(
                    eval_results, 
                    width="stretch",
                    column_config={
                        "retrieved_contexts": st.column_config.ListColumn("Retrieved Contexts"),
                        "response": st.column_config.TextColumn("Model Response", width="large"),
                        "contexts": st.column_config.ListColumn("Contexts")
                    },
                    hide_index=True
                )
