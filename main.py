import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate


# 🚀 Page Config
st.set_page_config(
    page_title="InsightSphere: GenAI Knowledge Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🖌️ Custom CSS for padding, colors, footer position
st.markdown("""
    <style>
    .block-container {
        padding-left: 50px;
        padding-right: 50px;
    }
    body {
        background-color: #0B132B;
        margin: 0;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #1C2541;
        color: #E0E1DD;
        text-align: center;
        padding: 8px 0;
        font-size: 14px;
        z-index: 100;
    }
    </style>
""", unsafe_allow_html=True)

# 📌 Title Banner
st.markdown("""
    <div style='background-color: #1C2541; padding: 20px; border-radius: 12px; text-align: center;'>
        <h1 style='color: #F4D35E; font-family: Arial, sans-serif; margin: 5px 0 0 0;'>
            InsightSphere: GenAI Knowledge Hub 🚀
        </h1>
        <p style='color: #E0E1DD; font-size: 18px; margin-top: 5px;'>
            A smart AI-powered assistant for research, insights, and information discovery.
        </p>
    </div>
""", unsafe_allow_html=True)

# 📦 Load environment variables
load_dotenv()

# 📂 Sidebar Inputs
st.sidebar.title("📂 Upload / Enter Content")

urls = []
for i in range(1):
    url = st.sidebar.text_input(f"🔗 URL {i+1}")
    urls.append(url)

uploaded_pdfs = st.sidebar.file_uploader("📄 Upload PDF files", type=['pdf'], accept_multiple_files=True)
uploaded_texts = st.sidebar.file_uploader("📝 Upload Text files", type=['txt'], accept_multiple_files=True)
process_clicked = st.sidebar.button("📚 Process All Data")

vectorstore_folder = "faiss_store_combined"
main_placeholder = st.empty()

# 📖 Local LLM Pipeline
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=500, temperature=0.7)
llm = HuggingFacePipeline(pipeline=pipe)

# 📚 Process Documents
if process_clicked:
    all_data = []

    url_loader = UnstructuredURLLoader(urls=[u for u in urls if u])
    url_data = url_loader.load()
    for i, doc in enumerate(url_data):
        doc.metadata['source'] = urls[i]
    all_data.extend(url_data)

    for uploaded_pdf in uploaded_pdfs or []:
        with open(f"temp_{uploaded_pdf.name}", "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        pdf_loader = PyPDFLoader(f"temp_{uploaded_pdf.name}")
        pdf_docs = pdf_loader.load()
        for doc in pdf_docs:
            doc.metadata['source'] = uploaded_pdf.name
        all_data.extend(pdf_docs)
        os.remove(f"temp_{uploaded_pdf.name}")

    for uploaded_text in uploaded_texts or []:
        with open(f"temp_{uploaded_text.name}", "wb") as f:
            f.write(uploaded_text.getbuffer())
        text_loader = TextLoader(f"temp_{uploaded_text.name}")
        text_docs = text_loader.load()
        for doc in text_docs:
            doc.metadata['source'] = uploaded_text.name
        all_data.extend(text_docs)
        os.remove(f"temp_{uploaded_text.name}")

    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    docs = text_splitter.split_documents(all_data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(vectorstore_folder)
    main_placeholder.success("✅ Vector Index ready!")

# 🔍 Ask Question Section
st.markdown("""
    <h3 style='margin-top:15px; color: #F4D35E;'>🔍 <b>Ask a New Question:</b></h3>
""", unsafe_allow_html=True)

query = st.text_input("")

# 📖 Process Question & Show Answer
if query:
    if os.path.exists(vectorstore_folder):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(vectorstore_folder, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()

        prompt_template = """Use the following context to answer the question.
        If you don't know, just say you don't know.

        Context:
        {context}

        Question: {question}

        Answer:"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        result = qa_chain(query)

        st.markdown(f"<h3 style='color:#F4D35E;'>📖 Answer:</h3>", unsafe_allow_html=True)
        st.write(result["result"])

        st.markdown(f"<h4 style='color:#F4D35E;'>📌 Sources:</h4>", unsafe_allow_html=True)
        source_docs = result["source_documents"]
        if source_docs:
            for i, doc in enumerate(source_docs):
                if i >= 2:
                    break
                source_url = doc.metadata.get("source", "No source found")
                st.write(f"🔗 {source_url}")
        else:
            st.write("No sources found.")
    else:
        st.error("No vectorstore found! Please process data first.")

# 📌 Footer Fixed Bottom
st.markdown("""
    <div class="footer">
    © 2025 InsightSphere | Powered by <span style='color: #4CAF50;'>GenAI 🚀</span> | Built by <span style='color: #FF9800;'>Nikita Mahajan</span>
    </div>
""", unsafe_allow_html=True)
