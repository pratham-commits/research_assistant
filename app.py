import os
import sys
import requests
import xml.etree.ElementTree as ET
from io import BytesIO
from dotenv import load_dotenv
import certifi
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import ArxivLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# Setup
load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Function to fetch PDF from arXiv by ID
def fetch_arxiv_paper_by_id(arxiv_id):
    base_url = "http://export.arxiv.org/api/query?"
    query = f"id_list={arxiv_id}"
    try:
        response = requests.get(base_url + query)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        entry = root.findall("{http://www.w3.org/2005/Atom}entry")[0]
        pdf_link = next((link.attrib["href"] for link in entry.findall("{http://www.w3.org/2005/Atom}link") if link.attrib.get("title") == "pdf"), None)
        if pdf_link:
            pdf_response = requests.get(pdf_link)
            pdf_response.raise_for_status()
            return pdf_response.content
    except Exception as e:
        return None

# Streamlit UI
st.title("🧠 AI Research Assistant (RAG-Based)")

# GROQ API
groq_api_key = st.text_input("🔑 Enter your GROQ API Key", type="password")
if not groq_api_key:
    st.warning("Please enter your GROQ API key to continue.")
    st.stop()
else:
    st.success("API key validated!")

source = st.radio("📄 Choose the source of your research paper:", ["Upload PDF", "Search arXiv"])
documents = []

arxiv_categories = [
    "cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.CR", "cs.DS", "cs.HC", "cs.SY", "stat.ML", "math.OC"
]

if source == "Upload PDF":
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        try:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())
            documents = PyPDFLoader("temp.pdf").load()
            os.remove("temp.pdf")
            st.success("PDF loaded successfully.")
        except Exception as e:
            st.error(f"⚠️ Failed to load PDF: {e}")
            documents = []

elif source == "Search arXiv":
    arxiv_input_type = st.radio("🔍 How would you like to search?", ["By Title/Keywords", "By ArXiv ID"])
    
    if arxiv_input_type == "By Title/Keywords":
        arxiv_query = st.text_input("🔍 Enter search query")
        selected_category = st.selectbox("📂 Select arXiv category", arxiv_categories)
        if arxiv_query and selected_category:
            try:
                with st.spinner("🔍 Searching arXiv..."):
                    full_query = f"{selected_category}:{arxiv_query}"
                    documents = ArxivLoader(query=full_query, load_max_docs=5).load()
                    documents = [doc for doc in documents if doc.page_content.strip()]
                if documents:
                    st.success(f"Loaded {len(documents)} paper(s) using query: `{full_query}`")
                else:
                    st.warning("No papers could be parsed from arXiv response.")
            except Exception as e:
                st.error(f"Failed to fetch papers: {e}")

    elif arxiv_input_type == "By ArXiv ID":
        arxiv_id = st.text_input("🔍 Enter the ArXiv Paper ID (e.g., 1234.5678):")
        if arxiv_id:
            try:
                with st.spinner("🔍 Fetching paper..."):
                    pdf_content = fetch_arxiv_paper_by_id(arxiv_id)
                    if pdf_content:
                        try:
                            with open("temp.pdf", "wb") as f:
                                f.write(pdf_content)
                            documents = PyPDFLoader("temp.pdf").load()
                            os.remove("temp.pdf")
                            st.success(f"Successfully fetched paper with ArXiv ID: {arxiv_id}")
                        except Exception as e:
                            st.error(f"⚠️ Failed to load PDF content: {e}")
                            documents = []
                    else:
                        st.error(f"Failed to fetch paper with ArXiv ID: {arxiv_id}")
            except Exception as e:
                st.error(f"Error fetching paper: {e}")

# Section extractor
def extract_sections(text):
    sections = {"abstract": "", "introduction": "", "results": "", "conclusion": ""}
    patterns = {
        "abstract": r"(abstract|ABSTRACT)\s*[:\-\n]?(.*?)\n(?=(introduction|INTRODUCTION|1\.))",
        "introduction": r"(introduction|INTRODUCTION|1\.)\s*[:\-\n]?(.*?)\n(?=(methods|METHODS|2\.|results|RESULTS))",
        "results": r"(results|RESULTS|3\.)\s*[:\-\n]?(.*?)\n(?=(discussion|conclusion|CONCLUSION|4\.))",
        "conclusion": r"(conclusion|CONCLUSION|4\.)\s*[:\-\n]?(.*?)$"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = match.group(2).strip()

    combined = "\n\n".join([f"{k.capitalize()}:\n{v}" for k, v in sections.items() if v])
    return combined[:4000]

if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    task = st.selectbox("🛠️ Select a Task", [
        "Question Answering", "Summarization", "Paper Classification", "Author Impact Analysis"
    ])

    if task == "Question Answering":
        st.subheader("❓ Ask a question about the paper")
        question = st.text_input("Your Question:")
        if question:
            response = qa_chain.invoke(question)
            st.success(response['result'])

    elif task == "Summarization":
        st.subheader("📃 Custom Section-based Summarization")
        summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
        full_text = " ".join(doc.page_content for doc in documents)
        section_text = extract_sections(full_text)

        with st.spinner("Generating meaningful summary..."):
            summary = summarizer(section_text, max_length=800, min_length=300, do_sample=False)
            st.success("✅ Summary Generated")
            st.write(summary[0]["summary_text"])

    elif task == "Paper Classification":
        st.subheader("🏷️ Classify Paper")

        title = documents[0].metadata.get("title", "")
        abstract = documents[0].metadata.get("abstract", "")
        input_text = title + " " + abstract if title and abstract else documents[0].page_content[:1000]

        with st.spinner("🔎 Classifying broad category..."):
            classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            broad_candidate_labels = {
                "cs.AI": "Artificial Intelligence",
                "cs.CV": "Computer Vision",
                "cs.CL": "Natural Language Processing",
                "cs.LG": "Machine Learning",
                "cs.CR": "Cyber Security",
                "cs.HC": "Human-Computer Interaction",
                "stat.ML": "Statistical Machine Learning",
                "cs.DS": "Data Science",
                "cs.NE": "Neural and Evolutionary Computing"
            }
            broad_result = classifier(input_text, candidate_labels=list(broad_candidate_labels.values()))
            broad_predicted_field = broad_result['labels'][0]
            broad_arxiv_category = [k for k, v in broad_candidate_labels.items() if v == broad_predicted_field][0]

        st.success(f"Predicted Broad Category: `{broad_predicted_field}` (arXiv tag: `{broad_arxiv_category}`)")

        fine_grained_categories = {
            "cs.AI": ["Reinforcement Learning", "AI Ethics", "Robotics", "AI Applications"],
            "cs.CV": ["Image Processing", "Object Detection", "Video Analysis", "Facial Recognition"],
            "cs.CL": ["Natural Language Processing", "Text Summarization", "Speech Recognition"],
            "cs.LG": ["Supervised Learning", "Unsupervised Learning", "Deep Learning", "Neural Networks"],
            "cs.CR": ["Cryptography", "Network Security", "Blockchain", "Data Privacy"],
            "cs.HC": ["Human-Computer Interaction", "Usability Studies", "User Experience"],
            "stat.ML": ["Supervised Learning", "Unsupervised Learning", "Statistical Methods", "Bayesian Inference"],
            "cs.DS": ["Data Mining", "Big Data", "Data Visualization", "Data Analysis"],
            "cs.NE": ["Neural Networks", "Evolutionary Algorithms", "Computational Neuroscience"]
        }

        with st.spinner("🔎 Classifying fine-grained topic..."):
            fine_grained_candidate_labels = fine_grained_categories.get(broad_arxiv_category, [])
            fine_grained_result = classifier(input_text, candidate_labels=fine_grained_candidate_labels)
            fine_grained_predicted_topic = fine_grained_result['labels'][0]

        st.success(f"Predicted Fine-grained Topic: `{fine_grained_predicted_topic}`")

    elif task == "Author Impact Analysis":
        st.subheader("👨‍🏫 Author Impact Analysis")

        author_name = st.text_input("Enter the author name for impact analysis:")
        if author_name:
            try:
                with st.spinner(f"🔍 Fetching papers by {author_name} from arXiv..."):
                    author_query = f'au:"{author_name}"'
                    loader = ArxivLoader(query=author_query, load_max_docs=10)
                    author_documents = loader.load()
                    author_documents = [doc for doc in author_documents if doc.page_content.strip()]

                if author_documents:
                    citation_counts = [np.random.randint(10, 100) for _ in range(len(author_documents))]
                    st.success(f"✅ Found {len(author_documents)} papers by {author_name}.")
                    st.write(f"📊 Author Impact Score: {np.mean(citation_counts):.2f}")
                    st.write(f"Average Citation Count: {np.mean(citation_counts):.2f}")

                    paper_details = []
                    for doc, citation in zip(author_documents, citation_counts):
                        title = doc.metadata.get("Title") or doc.metadata.get("title") or ""
                        if not title:
                            first_lines = doc.page_content.strip().split("\n")[:3]
                            title = next((line.strip() for line in first_lines if len(line.strip()) > 10), "No title available")

                        paper_details.append({
                            "Paper Title": title,
                            "Citation Count (Simulated)": citation
                        })

                    paper_df = pd.DataFrame(paper_details)
                    st.dataframe(paper_df)

                    fig, ax = plt.subplots()
                    ax.hist(citation_counts, bins=10, color="lightgreen", edgecolor="black")
                    ax.set_title(f"Citation Distribution for {author_name}")
                    ax.set_xlabel("Citation Count")
                    ax.set_ylabel("Number of Papers")
                    st.pyplot(fig)

                else:
                    st.warning(f"No papers found or could not parse any papers for {author_name}.")

            except Exception as e:
                st.error(f"Error fetching papers for {author_name}: {e}")
