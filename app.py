import os
import streamlit as st

from dotenv import load_dotenv
from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found.")
    st.info(
        "Create a .env file in the project folder:\n\n"
        "GEMINI_API_KEY=your_api_key_here"
    )
    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# MODELS
# ============================================================

LLM_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "gemini-embedding-001"


# ============================================================
# GEMINI EMBEDDING CLASS
# ============================================================

class GeminiEmbeddings(Embeddings):

    def embed_documents(self, texts):
        """
        Convert multiple documents into embeddings.
        """

        embeddings = []

        for text in texts:

            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text
            )

            embeddings.append(
                result.embeddings[0].values
            )

        return embeddings


    def embed_query(self, text):
        """
        Convert a query into an embedding.
        """

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text
        )

        return result.embeddings[0].values


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Gemini PDF Q&A",
    page_icon="📄",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("📄 Gemini PDF Q&A Bot")

st.write(
    "Upload a PDF and ask questions about it using "
    "Gemini + FAISS."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


if uploaded_file is not None:

    # ========================================================
    # SAVE PDF TEMPORARILY
    # ========================================================

    temp_file = os.path.join(
        os.getcwd(),
        f"temp_{uploaded_file.name}"
    )

    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())


    try:

        # ====================================================
        # LOAD PDF
        # ====================================================

        with st.spinner("📖 Reading PDF..."):

            loader = PyPDFLoader(temp_file)

            documents = loader.load()

        st.success(
            f"✅ PDF loaded — {len(documents)} pages"
        )


        # ====================================================
        # SPLIT PDF
        # ====================================================

        with st.spinner("✂️ Splitting document..."):

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )

            docs = text_splitter.split_documents(
                documents
            )

        st.info(
            f"📚 Created {len(docs)} chunks"
        )


        # ====================================================
        # CREATE GEMINI EMBEDDINGS
        # ====================================================

        with st.spinner(
            "🧠 Creating Gemini embeddings..."
        ):

            embeddings = GeminiEmbeddings()


        # ====================================================
        # CREATE FAISS DATABASE
        # ====================================================

        with st.spinner(
            "🔎 Building FAISS vector database..."
        ):

            vector_store = FAISS.from_documents(
                docs,
                embeddings
            )

            retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": 4
                }
            )


        st.success(
            "✅ Document is ready!"
        )


        # ====================================================
        # QUESTION INPUT
        # ====================================================

        st.divider()

        question = st.text_input(
            "💬 Ask a question about the document:"
        )


        if question:

            with st.spinner(
                "🤖 Gemini is thinking..."
            ):

                # ============================================
                # RETRIEVE DOCUMENTS
                # ============================================

                relevant_docs = retriever.invoke(
                    question
                )


                # ============================================
                # BUILD CONTEXT
                # ============================================

                context = "\n\n".join(
                    doc.page_content
                    for doc in relevant_docs
                )


                # ============================================
                # PROMPT
                # ============================================

                prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the
information provided in the document context.

Do not use outside knowledge.

If the answer is not present in the context,
say exactly:

"I couldn't find the answer in the document."

Be clear and concise.

================ DOCUMENT CONTEXT ================

{context}

================ END CONTEXT ======================

QUESTION:

{question}

ANSWER:
"""


                # ============================================
                # GEMINI GENERATION
                # ============================================

                response = client.models.generate_content(
                    model=LLM_MODEL,
                    contents=prompt
                )

                answer = response.text


            # =================================================
            # DISPLAY ANSWER
            # =================================================

            st.subheader("🤖 Answer")

            st.write(answer)


            # =================================================
            # SOURCES
            # =================================================

            with st.expander("📚 View Sources"):

                for i, doc in enumerate(
                    relevant_docs,
                    start=1
                ):

                    page_number = (
                        doc.metadata.get(
                            "page",
                            0
                        ) + 1
                    )

                    st.write(
                        f"### Source {i}"
                    )

                    st.write(
                        f"📄 Page: {page_number}"
                    )

                    st.write(
                        doc.page_content[:700]
                    )

                    st.divider()


    except Exception as e:

        st.error(
            "❌ Something went wrong."
        )

        st.exception(e)


    finally:

        # ====================================================
        # DELETE TEMP FILE
        # ====================================================

        if os.path.exists(temp_file):
            os.remove(temp_file)

