import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Configure Streamlit page
st.set_page_config(
    page_title="PDF Chatbot with LangChain",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
}

.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}

.user-message {
    background-color: #e3f2fd;
    margin-left: 20%;
}

.bot-message {
    background-color: #f5f5f5;
    margin-right: 20%;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

@st.cache_resource
def load_qa_model():
    """Load and cache the question-answering model"""
    try:
        model_name = "google/flan-t5-small"  # Better for Q&A tasks
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Create pipeline for text generation
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=1024,
            temperature=0.3,
            do_sample=True,
            device=-1  # CPU
        )
        
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def load_and_process_pdf(uploaded_file):
    """Load PDF and create vector store"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load PDF using LangChain
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        
        # Split text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks
            chunk_overlap=100,
            length_function=len
        )
        docs = text_splitter.split_documents(pages)
        
        # Create embeddings and vector store using Hugging Face
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return vectorstore, len(pages), len(docs)
        
    except Exception as e:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        raise e

def answer_question_directly(question, vectorstore, qa_pipeline):
    """Answer questions directly without RetrievalQA chain"""
    try:
        # Get relevant documents
        relevant_docs = vectorstore.similarity_search(question, k=2)
        
        if not relevant_docs:
            return "I couldn't find relevant information in the document to answer your question."
        
        # Combine relevant text
        context = " ".join([doc.page_content for doc in relevant_docs])
        
        # Truncate context if too long
        max_context_length = 400
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        # Create a proper prompt for the model
        prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer based on the context:"
        
        # Generate answer
        response = qa_pipeline(prompt, max_length=150, min_length=10)
        
        if response and len(response) > 0:
            answer = response[0]['generated_text']
            
            # Clean up the answer - remove the prompt part
            if "Answer based on the context:" in answer:
                answer = answer.split("Answer based on the context:")[-1].strip()
            
            return answer if answer else "I couldn't generate a clear answer from the document."
        else:
            return "I couldn't generate an answer. Please try rephrasing your question."
            
    except Exception as e:
        return f"Error processing question: {str(e)}"

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 PDF Chatbot with LangChain</h1>
        <p>Upload a PDF and ask questions about its content</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Load QA model
    with st.spinner("Loading AI model..."):
        qa_pipeline = load_qa_model()
    
    if qa_pipeline is None:
        st.error("Failed to load the question-answering model. Please refresh the page.")
        return
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Configuration")
        
        # File upload
        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF file to chat with"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            if not st.session_state.pdf_processed:
                if st.button("Process PDF", type="primary"):
                    with st.spinner("Processing PDF... This may take a moment."):
                        try:
                            vectorstore, num_pages, num_chunks = load_and_process_pdf(uploaded_file)
                            
                            st.session_state.vectorstore = vectorstore
                            st.session_state.pdf_processed = True
                            
                            st.success(f"✅ PDF processed successfully!")
                            st.info(f"📄 Pages: {num_pages} | Chunks: {num_chunks}")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error processing PDF: {str(e)}")
            else:
                st.success("✅ PDF is ready for questions!")
                if st.button("Upload New PDF"):
                    st.session_state.pdf_processed = False
                    st.session_state.vectorstore = None
                    st.session_state.messages = []
                    st.rerun()
    
    # Main chat interface
    if st.session_state.pdf_processed and st.session_state.vectorstore is not None:
        st.subheader("💬 Chat with your PDF")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your PDF"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document..."):
                    try:
                        # Get answer using direct method
                        answer = answer_question_directly(
                            prompt, 
                            st.session_state.vectorstore, 
                            qa_pipeline
                        )
                        
                        # Display answer
                        st.write(answer)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        error_message = f"Error generating response: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
            
    else:
        # Instructions when no PDF is loaded
        st.info("👈 Please upload and process a PDF file to start chatting!")
        
        with st.expander("📋 How to use this chatbot"):
            st.markdown("""
            1. **Upload PDF**: Choose a PDF file from your computer
            2. **Process PDF**: Click "Process PDF" to analyze the document
            3. **Ask Questions**: Start chatting about the PDF content!
            
            **Example questions:**
            - "What is this document about?"
            - "Can you summarize the main points?"
            - "What does it say about [specific topic]?"
            - "What are the key conclusions?"
            
            **Note**: This app uses free Hugging Face models.
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with LangChain, Hugging Face, and Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()