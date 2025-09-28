import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
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
        # Use a dedicated question-answering model instead of text generation
        qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            tokenizer="distilbert-base-cased-distilled-squad",
            device=-1  # CPU
        )
        
        return qa_pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_summarization_model():
    """Load summarization model for fallback"""
    try:
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1
        )
        return summarizer
    except Exception as e:
        st.warning(f"Summarization model not available: {e}")
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
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
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

def answer_question_with_qa_model(question, vectorstore, qa_pipeline, summarizer=None):
    """Answer questions using dedicated Q&A model"""
    try:
        # Get relevant documents
        relevant_docs = vectorstore.similarity_search(question, k=3)
        
        if not relevant_docs:
            return "I couldn't find relevant information in the document to answer your question."
        
        # Try Q&A with each relevant chunk and find best answer
        best_answer = ""
        best_score = 0
        
        for doc in relevant_docs:
            context = doc.page_content
            
            # Skip very short contexts
            if len(context.strip()) < 50:
                continue
                
            try:
                # Limit context length for Q&A model
                if len(context) > 1000:
                    context = context[:1000]
                
                result = qa_pipeline(question=question, context=context)
                
                # Keep track of best answer based on confidence score
                if result['score'] > best_score and len(result['answer'].strip()) > 10:
                    best_answer = result['answer']
                    best_score = result['score']
                    
            except Exception as e:
                continue
        
        # If we found a good answer, return it
        if best_score > 0.1 and best_answer:
            confidence_text = f" (Confidence: {best_score:.2f})" if best_score < 0.8 else ""
            return f"{best_answer}{confidence_text}"
        
        # Fallback: Try to provide a summary of relevant content
        combined_context = " ".join([doc.page_content for doc in relevant_docs[:2]])
        
        if summarizer and len(combined_context) > 100:
            try:
                # Use summarization as fallback
                if len(combined_context) > 1000:
                    combined_context = combined_context[:1000]
                    
                summary = summarizer(combined_context, max_length=100, min_length=30, do_sample=False)
                return f"Based on the document content: {summary[0]['summary_text']}"
                
            except Exception as e:
                pass
        
        # Final fallback: return relevant text directly
        if combined_context:
            sentences = combined_context.split('. ')
            relevant_sentences = [s for s in sentences if any(word.lower() in s.lower() 
                                for word in question.split() if len(word) > 3)]
            
            if relevant_sentences:
                return '. '.join(relevant_sentences[:2]) + '.'
        
        return "I found some information but couldn't extract a specific answer. Try asking about specific topics mentioned in the document."
        
    except Exception as e:
        return f"Error processing question: {str(e)}"

def generate_summary(vectorstore, summarizer=None):
    """Generate document summary"""
    try:
        # Get a sample of documents to summarize
        all_docs = vectorstore.similarity_search("", k=10)
        
        if not all_docs:
            return "No content available for summary."
        
        # Combine text from multiple chunks
        combined_text = " ".join([doc.page_content for doc in all_docs[:5]])
        
        if summarizer and len(combined_text) > 200:
            try:
                if len(combined_text) > 1500:
                    combined_text = combined_text[:1500]
                
                summary = summarizer(combined_text, max_length=150, min_length=50, do_sample=False)
                return summary[0]['summary_text']
                
            except Exception as e:
                pass
        
        # Fallback: extract key sentences
        sentences = combined_text.split('. ')
        # Take first few sentences as basic summary
        summary_sentences = sentences[:3]
        return '. '.join(summary_sentences) + '.'
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

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
    
    # Load models
    with st.spinner("Loading AI models..."):
        qa_pipeline = load_qa_model()
        summarizer = load_summarization_model()
    
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
                
                # Make summary button more prominent
                st.markdown("### 📋 Quick Actions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📝 Get Summary", type="secondary", use_container_width=True):
                        with st.spinner("Generating summary..."):
                            summary = generate_summary(st.session_state.vectorstore, summarizer)
                            st.session_state.messages.append({"role": "assistant", "content": f"📄 **Document Summary:**\n\n{summary}"})
                            st.rerun()
                
                with col2:
                    if st.button("🗑️ New PDF", type="secondary", use_container_width=True):
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
                        # Get answer using Q&A model
                        answer = answer_question_with_qa_model(
                            prompt, 
                            st.session_state.vectorstore, 
                            qa_pipeline,
                            summarizer
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
            3. **Get Summary**: Use the "Get Document Summary" button for an overview
            4. **Ask Questions**: Start chatting about the PDF content!
            
            **Example questions:**
            - "What is this document about?"
            - "Who are the main authors?"
            - "What methodology was used?"
            - "What are the key findings?"
            - "What are the conclusions?"
            
            **Features:**
            - Uses DistilBERT for accurate question answering
            - Provides confidence scores for answers
            - Includes document summarization
            - No repetitive responses
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
