# 📚 Simple PDF Chat Bot

A PDF chatbot powered by LangChain and Hugging Face models that extracts text from PDFs, generates summaries, and provides intelligent answers.

## ✨ Features

- 📤 **PDF Upload**: Upload any PDF file for analysis
- 📝 **Smart Summaries**: Get document summaries instantly
- 💬 **Interactive Q&A**: Ask questions and get intelligent answers
- 🎨 **Beautiful UI**: Modern, colorful interface with gradient designs
- 🔍 **Vector Search**: FAISS vector store for accurate document retrieval
- 🤖 **AI-Powered**: Uses Hugging Face models for intelligent responses

## 🛠️ Installation & Setup

### Quick Setup

**For Windows:**
```bash
# Double-click this file to install and run:
run_simple_only.bat
```

**For Linux/Mac:**
```bash
# Install packages:
pip install -r requirements_simple_only.txt

# Download NLTK data:
python download_nltk_data.py

# Run the app:
streamlit run simple_app.py
```

### Manual Setup

1. **Install required packages**:
   ```bash
   pip install -r requirements_simple_only.txt
   ```

2. **Download NLTK data**:
   ```bash
   python download_nltk_data.py
   ```

3. **Run the application**:
   ```bash
   streamlit run simple_app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 🚀 How to Use

1. **Upload PDF**: Choose any PDF file to analyze
2. **Process**: Click "Process PDF" to extract content
3. **Get Summary**: Click "Get Summary" for document overview
4. **Ask Questions**: Type questions about your PDF content
5. **Get Answers**: Receive intelligent responses based on your document

## 📋 Technical Stack

- **Streamlit**: Web interface
- **PyPDF2**: PDF text extraction
- **LangChain**: Document processing and Q&A chains
- **Hugging Face Transformers**: AI models for text generation
- **FAISS**: Vector similarity search
- **NLTK**: Natural language processing

## 📝 Example Questions

- "What is this document about?"
- "Give me a summary"
- "What are the main points?"
- "What does it say about [specific topic]?"

## 🎯 Use Cases

- Document analysis and research
- PDF content exploration
- Educational material review
- Quick document insights

## 📋 Requirements

See `requirements_simple_only.txt` for the full list of dependencies.

## 🚨 Notes

- First run may take time to download AI models
- Processing time depends on PDF size
- Requires internet connection for model downloads
- Some Hugging Face models may have usage limitations

## 🎉 Enjoy Your PDF Chat Bot!

Upload your PDFs and start chatting with your documents!
