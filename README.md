# PDF Chatbot with LangChain

A PDF chatbot that uses LangChain and Hugging Face models to answer questions about uploaded documents with high accuracy and no repetitive responses.

## Features

- **PDF Upload & Processing**: Extract and analyze text from any PDF document
- **Accurate Question Answering**: Uses DistilBERT Q&A model for precise answers
- **Document Summarization**: BART model generates intelligent summaries  
- **Vector Search**: FAISS-powered semantic search for relevant content
- **Chat Interface**: Modern chat experience with message history
- **Confidence Scoring**: Shows reliability of each answer
- **No Repetitive Responses**: Extractive Q&A prevents text repetition
- **Beautiful UI**: Clean interface with gradient designs

## Installation & Setup

### Quick Setup

**For Windows:**
```bash
# Double-click to install and run:
run_simple_only.bat
```

**For Linux/Mac:**
```bash
# Install dependencies:
pip install -r requirements_simple_only.txt

# Download NLTK data:
python download_nltk_data.py

# Run the app:
streamlit run simple_app.py
```

### Manual Setup

1. **Install packages**:
   ```bash
   pip install -r requirements_simple_only.txt
   ```

2. **Download NLTK data**:
   ```bash
   python download_nltk_data.py
   ```

3. **Run application**:
   ```bash
   streamlit run simple_app.py
   ```

4. **Open browser** to `http://localhost:8501`

## How to Use

1. **Upload PDF**: Choose a PDF file to analyze
2. **Process**: Click "Process PDF" to extract content
3. **Get Summary**: Use "Get Document Summary" for overview
4. **Ask Questions**: Chat about specific topics in your document
5. **Review Answers**: Check confidence scores and ask follow-ups

## Technical Stack

- **Streamlit**: Web interface and chat functionality
- **LangChain**: Document processing and retrieval chains
- **Hugging Face Models**:
  - DistilBERT for question answering
  - BART for document summarization
  - Sentence-transformers for embeddings
- **FAISS**: Vector similarity search
- **PyPDF**: PDF text extraction

## Answer Quality Features

### Improved Accuracy
- **Extractive Q&A**: Finds specific answers in your document
- **Multi-chunk analysis**: Checks multiple document sections
- **Confidence scoring**: Shows answer reliability (0.0-1.0)
- **Smart fallbacks**: Uses summarization when Q&A fails

### No Repetition Issues
- Uses dedicated Q&A models instead of text generation
- Prevents the repetitive responses common in generative models
- Validates answer quality before displaying

## Example Questions

- "What is the main topic of this document?"
- "Who are the authors mentioned?"
- "What methodology was used?"
- "What are the key findings?"
- "Can you explain [specific concept]?"

## Requirements

```
streamlit>=1.28.0
langchain>=0.0.350
langchain-community>=0.0.10
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
transformers>=4.35.0
torch>=2.1.0
pypdf>=3.17.0
tiktoken>=0.5.0
```

## Important Notes

### First Run Setup
- **Model downloads**: 1-2GB of models download automatically on first use
- **Processing time**: Initial setup takes 2-3 minutes
- **Internet required**: For downloading models (subsequent runs work offline)

### Performance Expectations
- **Document processing**: 1-2 minutes depending on PDF size
- **Answer generation**: 3-10 seconds per question
- **Memory usage**: Requires 2-4GB RAM for optimal performance

### Limitations
- **Document scope**: Answers limited to uploaded PDF content
- **Processing time**: Large PDFs (50+ pages) may be slow
- **Model accuracy**: Depends on document quality and question clarity
- **Context limits**: Very long documents may need sectioning

## File Structure

```
pdf-chatbot-langchain/
├── simple_app.py                    # Main application
├── requirements_simple_only.txt     # Dependencies
├── download_nltk_data.py            # NLTK setup script
├── run_simple_only.bat             # Windows launcher
└── README.md                       # Documentation
```

## Use Cases

- Research paper analysis
- Educational content study
- Report summarization

## Troubleshooting

### Common Issues
- **Slow first run**: Models downloading (normal)
- **Memory errors**: Close other applications or use smaller PDFs
- **Import errors**: Run `pip install -r requirements_simple_only.txt`
- **PDF processing fails**: Check PDF isn't password-protected

### Performance Tips
- Use PDFs under 50 pages for best speed
- Ensure 4GB+ available RAM
- Close memory-intensive applications
- Restart app if responses become slow

Built with modern NLP techniques for reliable document interaction.
