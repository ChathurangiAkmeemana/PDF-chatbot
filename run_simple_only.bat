@echo off
echo Starting Simple PDF Chat Bot...
echo.

REM Install required packages
echo Installing required packages...
py -m pip install --user streamlit PyPDF2 pypdf nltk langchain langchain-community sentence-transformers requests faiss-cpu transformers torch accelerate

REM Download NLTK data
echo.
echo Downloading NLTK data...
py download_nltk_data.py

REM Run the simple application
echo.
echo Starting the application...
echo Open your browser to: http://localhost:8501
echo.
py -m streamlit run simple_app.py

pause
