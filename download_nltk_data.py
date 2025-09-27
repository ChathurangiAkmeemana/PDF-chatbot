#!/usr/bin/env python3
"""
Script to download required NLTK data for the PDF Chat Bot
"""

import nltk
import sys

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data for PDF Chat Bot...")
    
    try:
        print("Downloading punkt_tab...")
        nltk.download('punkt_tab', quiet=False)
        print("✅ punkt_tab downloaded successfully")
        
        print("Downloading stopwords...")
        nltk.download('stopwords', quiet=False)
        print("✅ stopwords downloaded successfully")
        
        print("\n🎉 All NLTK data downloaded successfully!")
        print("You can now run the PDF Chat Bot without issues.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

if __name__ == "__main__":
    success = download_nltk_data()
    sys.exit(0 if success else 1)
