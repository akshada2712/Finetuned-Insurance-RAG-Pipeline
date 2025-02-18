# Fine-Tuned Insurance Company Support Bot

## Overview
This project aims to develop a Retrieval-Augmented Generation (RAG) pipeline to create an AI-powered customer support bot for Niva Bupa Insurance. The bot answers customer queries in the style of an insurance support agent while ensuring responses align with official policy documents.

## Problem Statement
Traditional customer support systems often struggle with inefficiencies, requiring manual intervention to fetch relevant information. Customers may face delays in obtaining responses or receive inaccurate answers. This project addresses these issues by leveraging web scraping, fine-tuned language models, and vectorized retrieval techniques to deliver quick and accurate insurance-related responses.

## Project Structure
```
Fine-Tuned Insurance Company Support Bot/
│── niva_bupa_pdfs/                  # Extracted PDFs
│── niva_bupa_vectorstore/           # Chroma vector store
│── app.py                            # Main application script
│── chunk_vectorize.ipynb             # Chunking and vectorization notebook
│── crawler.ipynb                      # Web scraping notebook
│── finetune_model.ipynb              # Fine-tuning OpenAI model
│── insurance_customer_support_conversation.csv  # Training dataset
│── insurance_finetune_clean.jsonl    # Processed clean data
│── insurance_finetune_train.jsonl    # Training data
│── insurance_finetune_val.jsonl      # Validation data
│── validated_data.jsonl              # Final validated dataset
```

## Features
- **Automated Web Scraping**: Extracts insurance-related PDFs from the Niva Bupa website.
- **Document Categorization**: Classifies documents into application forms, brochures, claim forms, and policy wordings.
- **Chunking and Vectorization**: Implements document segmentation strategies tailored to different document types.
- **Fine-Tuned LLM**: Enhances OpenAI models for domain-specific customer support.
- **RAG-Based Retrieval**: Ensures accurate responses through embedding-based similarity search using ChromaDB.
- **Scalable API**: Provides an interface for querying the system in a conversational manner.

## Technology Stack
- **Backend**: Python, FastAPI
- **Web Scraping**: BeautifulSoup, Selenium
- **Machine Learning**: OpenAI API, Llama3.1, LangChain
- **Vector Database**: ChromaDB
- **Data Processing**: Pandas, NumPy, PyPDFLoader
- **Storage**: JSONL files for dataset management

## Backend Modules
### 1. **Web Scraping Module**
- Extracts and categorizes PDFs from the Niva Bupa website.
- Implements automation for periodic updates.

### 2. **Document Processing Module**
- Reads PDFs using PyPDFLoader.
- Applies chunking strategies based on document type:
  ```python
  def chunk_strategy(file_path):
      file_name = os.path.basename(file_path).lower()
      if 'policy' in file_name:
          return {'chunk_size': 1500, 'chunk_overlap': 300}
      elif 'form' in file_name:
          return {'chunk_size': 500, 'chunk_overlap': 100}
      elif 'brochure' in file_name:
          return {'chunk_size': 1000, 'chunk_overlap': 200}
      else:
          return {'chunk_size': 800, 'chunk_overlap': 150}
  ```

### 3. **Fine-Tuning Module**
- Processes and cleans customer support data from Hugging Face dataset.
- Trains the model for improved domain-specific response generation.

### 4. **Vectorization and Retrieval Module**
- Generates OpenAI embeddings for chunked documents.
- Stores embeddings in ChromaDB.
- Performs similarity searches for customer queries.

### 5. **API and Query Handling Module**
- Accepts user queries.
- Retrieves relevant document chunks from the vector store.
- Generates LLM-based responses.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/akshada2712/Finetuned-Insurance-RAG-Pipeline.git
   cd Finetuned-Insurance-RAG-Pipeline
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```



## Usage
- **User queries are processed via the API**.
- **A similarity search is conducted on the vector store** to find relevant document chunks.
- **A response is generated** based on the retrieved chunks and the fine-tuned LLM model.

## Future Enhancements
- **Improve chunking strategies** based on document semantics.
- **Optimize retrieval accuracy** with advanced embedding techniques.
- **Enhance UI/UX** for seamless interaction.
- **Integrate multilingual support** for broader accessibility.

This project establishes a scalable RAG pipeline for insurance-related document retrieval and response generation.


