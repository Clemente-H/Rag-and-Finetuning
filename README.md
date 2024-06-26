# Rag-and-Finetuning
This repository contains the code and tools needed to evaluate and compare large language models (LLM) using the RAG architecture and finetuning to verify facts about Chilean reality in the period 2019-2023.

## Repository Structure
- **Data**: Contains evaluation datasets and experiment results.
- **Eval**: Codes to evaluate the RAG models, finetuning (still in development), and the base model.
- **Extra**: Useful codes and examples.
- **Milvus**: Codes for vector database connections, data loading, and conversion of data into embeddings.
- **Streamlit**: Web app to visualize the chatbot with RAG.

## Models Used
- **Language Model**: `mistralai/Mixtral-8x7B-Instruct-v0.1` used for research.
- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` with dimensions of 384 and cosine metric. Embeddings are stored in Milvus.


## Evaluation Prompt
```plaintext
"You are an AI assistant specialized in fact-checking about the Chilean reality in the year 2023. Your duty is to verify the truthfulness of the statements presented related to events, situations, or data about Chile during the year 2023, and respond whether they are true or false concisely, without providing additional explanations. If you do not have enough information in your knowledge base to determine the veracity of a statement about the Chilean reality in 2023, you must honestly respond that you do not know. Your responses must always be in Spanish.
Respond only with 'True', 'False' or 'I do not know' to the following statement about the Chilean reality in 2023:"
```

## Installation
To install and run this project, the following dependencies need to be installed:
- milvus
- pymilvus
- langchain
- python-dotenv
- openai
- sentence-transformers
- langchainhub
- langchain-together
- python-dotenv
