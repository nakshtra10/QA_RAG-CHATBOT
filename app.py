import os
import streamlit as st
from io import BytesIO
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import cohere

# Initialize models and clients
model = SentenceTransformer('all-MiniLM-L6-v2')
co = cohere.Client('MmasEiUnHXjbS7pMQppxjpjfRJH7l0gnRqkSEQdb')  # Replace with your actual Cohere API key

# Function to process PDF and store embeddings
def process_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf = PdfReader(f)
        document = ''
        for page in range(len(pdf.pages)):  # Updated to use len(pdf.pages)
            document += pdf.pages[page].extract_text()

    # Split the document into segments (e.g., sentences)
    segments = document.split('. ')

    # Generate embeddings for each segment
    embeddings = model.encode(segments)

    # Store embeddings in FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, segments

# Function to query the database
def query_database(index, segments, question):
    question_embedding = model.encode([question])
    D, I = index.search(question_embedding, k=5)  # Retrieve top 5 segments
    relevant_segments = [segments[i] for i in I[0]]
    return relevant_segments

# Function to generate a response
def generate_response(question, context_segments):
    response = co.generate(
        model='command-light',  # Model suitable for free-tier API key
        prompt=f"Question: {question}\nContext: {' '.join(context_segments)}\nAnswer:",
        max_tokens=200
    )
    return response.generations[0].text

# Initialize Streamlit app
st.title('QA Bot with Optional PDF Upload')

# Chat context
chat_context = []

# Checkbox to enable or disable PDF processing
use_pdf = st.checkbox('Use PDF for context (optional)')

# Upload PDF section
if use_pdf:
    uploaded_file = st.file_uploader('Upload a PDF document', type='pdf')
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_file_path = 'tmp/uploaded_file.pdf'
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Process the PDF and store embeddings
        index, segments = process_pdf(temp_file_path)
        st.success('PDF uploaded and processed successfully!')

# Ask a question and chat
question = st.text_input('Ask a question')

if st.button('Submit'):
    # Use PDF context if available
    if use_pdf and uploaded_file is not None:
        relevant_segments = query_database(index, segments, question)
    else:
        relevant_segments = []

    # Add chat history as additional context
    chat_context.append(f"Question: {question}")
    if relevant_segments:
        chat_context.append(f"Relevant Segments: {' '.join(relevant_segments)}")

    response = generate_response(question, chat_context)
    st.write('Answer:', response)

    # Save the conversation context
    chat_context.append(f"Answer: {response}")
