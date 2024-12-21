# -*- coding: utf-8 -*-
import os
import streamlit as st
import pytesseract
from PIL import Image
from pdfminer.high_level import extract_text
from transformers import pipeline, MarianMTModel, MarianTokenizer
import spacy
import torch
from huggingface_hub import login, HfApi
import pytesseract

# Specify the path to the Tesseract executable (if not in PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path as needed

# Set TensorFlow environment variable to minimize warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations if causing issues

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Login to Hugging Face Hub
try:
    hf_token = "Your_hf_token_here"  # Replace with your Hugging Face token
    login(hf_token)
    # Validate token
    api = HfApi()
    user_info = api.whoami(token=hf_token)
    st.write(f"Logged in as: {user_info['name']}")
except Exception as e:
    st.error(f"Error logging into Hugging Face Hub: {e}")
    st.stop()

# Initialize models lazily to save time
summarizer = None
translation_model = None
translation_tokenizer = None
nlp = None

# Function to initialize models
def initialize_models():
    global summarizer, translation_model, translation_tokenizer, nlp
    if summarizer is None:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)
    if translation_model is None or translation_tokenizer is None:
        translation_model_name = "Helsinki-NLP/opus-mt-ROMANCE-en"
        translation_model = MarianMTModel.from_pretrained(translation_model_name)
        translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

# Function to extract text from an image
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error extracting text from image: {e}"

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    try:
        text = extract_text(pdf_file)
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

# Function to detect legal clauses
def detect_legal_clauses(text):
    key_clauses = ['termination',
        'indemnity',
        'confidentiality',
        'dispute resolution',
        'liability',
        'force majeure',
        'non-compete',
        'non-disclosure',
        'payment terms',
        'late payment penalty',
        'fee adjustment',
        'refund and cancellation',
        'tax obligations',
        'intellectual property rights',
        'trademark protection',
        'copyright ownership',
        'patent assignment',
        'non-solicitation',
        'code of conduct',
        'workplace confidentiality',
        'limitation of liability',
        'risk allocation',
        'hold harmless',
        'data protection',
        'privacy policy',
        'GDPR compliance',
        'scope of work',
        'delivery timeline',
        'milestone completion',
        'severability',
        'entire agreement',
        'amendment',
        'assignment',
        'waiver',
        'notices',
        'survival',
        'mediation',
        'litigation',
        'dispute escalation process',
        'breach of contract',
        'warranty',
        'performance bond']
    detected_clauses = [clause for clause in key_clauses if clause.lower() in text.lower()]
    return detected_clauses

# Function to translate text to English
def translate_to_english(text, src_lang):
    try:
        if src_lang != "en" and len(text.strip()) > 0:
            initialize_models()  # Ensure models are initialized
            translated = translation_model.generate(
                **translation_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            )
            translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
            return translated_text
        return text
    except Exception as e:
        return f"Error during translation: {e}"

# Streamlit UI setup
def main():
    st.title("AI-Powered Legal Document Summarizer")
    st.markdown("Upload a scanned image (JPG, PNG) or PDF to summarize its content.")

    uploaded_file = st.file_uploader("Upload your legal document (PDF or Image)", type=["pdf", "jpg", "png"])

    if uploaded_file is not None:
        text = ""
        # Extract text based on file type
        if uploaded_file.type in ["image/jpeg", "image/png"]:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            text = extract_text_from_image(image)
        elif uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)

        if not text.strip():
            st.warning("No text could be extracted from the document.")
            return

        # Display extracted text
        st.subheader("Extracted Text:")
        st.write(text)

        # Language selection and translation
        language = st.selectbox("Select document language", ["en", "fr", "es", "de", "it"])
        if language != "en":
            text = translate_to_english(text, language)
            st.subheader("Translated Text:")
            st.write(text)

        # Legal clause detection
        detected_clauses = detect_legal_clauses(text)
        st.subheader("Detected Legal Clauses:")
        st.write(", ".join(detected_clauses) if detected_clauses else "No significant legal clauses detected.")

        # Summary generation
        initialize_models()  # Ensure summarizer is initialized
        summary_length = st.slider("Select summary length", min_value=50, max_value=500, value=200)
        if text.strip():
            try:
                summary = summarizer(text, max_length=summary_length, min_length=50, do_sample=False)
                st.subheader("Summary:")
                st.write(summary[0]["summary_text"])
            except Exception as e:
                st.error(f"Error generating summary: {e}")

        # Named Entity Recognition (NER)
        st.subheader("Named Entities in the Document:")
        initialize_models()  # Ensure NER model is initialized
        doc = nlp(text)
        entities = [(entity.text, entity.label_) for entity in doc.ents]
        st.write(entities if entities else "No named entities detected.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
