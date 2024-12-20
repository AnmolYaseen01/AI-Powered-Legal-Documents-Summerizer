# **AI-Powered Legal Document Summarizer**

## **Overview**
The AI-Powered Legal Document Summarizer is a Streamlit-based application designed to help users process legal documents by:
- Extracting text from scanned images or PDF files.
- Translating text into English, if necessary.
- Identifying and highlighting significant legal clauses.
- Summarizing the content for better readability and quicker understanding.
- Detecting named entities using Natural Language Processing (NLP).

## **Features**
1. **Document Text Extraction**:
   - Supports image formats (JPG, PNG) using Tesseract OCR.
   - Handles PDF files using the `pdfminer` library.

2. **Translation**:
   - Automatically translates text into English using the Hugging Face translation model (`Helsinki-NLP/opus-mt-ROMANCE-en`).
   - Supports multiple source languages: English, French, Spanish, German, and Italian.

3. **Legal Clause Detection**:
   - Detects key clauses such as confidentiality, liability, termination, payment terms, and GDPR compliance.

4. **Summarization**:
   - Summarizes extracted text using Hugging Face's `facebook/bart-large-cnn` summarization model.
   - Allows users to customize the summary length.

5. **Named Entity Recognition (NER)**:
   - Identifies and categorizes entities (e.g., names, dates, locations) using SpaCy's `en_core_web_sm` NLP model.

6. **GPU Support**:
   - Optimized to leverage GPU acceleration when available.

## **Technologies Used**
- **Programming Language**: Python
- **Framework**: Streamlit (for user interface)
- **OCR**: Tesseract
- **PDF Parsing**: pdfminer
- **NLP**: SpaCy, Hugging Face Transformers
- **Translation**: MarianMTModel and MarianTokenizer
- **Summarization**: Facebook BART model
- **Cloud Services**: Hugging Face Hub for model hosting and token authentication

## **Setup Instructions**
1. **Prerequisites**:
   - Python 3.8 or higher
   - Installed dependencies: `streamlit`, `pytesseract`, `pdfminer.six`, `transformers`, `torch`, `spacy`, `Pillow`, `huggingface_hub`

2. **Tesseract Installation**:
   - Download and install Tesseract OCR from [here](https://github.com/tesseract-ocr/tesseract).
   - Update the `pytesseract.pytesseract.tesseract_cmd` variable with the path to the Tesseract executable.

3. **Environment Configuration**:
   - Install Python packages:
     ```bash
     pip install -r requirements.txt
     ```
   - Download SpaCy model:
     ```bash
     python -m spacy download en_core_web_sm
     ```

4. **Hugging Face Hub Authentication**:
   - Obtain a token from Hugging Face.
   - Replace `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` with your token in the code.

5. **Run the Application**:
   - Execute the Streamlit app:
     ```bash
     streamlit run app.py
     ```

## **Workflow**
1. **Upload Document**:
   - User uploads a PDF or image file.
   - The app extracts text using OCR or PDF parsing.

2. **Translation**:
   - If the document is not in English, it is translated automatically.

3. **Clause Detection**:
   - Key legal clauses are identified in the extracted/translated text.

4. **Summarization**:
   - The text is summarized based on the user’s preferred summary length.

5. **Named Entity Recognition**:
   - The app highlights named entities for better context understanding.

## **Challenges and Solutions**
- **Challenge**: Handling low-quality images.
  - **Solution**: Leveraged Tesseract OCR for better text extraction.
- **Challenge**: Handling large documents.
  - **Solution**: Summarization with adjustable length to manage verbosity.
- **Challenge**: Accurate legal clause detection.
  - **Solution**: Created a comprehensive list of clauses for pattern matching.

## **Future Enhancements**
- Add support for additional languages and document types.
- Include functionality for legal clause editing and exporting summarized text.
- Improve OCR accuracy by integrating advanced preprocessing techniques.
- Extend NER to support custom entity types specific to legal documents.

## **Conclusion**
This application serves as a robust tool for legal professionals, researchers, and anyone dealing with lengthy legal documents. By automating text extraction, translation, clause detection, and summarization, it saves significant time and effort.
