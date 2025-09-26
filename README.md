# 📘 Public Policy Navigation Using AI

An AI-powered system designed to simplify navigation and understanding of **public policy documents**.  
The project extracts text from lengthy policy PDFs, summarizes them, and allows users to ask **natural language questions** for quick answers.  

---

## 🚀 Features
- **OCR-based text extraction** from PDF policy documents.  
- **Summarization** of large and complex policy sections.  
- **Question Answering (Q&A)** using NLP models.  
- **JSON-based storage** for structured and reusable data.  
- Simple **CLI/GUI interface** for users.  

---

## 🎯 Problem Statement
Public policies are often lengthy, unstructured, and filled with technical or legal jargon.  
Traditional keyword-based search fails to provide **context-aware answers**.  
This project addresses the gap by using **AI and NLP** to make policies more accessible.  

---

## 📌 Objectives
- Process and structure public policy documents.  
- Enable **AI-driven summarization** of large texts.  
- Support **natural language queries** for retrieving policy details.  
- Provide a user-friendly interface for researchers, students, and citizens.  

---

## 🛠️ System Architecture
```mermaid
flowchart TD
    A[Policy Document (PDF/Text)] --> B[OCR & Preprocessing]
    B --> C[NLP Model: Summarization + Q&A]
    C --> D[JSON Storage]
    D --> E[User Interface / CLI]
🗂️ Methodology

Data Collection & Preprocessing – Gather policy documents, convert PDFs to text (OCR), and clean.

AI/NLP Integration – Use transformer-based models for summarization and Q&A.

Storage Layer – Store structured data in JSON format for quick retrieval.

Interface – Provide a simple Python-based CLI/GUI for interaction.

🧰 Tools & Technologies

Programming Language: Python

Libraries:

transformers, spacy, nltk (for NLP)

PyPDF2, pytesseract (for OCR and text extraction)

pandas (data handling)

Storage: JSON

Version Control: GitHub

📊 Results

Successfully processed lengthy policies into structured JSON format.

Generated short summaries for complex sections.

Answered context-aware queries (beyond simple keyword search).

🔮 Future Enhancements

Voice-enabled queries.

Multi-language policy support.

Web-based or chatbot interface.

Integration with government policy portals.

📚 References

Vaswani et al., “Attention is All You Need”, NeurIPS 2017.

Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers”, NAACL 2019.

Hugging Face Transformers: https://huggingface.co/transformers

SpaCy Documentation: https://spacy.io

Tesseract OCR: https://github.com/tesseract-ocr/tesseract

Government of India Policy Portal: https://www.mygov.in/policy

Government of Canada Policy Portal: https://www.canada.ca/en/policy.html

👨‍💻 Author

Rajesh Polipalli

B.Tech, Electronics and Communication Engineering (ECE)

Aspiring AI & IoT Engineer | Full Stack Developer

GitHub: rajeshpolipalli
