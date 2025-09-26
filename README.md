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

🛠️ System Architecture

Workflow:
Policy Document (PDF/Text) → OCR & Preprocessing → NLP Model (Summarization + Q&A) → JSON Storage → User Interface (CLI/GUI)

### 🗂️ Methodology
- Data Collection & Preprocessing
- Gather policy documents
- Convert PDFs to text (OCR)
- Clean and tokenize
- AI/NLP Integration
- Use transformer-based models for summarization and Q&A
- Storage Layer
- Store structured data in JSON format for quick retrieval Interface
- Provide a Python-based CLI/GUI for user interaction

### 🧰 Tools & Technologies
- Programming Language: Python
- Libraries: transformers, spaCy, NLTK (for NLP)
   PyPDF2, pytesseract (for OCR & text extraction)
  pandas (data handling)
- Storage: JSON
- Version Control: GitHub

### 📊 Results
- Processed lengthy policies into structured JSON format
- Generated concise summaries for complex sections
- Answered context-aware queries (beyond keyword search)

#### 🔮 Future Enhancements
- Voice-enabled queries
- Multi-language support
- Web-based or chatbot interface
- Integration with government policy portals
