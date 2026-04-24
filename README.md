# 🧠 MedVerify — Medical Hallucination Detection System

> An intelligent multi-agent NLP system that detects and corrects hallucinations in AI-generated medical text using fine-tuned biomedical models and real-time evidence retrieval.



---

## 🚨 The Problem

Large language models frequently generate **medically inaccurate statements** — a phenomenon called *hallucination*. In healthcare contexts, this is dangerous.

> **Example:** An LLM might claim *"Corona is not a virus"* — factually wrong, and potentially harmful if trusted without verification.

MedVerify catches these errors, explains why they're wrong, and provides corrected, evidence-backed responses.

---

## ✨ Features

-  **Hallucination Detection** — Classifies medical claims as hallucinated or factual with 94.3% accuracy
-  **Multi-Agent Pipeline** — 9 specialized agents handle detection, retrieval, reranking, and correction
-  **Real-Time Evidence Retrieval** — Pulls from PubMed, Semantic Scholar, and Europe PMC
   **Automated Correction** — Generates corrected statements using Groq LLaMA 3.3 70B
-  **Full-Stack Interface** — React frontend + Flask API backend

---



## 🏗️ System Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────┐
│           React Frontend                │
└────────────────┬────────────────────────┘
                 │ REST API
                 ▼
┌─────────────────────────────────────────┐
│           Flask API (api.py)            │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴─────────┐
        ▼                  ▼
┌──────────────┐   ┌──────────────────────┐
│  BiomedBERT  │   │   Evidence Retrieval  │
│  Classifier  │   │  PubMed / Semantic   │
│  (94.3% acc) │   │  Scholar / Europe PMC│
└──────────────┘   └──────────────────────┘
        │                  │
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │  Groq LLaMA 3.3  │
        │  70B Correction  │
        └──────────────────┘
```

---

## 🤖 The 9-Agent Pipeline

| Agent | Role |
|---|---|
| **Input Validator** | Cleans and preprocesses the medical claim |
| **Hallucination Detector** | BiomedBERT classifier — hallucinated or not |
| **Query Formulator** | Converts claim into retrieval queries |
| **PubMed Retriever** | Fetches evidence from PubMed |
| **Semantic Scholar Retriever** | Fetches from Semantic Scholar |
| **Europe PMC Retriever** | Fetches from Europe PMC |
| **BM25 Reranker** | Ranks retrieved evidence by relevance |
| **Correction Generator** | Uses Groq LLaMA 3.3 70B to generate corrected claim |
| **Response Assembler** | Combines all results into final structured output |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | **94.3%** |
| Macro F1 | **0.871** |
| Base Model | BiomedBERT |
| Dataset | MedHallu |

---

## 🛠️ Tech Stack

**Backend**
- Python, Flask
- HuggingFace Transformers (BiomedBERT)
- Groq API (LLaMA 3.3 70B Versatile)
- BM25 (rank-bm25)
- PubMed API, Semantic Scholar API, Europe PMC API

**Frontend**
- React.js
- Axios

---

## 🚀 Running Locally

### Prerequisites
- Python 3.9+
- Node.js 16+
- Groq API Key ([get one here](https://console.groq.com))

### 1. Clone the repo
```bash
git clone https://github.com/tanspan/medverify.git
cd medverify
```

### 2. Set up the backend
```bash
pip install -r requirements.txt
```

Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Start the Flask API:
```bash
python api.py
```

### 3. Set up the frontend
```bash
cd frontend
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 📁 Project Structure

```
medverify/
├── api.py                  # Flask backend & agent orchestration
├── pipeline.py             # Core multi-agent pipeline
├── requirements.txt        # Python dependencies
├── saved_model/            # Fine-tuned BiomedBERT weights
├── training/               # Model training scripts
├── vision/                 # (vision module)
└── frontend/
    ├── public/
    └── src/
        ├── App.js          # Main React component
        └── index.js
```

---

## 📌 Important Notes

- The `saved_model/` folder contains fine-tuned weights (~400MB). These are **not included** in the repo due to size. Download separately or retrain using scripts in `training/`.
- Never commit your `.env` file — it contains your Groq API key.

---

##Demo
<img width="1855" height="527" alt="Screenshot 2026-04-24 133348" src="https://github.com/user-attachments/assets/de11a288-78b1-4593-a382-fc851c309f01" />
<img width="1219" height="752" alt="Screenshot 2026-04-24 133339" src="https://github.com/user-attachments/assets/95ca1358-3839-4787-9a03-8c5b0572d212" />
<img width="1140" height="788" alt="Screenshot 2026-04-24 133358" src="https://github.com/user-attachments/assets/7e71c66a-4bbb-4f28-9a19-cccd7b8aa84b" />
<img width="1102" height="367" alt="Screenshot 2026-04-24 133405" src="https://github.com/user-attachments/assets/d11329dd-8a9b-4096-b844-a423cf1cdcc4" />
