# 💼 Local LinkedIn Post Generator using LLM

## 📌 Objective

As part of a Machine Learning internship assignment, the objective was to design and develop a **fully offline, local language model (LLM)** capable of generating **human-like, professional LinkedIn posts** across various themes — without relying on any external APIs like **OpenAI**, **Anthropic**, or **Hugging Face Inference APIs**.

---

## 🧠 Model Used

- ✅ `distilgpt2` – A lightweight, open-source transformer-based model from Hugging Face
- ✅ Trained **locally** on a curated dataset of real LinkedIn-style posts
- ✅ No use of GPT-generated content or external inference

---

## 🛠️ Tools & Libraries

- Python 3.10
- PyTorch
- Hugging Face Transformers & Datasets
- Streamlit (optional UI)
- Trained and runs **fully offline**

---

## 📂 Files Included

- `train.py` – Fine-tunes the distilgpt2 model on the LinkedIn post dataset
- `generate.py` – CLI script for generating posts using the trained model
- `app.py` – Streamlit-based UI for generating posts interactively
- `linkedin_posts.txt` – Custom training dataset
- `requirements.txt` – Required libraries
- `Model_Link.txt` – Drive link to download the full trained model

---

## 🔗 Model Download Link

Due to size limitations, the trained model (approx. 1.18 GB) has been uploaded to Google Drive:

🔗 [Click here to download the model](https://drive.google.com/file/d/1Q2m0YRerVGI_LBD1dsojMoZ8TDJ3ud8Q/view?usp=drive_link)

> To use the model, place the extracted folder as `./linkedin_model` in the root directory.

---

## ✅ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
