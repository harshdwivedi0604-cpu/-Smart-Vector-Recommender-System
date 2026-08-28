# 🎬 Smart Vector Recommender System (v1.0)

A lightweight, high-performance content recommendation engine built in Python. The system utilizes **NumPy Cosine Similarity** to compare item feature vectors fairly regardless of scale, and offers a **Streamlit Web Dashboard** alongside multi-source data parsing capabilities (**CSV**, **PDF**, and **Web APIs**).

---

## 📌 Features

- **Mathematical Fairness**: Uses **Cosine Similarity** ($\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\Vert{}\mathbf{A}\Vert{} \Vert{}\mathbf{B}\Vert{}}$) instead of raw dot products to prevent size bias when calculating match scores.
- **Interactive Web UI**: Built with **Streamlit** for real-time item selection, customizable result sizing, percentage progress bars, and feature matrix previews.
- **Multi-Source Data Ingestion**:
  - **Local CSV Uploads**: Automated parsing using **Pandas**, complete with null-value handling (`fillna(0)`).
  - **Local PDF Parsing**: Extracts titles and feature vectors from PDF documents using **pdfplumber**.
  - **Remote URLs / APIs**: Fetches remote JSON API payloads or raw web-hosted CSV files via **requests**.
- **Case-Insensitive & Flexible Search**: Handles arbitrary capitalization and whitespace trimming smoothly.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your environment.

### 2. Install Required Dependencies
Clone the repository (or copy your files) and install the dependencies:

```bash
pip install numpy pandas pdfplumber requests streamlit reportlab