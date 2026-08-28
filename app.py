# app.py
import io
import numpy as np
import pandas as pd
import pdfplumber
import requests
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Smart Recommender System v1.0", page_icon="🎬", layout="wide"
)


# -------------------------------------------------------------
# Core Algorithmic Logic
# -------------------------------------------------------------
def calculate_cosine_similarity(vec_a, vec_b):
    """Computes cosine similarity between two numeric arrays using NumPy."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def recommend(chosen, library, how_many=3):
    """Generates top recommendations for a chosen item using cosine similarity."""
    scores = []
    chosen_vec = library[chosen]

    for title, vector in library.items():
        if title == chosen:
            continue
        if len(chosen_vec) != len(vector):
            continue
        s = calculate_cosine_similarity(chosen_vec, vector)
        scores.append((title, s))

    scores.sort(key=lambda pair: pair[1], reverse=True)
    return scores[:how_many]


def process_dataframe(df):
    """Processes a Pandas DataFrame into item names and feature vectors."""
    if df.empty or len(df.columns) < 2:
        st.sidebar.error("Dataset lacks sufficient rows or columns.")
        return {}

    name_col = df.columns[0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        st.sidebar.error("No numeric feature columns found in dataset.")
        return {}

    df[numeric_cols] = df[numeric_cols].fillna(0)

    library = {}
    for _, row in df.iterrows():
        item_name = str(row[name_col]).strip()
        vector = row[numeric_cols].values.astype(float)
        library[item_name] = vector

    return library


# -------------------------------------------------------------
# UI Sidebar Data Source Handlers
# -------------------------------------------------------------
st.sidebar.title("⚙️ Data Settings")

source = st.sidebar.radio(
    "Select Data Source:",
    [
        "Default Dataset",
        "Upload CSV File",
        "Upload PDF File",
        "Fetch from URL / API",
    ],
)

movies = {}

# 1. Default Dataset
if source == "Default Dataset":
    movies = {
        "Toy Story": np.array([5.0, 4.0, 3.0, 2.0]),
        "Die Hard": np.array([5.0, 1.0, 5.0, 2.0]),
        "Deadpool": np.array([4.0, 4.0, 5.0, 1.0]),
        "Inception": np.array([5.0, 1.0, 5.0, 4.0]),
        "The Dark Knight": np.array([5.0, 2.0, 2.0, 5.0]),
        "Superbad": np.array([1.0, 5.0, 0.0, 2.0]),
    }
    st.sidebar.success(f"Loaded built-in dataset ({len(movies)} items).")

# 2. Upload CSV File
elif source == "Upload CSV File":
    uploaded_csv = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            movies = process_dataframe(df)
            st.sidebar.success(f"Imported {len(movies)} items from CSV.")
        except Exception as e:
            st.sidebar.error(f"Failed to parse CSV: {e}")

# 3. Upload PDF File
elif source == "Upload PDF File":
    uploaded_pdf = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_pdf:
        try:
            library = {}
            with pdfplumber.open(uploaded_pdf) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        for line in text.split("\n"):
                            if ":" in line:
                                title, vector_str = line.split(":", 1)
                                vector = [
                                    float(x.strip())
                                    for x in vector_str.split(",")
                                ]
                                library[title.strip()] = np.array(vector)
            movies = library
            st.sidebar.success(f"Imported {len(movies)} items from PDF.")
        except Exception as e:
            st.sidebar.error(f"Failed to parse PDF: {e}")

# 4. Fetch from URL / API
elif source == "Fetch from URL / API":
    url = st.sidebar.text_input("Enter Dataset URL (JSON API or raw CSV link):")
    if url:
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            try:
                data = res.json()
                movies = {
                    title: np.array(vec, dtype=float)
                    for title, vec in data.items()
                }
            except ValueError:
                df = pd.read_csv(io.StringIO(res.text))
                movies = process_dataframe(df)
            st.sidebar.success(f"Imported {len(movies)} items from URL.")
        except Exception as e:
            st.sidebar.error(f"Failed to fetch data: {e}")

# -------------------------------------------------------------
# Main Application UI
# -------------------------------------------------------------
st.title("🎬 Smart Recommender Dashboard")
st.write(
    "Discover content similarities using vector alignment and NumPy cosine scoring."
)
st.divider()

if not movies:
    st.warning(
        "👈 Please select or upload a valid dataset using the sidebar menu to get started."
    )
else:
    # Case-insensitive mapping lookup
    movie_lookup = {title.lower(): title for title in movies}

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_input = st.selectbox(
            "Select or search for an item you like:",
            options=list(movies.keys()),
        )

    with col2:
        how_many = st.number_input(
            "Number of recommendations:", min_value=1, max_value=20, value=3
        )

    if st.button("Generate Recommendations", type="primary"):
        original_title = movie_lookup[selected_input.lower()]
        results = recommend(original_title, movies, how_many=how_many)

        st.subheader(f"Top Suggestions for '{original_title}':")

        if not results:
            st.info("No matching recommendations could be calculated.")
        else:
            for rank, (title, score) in enumerate(results, start=1):
                col_rank, col_title, col_score, col_bar = st.columns(
                    [1, 4, 3, 5]
                )
                with col_rank:
                    st.write(f"**#{rank}**")
                with col_title:
                    st.write(f"**{title}**")
                with col_score:
                    st.write(f"Match: **{score * 100:.1f}%**")
                with col_bar:
                    st.progress(max(0.0, min(1.0, score)))

            st.divider()

    # Data preview Expander
    with st.expander("🔍 View Active Library Dataset"):
        df_display = pd.DataFrame.from_dict(
            movies, orient="index"
        ).reset_index()
        df_display.columns = ["Item Title"] + [
            f"Feature_{i+1}" for i in range(df_display.shape[1] - 1)
        ]
        st.dataframe(df_display, use_container_width=True)