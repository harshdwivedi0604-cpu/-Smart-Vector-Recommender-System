# app.py
import io
import urllib.parse
import numpy as np
import pandas as pd
import pdfplumber
import requests
import streamlit as st
from PIL import Image

# IMDb Official API Integration
try:
    from imdb import Cinemagoer

    ia = Cinemagoer()
    HAS_IMDB = True
except Exception:
    HAS_IMDB = False

# 3D Data Visualization Libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA

    HAS_3D_LIBS = True
except ImportError:
    HAS_3D_LIBS = False

# Page Configuration
st.set_page_config(
    page_title="Smart Recommender 3D Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# DIRECT OFFICIAL MOVIE POSTER CDN MAP (Apple iTunes HD CDNs)
# -------------------------------------------------------------
OFFICIAL_POSTER_MAP = {
    "toy story": "https://is1-ssl.mzstatic.com/image/thumb/Video113/v4/44/85/33/448533e7-1300-aa13-5a02-53609805908b/pos_US.jpg/600x600bb.jpg",
    "die hard": "https://is1-ssl.mzstatic.com/image/thumb/Video114/v4/d5/d3/eb/d5d3eb7a-0994-6e69-9c59-2a91219b2656/024543034988_cover.jpg/600x600bb.jpg",
    "deadpool": "https://is1-ssl.mzstatic.com/image/thumb/Video118/v4/0d/bb/0e/0dbb0e35-6187-b952-4753-48b264192b67/024543169000_cover.jpg/600x600bb.jpg",
    "inception": "https://is1-ssl.mzstatic.com/image/thumb/Video124/v4/91/95/43/919543ab-b789-f5c7-e612-581d60492822/mza_4833215273578749455.png/600x600bb.jpg",
    "the dark knight": "https://is1-ssl.mzstatic.com/image/thumb/Video113/v4/7e/c9/7e/7ec97e41-6548-c89b-f111-9a70058e0406/mza_11707577526978438692.jpg/600x600bb.jpg",
    "superbad": "https://is1-ssl.mzstatic.com/image/thumb/Video123/v4/31/58/68/315868ab-6a7c-bc7d-9a67-9d7a6be3b18d/043396218772_cover.jpg/600x600bb.jpg",
    "interstellar": "https://is1-ssl.mzstatic.com/image/thumb/Video115/v4/9d/b1/13/9db113e1-8071-6c24-5d92-bf3936d5b084/mza_8545811776510848039.jpg/600x600bb.jpg",
    "titanic": "https://is1-ssl.mzstatic.com/image/thumb/Video114/v4/80/7e/ef/807eef33-5c2c-809c-352b-7c01b96a9307/024543088004_cover.jpg/600x600bb.jpg",
    "the avengers": "https://is1-ssl.mzstatic.com/image/thumb/Video113/v4/bf/16/e2/bf16e2b8-971c-4375-10eb-0c0349479b18/pos_US.jpg/600x600bb.jpg",
    "la la land": "https://is1-ssl.mzstatic.com/image/thumb/Video124/v4/8a/04/b3/8a04b3a4-8b65-6832-6a56-6a2c3fb668a6/mza_2865955684617637841.jpg/600x600bb.jpg",
}


# -------------------------------------------------------------
# REAL OFFICIAL MOVIE POSTER PIPELINE
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_movie_poster(movie_title, custom_url=None):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    # 1. Custom URL from uploaded dataset
    if custom_url and isinstance(custom_url, str) and custom_url.strip().startswith("http"):
        try:
            res = requests.get(custom_url.strip(), headers=headers, timeout=4)
            if res.status_code == 200:
                return Image.open(io.BytesIO(res.content))
        except Exception:
            pass

    clean_title = str(movie_title).strip()
    title_key = clean_title.lower()

    # 2. Hardcoded Official iTunes Poster Mapping (Instant & 100% Reliable)
    if title_key in OFFICIAL_POSTER_MAP:
        try:
            res = requests.get(OFFICIAL_POSTER_MAP[title_key], headers=headers, timeout=4)
            if res.status_code == 200:
                return Image.open(io.BytesIO(res.content))
        except Exception:
            pass

    # 3. Dynamic iTunes Movie Store API Fetch (For any custom uploaded title)
    try:
        query_url = f"https://itunes.apple.com/search?term={urllib.parse.quote(clean_title)}&entity=movie&limit=1"
        res = requests.get(query_url, headers=headers, timeout=4)
        if res.status_code == 200:
            results = res.json().get("results", [])
            if results and "artworkUrl100" in results[0]:
                hd_poster_url = results[0]["artworkUrl100"].replace("100x100bb", "600x600bb")
                img_res = requests.get(hd_poster_url, headers=headers, timeout=4)
                if img_res.status_code == 200:
                    return Image.open(io.BytesIO(img_res.content))
    except Exception:
        pass

    # 4. Wikipedia REST Summary API (Official theatrical poster from infobox)
    for wiki_query in [clean_title, f"{clean_title}_(film)"]:
        try:
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(wiki_query)}"
            r = requests.get(wiki_url, headers=headers, timeout=3)
            if r.status_code == 200:
                data = r.json()
                if "thumbnail" in data and "source" in data["thumbnail"]:
                    poster_url = data["thumbnail"]["source"]
                    img_res = requests.get(poster_url, headers=headers, timeout=4)
                    if img_res.status_code == 200:
                        return Image.open(io.BytesIO(img_res.content))
        except Exception:
            pass

    # 5. IMDb Cinemagoer Fallback
    if HAS_IMDB:
        try:
            results = ia.search_movie(clean_title)
            if results:
                movie = ia.get_movie(results[0].movieID)
                cover_url = movie.get("full-size cover url") or movie.get("cover url")
                if cover_url:
                    img_res = requests.get(cover_url, headers=headers, timeout=4)
                    if img_res.status_code == 200:
                        return Image.open(io.BytesIO(img_res.content))
        except Exception:
            pass

    return None


# -------------------------------------------------------------
# Algorithmic Logic
# -------------------------------------------------------------
def calculate_cosine_similarity(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def recommend(chosen, library, how_many=3, min_threshold=0.0):
    scores = []
    chosen_vec = library[chosen]["vector"]

    for title, data in library.items():
        if title == chosen:
            continue
        vector = data["vector"]
        if len(chosen_vec) != len(vector):
            continue
        s = calculate_cosine_similarity(chosen_vec, vector)

        if s >= min_threshold:
            scores.append((title, s, data.get("poster_url")))

    scores.sort(key=lambda pair: pair[1], reverse=True)
    return scores[:how_many]


def process_dataframe(df):
    if df.empty or len(df.columns) < 2:
        st.sidebar.error("Dataset lacks sufficient rows or columns.")
        return {}

    name_col = df.columns[0]
    poster_col = None
    for col in df.columns:
        if col.lower() in ["poster", "image", "picture", "poster_url", "image_url"]:
            poster_col = col
            break

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.sidebar.error("No numeric feature columns found in dataset.")
        return {}

    df[numeric_cols] = df[numeric_cols].fillna(0)

    library = {}
    for _, row in df.iterrows():
        item_name = str(row[name_col]).strip()
        vector = row[numeric_cols].values.astype(float)
        custom_poster = str(row[poster_col]) if poster_col else None

        library[item_name] = {"vector": vector, "poster_url": custom_poster}

    return library


# -------------------------------------------------------------
# UI Sidebar Controls
# -------------------------------------------------------------
st.sidebar.title("⚙️ 3D Engine Control Center")

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

if source == "Default Dataset":
    raw_default = {
        "Toy Story": np.array([2.0, 5.0, 3.0, 2.0, 1.0, 1.0]),
        "Die Hard": np.array([5.0, 1.0, 2.0, 3.0, 1.0, 5.0]),
        "Deadpool": np.array([5.0, 5.0, 3.0, 1.0, 1.0, 3.0]),
        "Inception": np.array([4.0, 1.0, 5.0, 4.0, 1.0, 5.0]),
        "The Dark Knight": np.array([5.0, 1.0, 2.0, 5.0, 1.0, 5.0]),
        "Superbad": np.array([1.0, 5.0, 0.0, 2.0, 2.0, 0.0]),
        "Interstellar": np.array([3.0, 1.0, 5.0, 5.0, 2.0, 3.0]),
        "Titanic": np.array([1.0, 1.0, 0.0, 5.0, 5.0, 2.0]),
        "The Avengers": np.array([5.0, 3.0, 4.0, 2.0, 1.0, 3.0]),
        "La La Land": np.array([1.0, 3.0, 0.0, 4.0, 5.0, 0.0]),
    }
    movies = {title: {"vector": vec, "poster_url": None} for title, vec in raw_default.items()}
    st.sidebar.success(f"Loaded built-in dataset ({len(movies)} items).")

elif source == "Upload CSV File":
    uploaded_csv = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            movies = process_dataframe(df)
            st.sidebar.success(f"Imported {len(movies)} items from CSV.")
        except Exception as e:
            st.sidebar.error(f"Failed to parse CSV: {e}")

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
                                vector = [float(x.strip()) for x in vector_str.split(",")]
                                library[title.strip()] = {"vector": np.array(vector), "poster_url": None}
            movies = library
            st.sidebar.success(f"Imported {len(movies)} items from PDF.")
        except Exception as e:
            st.sidebar.error(f"Failed to parse PDF: {e}")

elif source == "Fetch from URL / API":
    url = st.sidebar.text_input("Enter Dataset URL (JSON API or raw CSV link):")
    if url:
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            try:
                data = res.json()
                movies = {title: {"vector": np.array(vec, dtype=float), "poster_url": None} for title, vec in data.items()}
            except ValueError:
                df = pd.read_csv(io.StringIO(res.text))
                movies = process_dataframe(df)
            st.sidebar.success(f"Imported {len(movies)} items from URL.")
        except Exception as e:
            st.sidebar.error(f"Failed to fetch data: {e}")

st.sidebar.divider()
min_match_percentage = st.sidebar.slider("Minimum Similarity Cutoff (%)", 0, 100, 20) / 100.0

# -------------------------------------------------------------
# Main Dashboard UI
# -------------------------------------------------------------
st.title("🎬 Smart Recommender Dashboard")

if not movies:
    st.warning("👈 Select or upload a dataset using the sidebar menu to activate the engine.")
else:
    tab_engine, tab_3d_pca, tab_3d_surface, tab_data = st.tabs(
        [
            "🎯 Recommendations",
            "🧊 3D Vector Space",
            "🏔️ 3D Similarity Landscape",
            "🔍 Active Matrix",
        ]
    )

    movie_lookup = {title.lower(): title for title in movies}

    with tab_engine:
        col1, col2 = st.columns([2, 1])

        with col1:
            selected_input = st.selectbox("Choose a movie you like:", options=list(movies.keys()))

        with col2:
            how_many = st.number_input("Recommendation Limit:", min_value=1, max_value=20, value=3)

        if st.button("Run Recommendation Engine", type="primary"):
            original_title = movie_lookup[selected_input.lower()]
            results = recommend(
                original_title,
                movies,
                how_many=how_many,
                min_threshold=min_match_percentage,
            )

            st.subheader(f"Recommendations for '{original_title}':")

            if not results:
                st.info("No items met your specified minimum similarity cutoff threshold.")
            else:
                grid_cols = st.columns(min(len(results), 3))

                for index, (title, score, custom_url) in enumerate(results):
                    col_target = grid_cols[index % len(grid_cols)]
                    poster_img = fetch_movie_poster(title, custom_url)

                    with col_target:
                        with st.container(border=True):
                            if poster_img:
                                st.image(poster_img, use_container_width=True)
                            else:
                                st.warning(f"No poster available for {title}")
                            st.subheader(f"#{index+1} {title}")
                            st.caption("Cosine Match Score")
                            st.markdown(f"### `{score * 100:.1f}%`")
                            st.progress(max(0.0, min(1.0, score)))

    with tab_3d_pca:
        st.subheader("Interactive 3D Spatial Vector Projection")
        if HAS_3D_LIBS and len(movies) >= 4:
            all_titles = list(movies.keys())
            all_vecs = np.array([item["vector"] for item in movies.values()])

            pca = PCA(n_components=3)
            coords_3d = pca.fit_transform(all_vecs)

            df_3d = pd.DataFrame(coords_3d, columns=["PCA Dim 1", "PCA Dim 2", "PCA Dim 3"])
            df_3d["Title"] = all_titles
            df_3d["Target Selected"] = df_3d["Title"] == selected_input

            fig_3d = px.scatter_3d(
                df_3d,
                x="PCA Dim 1",
                y="PCA Dim 2",
                z="PCA Dim 3",
                text="Title",
                color="Target Selected",
                color_discrete_map={True: "#FF3366", False: "#818CF8"},
                opacity=0.9,
                size_max=18,
            )
            fig_3d.update_layout(paper_bgcolor="#0E1117", font=dict(color="#E2E8F0"), height=600)
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("Requires at least 4 items and scikit-learn/plotly to display 3D space.")

    with tab_3d_surface:
        st.subheader("3D Pairwise Similarity Topology Surface")
        if HAS_3D_LIBS and len(movies) >= 3:
            all_titles = list(movies.keys())
            all_vecs = np.array([item["vector"] for item in movies.values()])

            n = len(all_vecs)
            matrix_3d = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    matrix_3d[i][j] = calculate_cosine_similarity(all_vecs[i], all_vecs[j])

            fig_surface = go.Figure(data=[go.Surface(z=matrix_3d, x=all_titles, y=all_titles, colorscale="Viridis")])
            fig_surface.update_layout(paper_bgcolor="#0E1117", font=dict(color="#E2E8F0"), height=600)
            st.plotly_chart(fig_surface, use_container_width=True)

    with tab_data:
        st.subheader("🔍 Dataset Matrix Inspector")
        data_rows = []
        for title, item in movies.items():
            row = {"Movie Title": title}
            for i, val in enumerate(item["vector"]):
                row[f"Feature_{i+1}"] = val
            data_rows.append(row)

        st.dataframe(pd.DataFrame(data_rows), use_container_width=True)