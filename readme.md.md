🎬 Smart 3D Recommender Dashboard

A machine learning-driven web application built with Streamlit that
provides content-

based recommendations, interactive 3D vector space visualizations,
similarity topology

landscapes, and a backend-driven poster fetching pipeline.

✨ Key Features

> \* 🎯 Content-Based Recommendation Engine: Calculates vector
> similarity between

items using Cosine Similarity to return top N matching recommendations.

> \* 🧊 3D Spatial Vector Projection (PCA): Reduces multi-dimensional
> item features into

3D spatial coordinates (X, Y, Z) using Principal Component Analysis for
interactive spatial

exploration.

> \* 🏔 3D Similarity Landscape: Generates an N \times N pairwise cosine
> matrix and

renders a 3D elevation surface map (topology terrain) using Plotly.

> \* 🖼 Multi-Source Backend Image Pipeline:
>
> \* Solves client-side browser CORS and 403 Forbidden errors by
> downloading raw

image bytes directly via Python backend (requests + PIL).

> \* Features a fallback chain: Custom Dataset URL \rightarrow iTunes HD
> Direct CDN

\rightarrow iTunes RESTAPI \rightarrow Wikipedia RESTAPI \rightarrow
IMDb

Cinemagoer.

> \* 📁 Multi-Format Data Input: Supports Default Dataset, CSV Upload,
> PDF Extraction

(via pdfplumber), and Live JSON/CSV API URLs.

> \* 🎛 Interactive Controls: Real-time filtering with similarity cutoff
> percentage sliders,

recommendation limit counters, and metric progress bars.

🛠 Tech Stack

\| Domain \| Libraries / Tools Used \|

\|---\|---\|

\| Frontend / Web UI \| Streamlit \|

\| Data Processing & Vectors \| NumPy, Pandas \|

\| Machine Learning & Math \| Scikit-Learn (PCA), Cosine Vector Math \|

\| 3D Data Visualizations \| Plotly Express, Plotly Graph Objects \|

\| Image & PDF Processing \| Pillow (PIL), Requests, Pdfplumber \|

\| External APIs / IMDb \| iTunes API, Wikipedia API, Cinemagoer \|

📐 Mathematical & Architectural Logic

1\. Cosine Similarity Formula

The recommendation engine measures the similarity between two
N-dimensional feature

vectors \mathbf{A} and \mathbf{B}:

> \* Dot Product (\mathbf{A} \cdot \mathbf{B}): Multiplies matching
> feature indices and

sums them up.

> \* Magnitude (\Vert{}\mathbf{A}\Vert{} \Vert{}\mathbf{B}\Vert{}):
> Normalizes vector lengths

to ensure magnitude scale does not skew similarity.

> \* Output is a normalized match score between 0.0 (0%) and 1.0 (100%).

2\. Backend Image Fetching Pipeline

To eliminate broken image icons caused by hotlink protection, the
application uses a

memory-buffered fetching pipeline:

\[User Interface\]

> │
>
> ▼

1\. CSV/PDF Custom URL Check ──(Found)──► Return Image

> │ (Not Found)
>
> ▼

2\. iTunes HD Direct CDN Map ──(Found)──► Download Raw Bytes

> │ (Not Found)
>
> ▼

3\. Dynamic iTunes API Search ─(Found)──► Convert to Bytes via PIL

> │ (Not Found)
>
> ▼

4\. Wikipedia RESTAPI Summary ─(Found)──► Render in Streamlit

> │ (Not Found)
>
> ▼

5\. IMDb Cinemagoer Fallback ──(Found)──► Fetch Official Cover

3\. Dimensionality Reduction (PCA)

Since feature vectors often contain 6+ dimensions that cannot be
visualized directly,

Principal Component Analysis (PCA) projects high-dimensional data onto 3
orthogonal

axes (X, Y, Z) that preserve maximum variance.

🚀 Installation & Setup

Prerequisites

Make sure Python 3.8+ is installed on your system.

1\. Clone or Download Project

git clone <https://github.com/your-username/smart-3d-recommender.git>

cd smart-3d-recommender

2\. Install Required Dependencies

Run the following command in your terminal/command prompt:

pip install streamlit numpy pandas pdfplumber requests pillow plotly
scikit-learn

cinemagoer

3\. Run the Dashboard

Launch the Streamlit app:

streamlit run app.py

The app will automatically open in your default browser at
http://localhost:8501.

📁 Project Structure

smart-3d-recommender/

│

├── app.py

├── README.md

\# Main application code (UI, Math Engine, Image Pipeline)

> \# Project documentation

└── requirements.txt \# List of dependencies

🎮 How to Use

> \* Select Data Source from the sidebar menu (Default Dataset, CSV
> Upload, PDF

Upload, or URLAPI).

> \* Set the Minimum Similarity Cutoff (%) slider in the sidebar.
>
> \* In the 🎯 Recommendations tab, choose a movie from the dropdown
> menu and set

the recommendation limit.

> \* Click Run Recommendation Engine.
>
> \* Explore the output tabs:
>
> \* 🎯 Recommendations: Displays matching items with match scores,
> progress bars,

and high-resolution posters.

> \* 🧊 3D Vector Space: Interactive 3D spatial plot showing relative
> distance between

item features.

> \* 🏔 3D Similarity Landscape: 3D surface grid showing pairwise
> similarity terrain

across all items.

> \* 🔍 Active Matrix: Live table view of the numerical feature vectors.
