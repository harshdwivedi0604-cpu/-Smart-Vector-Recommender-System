# recommender.py
import sys
import numpy as np
import pandas as pd
import pdfplumber
import requests


def calculate_cosine_similarity(vec_a, vec_b):
    """Computes cosine similarity between two numeric arrays using NumPy."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# -------------------------------------------------------------
# Data Loaders (CSV, PDF, URL)
# -------------------------------------------------------------
def load_from_csv(file_path):
    """Loads a CSV dataset using Pandas."""
    try:
        df = pd.read_csv(file_path)
        return process_dataframe(df)
    except FileNotFoundError:
        print(f"[Error] File not found at '{file_path}'.")
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
    return {}


def load_from_pdf(file_path):
    """Parses a PDF where each line contains 'Title: 5, 4, 3, 2'."""
    library = {}
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    for line in text.split("\n"):
                        if ":" in line:
                            title, vector_str = line.split(":", 1)
                            vector = [float(x.strip()) for x in vector_str.split(",")]
                            library[title.strip()] = np.array(vector)
        print(f"[PDF Loader] Successfully imported {len(library)} items.")
        return library
    except Exception as e:
        print(f"[Error] Failed to read PDF file: {e}")
        return {}


def load_from_url(url):
    """Fetches dataset from a web URL (JSON API or remote CSV)."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Try parsing as JSON first
        try:
            data = response.json()
            library = {title: np.array(vec, dtype=float) for title, vec in data.items()}
            print(f"[URL Loader] Loaded {len(library)} items from API.")
            return library
        except ValueError:
            # If not JSON, try parsing as CSV stream via Pandas
            import io
            df = pd.read_csv(io.StringIO(response.text))
            return process_dataframe(df)

    except Exception as e:
        print(f"[Error] Failed to load data from URL: {e}")
        return {}


def process_dataframe(df):
    """Processes a Pandas DataFrame to extract item titles and numeric vectors."""
    if df.empty or len(df.columns) < 2:
        print("[Warning] Dataset lacks sufficient rows or columns.")
        return {}

    name_col = df.columns[0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("[Warning] No numeric feature columns found.")
        return {}

    df[numeric_cols] = df[numeric_cols].fillna(0)

    library = {}
    for _, row in df.iterrows():
        item_name = str(row[name_col]).strip()
        vector = row[numeric_cols].values.astype(float)
        library[item_name] = vector

    print(f"[Pandas Loader] Imported {len(library)} items with {len(numeric_cols)} features each.")
    return library


# -------------------------------------------------------------
# Core Recommender Logic
# -------------------------------------------------------------
def recommend(chosen, library, how_many=3):
    """Generates top recommendations for a chosen item."""
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


# -------------------------------------------------------------
# Main Application Loop
# -------------------------------------------------------------
def main():
    # Built-in fallback dataset
    movies = {
        "Toy Story": np.array([5.0, 4.0, 3.0, 2.0]),
        "Die Hard": np.array([5.0, 1.0, 5.0, 2.0]),
        "Deadpool": np.array([4.0, 4.0, 5.0, 1.0]),
        "Inception": np.array([5.0, 1.0, 5.0, 4.0]),
        "The Dark Knight": np.array([5.0, 2.0, 2.0, 5.0]),
        "Superbad": np.array([1.0, 5.0, 0.0, 2.0]),
    }

    print("=" * 65)
    print("      SMART RECOMMENDER SYSTEM v1.0 (Multi-Source Support)      ")
    print("=" * 65)
    print("\nSelect Data Source:")
    print("  1. Default Built-in Dataset")
    print("  2. Load CSV File (Local)")
    print("  3. Load PDF File (Local)")
    print("  4. Fetch from URL / Web API")

    source_choice = input("\nEnter choice (1-4): ").strip()
    loaded_data = {}

    if source_choice == "2":
        path = input("Enter local CSV file path (e.g., movies.csv): ").strip()
        loaded_data = load_from_csv(path)
    elif source_choice == "3":
        path = input("Enter local PDF file path (e.g., movies.pdf): ").strip()
        loaded_data = load_from_pdf(path)
    elif source_choice == "4":
        url = input("Enter dataset URL (JSON API or CSV link): ").strip()
        loaded_data = load_from_url(url)

    if loaded_data:
        movies = loaded_data
    elif source_choice != "1":
        print("[System] Reverting to default built-in dataset.")

    if not movies:
        print("\n[Fatal Error] No items loaded. Exiting system.")
        sys.exit(1)

    movie_lookup = {title.lower(): title for title in movies}

    print("\n" + "-" * 65)
    print(f"Dataset active with {len(movies)} items.")
    print("-" * 65)

    while True:
        raw_choice = input("\nTell me an item you like (or type 'bye' to exit): ").strip()

        if not raw_choice:
            print("Please enter a valid title.")
            continue

        if raw_choice.lower() == "bye":
            print("\nThanks for using the Recommender! Goodbye!")
            break

        clean_choice = raw_choice.lower()

        if clean_choice not in movie_lookup:
            print(f"Sorry, couldn't find '{raw_choice}'. Please check spelling.")
            continue

        num_input = input("How many recommendations would you like? (default is 3): ").strip()
        how_many = int(num_input) if num_input.isdigit() and int(num_input) > 0 else 3

        original_title = movie_lookup[clean_choice]
        recommendations = recommend(original_title, movies, how_many=how_many)

        if not recommendations:
            print(f"\nNo valid matches found for '{original_title}'.")
        else:
            print(f"\nBecause you liked '{original_title}', you might enjoy:")
            for rank, (title, score) in enumerate(recommendations, start=1):
                print(
                    f"  {rank}. {title:<30} | Match: {score * 100:.1f}% "
                    f"(Cosine Score: {score:.4f})"
                )


if __name__ == "__main__":
    main()