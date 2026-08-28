# create_pdf.py
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Sample dataset of 50 movies with 6 feature vectors:
# [Action, Comedy, Sci-Fi, Drama, Romance, Thriller]
movie_data = [
    ("Toy Story", [2, 5, 3, 2, 1, 1]),
    ("Die Hard", [5, 1, 2, 3, 1, 5]),
    ("Deadpool", [5, 5, 3, 1, 1, 3]),
    ("Inception", [4, 1, 5, 4, 1, 5]),
    ("The Dark Knight", [5, 1, 2, 5, 1, 5]),
    ("Superbad", [1, 5, 0, 2, 2, 0]),
    ("Interstellar", [3, 1, 5, 5, 2, 3]),
    ("Titanic", [1, 1, 0, 5, 5, 2]),
    ("The Avengers", [5, 3, 4, 2, 1, 3]),
    ("La La Land", [1, 3, 0, 4, 5, 0]),
    ("Jurassic Park", [4, 1, 4, 3, 1, 4]),
    ("The Hangover", [1, 5, 0, 1, 1, 1]),
    ("The Matrix", [5, 1, 5, 3, 1, 4]),
    ("The Notebook", [0, 1, 0, 5, 5, 1]),
    ("Mad Max Fury Road", [5, 1, 3, 2, 0, 4]),
    ("Gladiator", [5, 0, 1, 5, 2, 3]),
    ("Pulp Fiction", [4, 3, 0, 4, 1, 5]),
    ("Forrest Gump", [2, 4, 0, 5, 4, 1]),
    ("The Silence of the Lambs", [2, 0, 0, 5, 0, 5]),
    ("Schindler's List", [1, 0, 0, 5, 1, 3]),
    ("Avatar", [5, 1, 5, 3, 2, 3]),
    ("Fight Club", [4, 2, 1, 5, 1, 5]),
    ("Goodfellas", [4, 2, 0, 5, 1, 4]),
    ("The Godfather", [4, 0, 0, 5, 2, 4]),
    ("Star Wars IV", [5, 2, 5, 2, 1, 3]),
    ("Back to the Future", [3, 5, 5, 2, 2, 2]),
    ("The Lion King", [2, 4, 0, 5, 2, 1]),
    ("Parasite", [3, 3, 0, 5, 1, 5]),
    ("Whiplash", [1, 1, 0, 5, 0, 4]),
    ("Spider-Man Into the Spider-Verse", [5, 4, 4, 3, 1, 2]),
    ("Joker", [3, 1, 0, 5, 0, 5]),
    ("Avengers Endgame", [5, 3, 5, 3, 1, 3]),
    ("Coco", [1, 4, 2, 5, 2, 1]),
    ("WALL-E", [2, 3, 5, 4, 3, 1]),
    ("Alien", [4, 0, 5, 2, 0, 5]),
    ("The Shining", [1, 0, 1, 4, 0, 5]),
    ("Terminator 2", [5, 1, 5, 2, 1, 4]),
    ("Saving Private Ryan", [5, 0, 0, 5, 1, 4]),
    ("Se7en", [3, 0, 0, 4, 0, 5]),
    ("The Prestige", [3, 1, 4, 4, 1, 5]),
    ("The Departure", [4, 1, 0, 5, 1, 5]),
    ("Mentalist", [2, 2, 0, 4, 1, 4]),
    ("Grand Budapest Hotel", [2, 5, 0, 3, 2, 2]),
    ("Knives Out", [2, 4, 0, 3, 0, 5]),
    ("Everything Everywhere All at Once", [5, 5, 5, 4, 2, 3]),
    ("Top Gun Maverick", [5, 1, 1, 3, 3, 4]),
    ("Dune", [4, 0, 5, 4, 1, 4]),
    ("Oppenheimer", [2, 0, 2, 5, 2, 4]),
    ("Barbie", [1, 5, 2, 3, 3, 0]),
    ("Spider-Man No Way Home", [5, 3, 4, 3, 2, 3])
]

def generate_pdf():
    pdf_filename = "movies_large.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()

    content = []

    # Title
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.navy,
        spaceAfter=12
    )
    content.append(Paragraph("Large Movie Feature Dataset", title_style))
    content.append(Spacer(1, 10))

    # Line Style
    line_style = ParagraphStyle(
        'LineStyle',
        parent=styles['Normal'],
        fontSize=11,
        leading=16
    )

    # Format each line as expected by your PDF loader: Title: 5, 4, 3, 2, ...
    for title, vector in movie_data:
        vec_str = ", ".join(map(str, vector))
        line_text = f"<b>{title}</b>: {vec_str}"
        content.append(Paragraph(line_text, line_style))

    doc.build(content)
    print(f"Successfully generated '{pdf_filename}' with {len(movie_data)} movies!")

if __name__ == "__main__":
    generate_pdf()