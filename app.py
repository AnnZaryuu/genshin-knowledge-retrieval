from flask import Flask, render_template, request
import numpy as np

# Import komponen yang sudah kita buat
from loader.pdf_loader import PDFLoader
from preprocessing.text_preprocessing import TextPreprocessor
from feature_extraction.tfidf import TFIDF

app = Flask(__name__)

# --- PROSES INITIALIZATION (Dijalankan saat server start) ---
# 1. Load semua dokumen PDF
loader = PDFLoader()
raw_docs = loader.load("dataset/")

# 2. Preprocessing semua dokumen
preprocessor = TextPreprocessor()
all_stemmed_tokens = {}
for filename, content in raw_docs.items():
    _, _, _, stemmed = preprocessor.preprocess_text(content)
    all_stemmed_tokens[filename] = stemmed

# 3. Hitung TF-IDF untuk seluruh dataset
tfidf_engine = TFIDF()
vocabulary = tfidf_engine.build_vocab(all_stemmed_tokens)
idf_scores = tfidf_engine.compute_idf(all_stemmed_tokens, vocabulary)

# 4. Simpan matrix TF-IDF tiap dokumen ke memori
tfidf_matrices = {}
for filename, tokens in all_stemmed_tokens.items():
    tf_scores = tfidf_engine.compute_tf(tokens)
    vector = tfidf_engine.compute_tfidf_vector(tf_scores, idf_scores, vocabulary)
    tfidf_matrices[filename] = vector

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html', results=None)

@app.route('/search')
def search():
    query = request.args.get('query', '')
    if not query:
        return render_template('index.html', results=None)

    # 1. Preprocess query dari user
    _, _, _, query_tokens = preprocessor.preprocess_text(query)

    # 2. Hitung vektor TF-IDF untuk query
    query_tf = tfidf_engine.compute_tf(query_tokens)
    query_vector = tfidf_engine.compute_tfidf_vector(query_tf, idf_scores, vocabulary)

    # 3. Hitung similarity query vs semua dokumen
    results = []
    for filename, doc_vector in tfidf_matrices.items():
        score = cosine_similarity(query_vector, doc_vector)
        if score > 0: # Hanya tampilkan yang relevan
            results.append((filename, score))

    # 4. Urutkan berdasarkan skor tertinggi
    results.sort(key=lambda x: x[1], reverse=True)

    return render_template('index.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)