from flask import Flask, render_template, request
import numpy as np

# Import komponen original
from loader.pdf_loader import PDFLoader
from preprocessing.text_preprocessing import TextPreprocessor
from feature_extraction.tfidf import TFIDF
# Import komponen BoW baru
from feature_extraction.BoW import BagOfWords

app = Flask(__name__)

# --- PROSES INITIALIZATION ---
# 1. Load semua dokumen PDF
loader = PDFLoader()
raw_docs = loader.load("dataset/")

# 2. Preprocessing semua dokumen
preprocessor = TextPreprocessor()
all_stemmed_tokens = {}
for filename, content in raw_docs.items():
    _, _, _, stemmed = preprocessor.preprocess_text(content)
    all_stemmed_tokens[filename] = stemmed

# 3. Hitung TF-IDF Setup
tfidf_engine = TFIDF()
vocabulary = tfidf_engine.build_vocab(all_stemmed_tokens)
idf_scores = tfidf_engine.compute_idf(all_stemmed_tokens, vocabulary)

tfidf_matrices = {}
for filename, tokens in all_stemmed_tokens.items():
    tf_scores = tfidf_engine.compute_tf(tokens)
    vector = tfidf_engine.compute_tfidf_vector(tf_scores, idf_scores, vocabulary)
    tfidf_matrices[filename] = vector

# 4. Inisialisasi & Hitung Matrix BoW (Baru)
bow_engine = BagOfWords()
# Kita gunakan hasil preprocessing yang sama agar adil
bow_engine.fit_transform(all_stemmed_tokens) 

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

    # 1. Preprocess query (digunakan untuk keduanya)
    _, _, _, query_tokens = preprocessor.preprocess_text(query)

    # --- LOGIKA TF-IDF ---
    query_tf = tfidf_engine.compute_tf(query_tokens)
    query_vector = tfidf_engine.compute_tfidf_vector(query_tf, idf_scores, vocabulary)

    tfidf_results = []
    for filename, doc_vector in tfidf_matrices.items():
        score = cosine_similarity(query_vector, doc_vector)
        if score > 0:
            tfidf_results.append((filename, score))
    tfidf_results.sort(key=lambda x: x[1], reverse=True)

    # --- LOGIKA BAG OF WORDS (Baru) ---
    # Memanggil fungsi rank_documents dari BoW.py
    bow_results = bow_engine.rank_documents(query_tokens)
    # Filter skor 0 agar konsisten dengan TF-IDF
    bow_results = [res for res in bow_results if res[1] > 0]

    # 4. Kirim kedua hasil ke template
    return render_template(
        'index.html', 
        query=query, 
        results=tfidf_results,     # Untuk tab TF-IDF
        bow_results=bow_results    # Untuk tab BoW
    )

# Route untuk Unduh File
@app.route('/download/<filename>')
def download_file(filename):
    # Pastikan path "dataset/" sesuai dengan lokasi file PDF kamu
    return send_from_directory("dataset", filename, as_attachment=True)

# Route untuk Detail (Melihat isi teks yang sudah diekstrak)
@app.route('/detail/<filename>')
def detail(filename):
    # Ambil konten asli dari raw_docs yang sudah di-load di awal
    content = raw_docs.get(filename, "Konten tidak ditemukan.")
    
    # Kamu bisa membuat template detail.html baru atau kirim balik ke index dengan modal
    return render_template('detail.html', filename=filename, content=content)

if __name__ == '__main__':
    app.run(debug=True)