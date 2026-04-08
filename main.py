from loader.pdf_loader import PDFLoader
from preprocessing.text_preprocessing import TextPreprocessor
from feature_extraction.tfidf import TFIDF

# 1. Load PDF
loader = PDFLoader()
raw_docs = loader.load("dataset/") # Hasil: {'file.pdf': 'isi teks...'}

# 2. Preprocessing
preprocessor = TextPreprocessor()
all_stemmed_tokens = {}

for filename, content in raw_docs.items():
    _, _, _, stemmed = preprocessor.preprocess_text(content)
    all_stemmed_tokens[filename] = stemmed

# 3. Feature Extraction (TF-IDF)
tfidf_engine = TFIDF()
vocabulary = tfidf_engine.build_vocab(all_stemmed_tokens)
idf_scores = tfidf_engine.compute_idf(all_stemmed_tokens, vocabulary)

# Hitung vector TF-IDF untuk setiap dokumen
tfidf_matrices = {}
for filename, tokens in all_stemmed_tokens.items():
    tf_scores = tfidf_engine.compute_tf(tokens)
    vector = tfidf_engine.compute_tfidf_vector(tf_scores, idf_scores, vocabulary)
    tfidf_matrices[filename] = vector

print(f"Vocab size: {len(vocabulary)} kata.")
print("Proses selesai!")

