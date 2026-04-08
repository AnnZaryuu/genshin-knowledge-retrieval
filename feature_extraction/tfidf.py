import math
from collections import Counter

class TFIDF:
    def build_vocab(self, docs_tokens):
        """Membangun daftar kata unik dari seluruh dokumen."""
        vocab = set()
        for tokens in docs_tokens.values():
            vocab.update(tokens)
        return sorted(list(vocab))

    def compute_tf(self, tokens):
        """Menghitung Term Frequency (TF)."""
        counter = Counter(tokens)
        total_terms = len(tokens)
        tf = {}
        for word, count in counter.items():
            tf[word] = count / total_terms
        return tf

    def compute_idf(self, docs_tokens, vocab):
        """Menghitung Inverse Document Frequency (IDF)."""
        N = len(docs_tokens)
        idf = {}
        for word in vocab:
            df = sum(1 for tokens in docs_tokens.values() if word in tokens)
            # Menghindari pembagian dengan nol dengan log(N / df)
            idf[word] = math.log(N / df) if df > 0 else 0
        return idf

    def compute_tfidf_vector(self, tf, idf, vocab):
        """Menghitung skor TF-IDF untuk setiap kata dalam vocab."""
        vector = []
        for word in vocab:
            value = tf.get(word, 0) * idf.get(word, 0)
            vector.append(value)
        return vector