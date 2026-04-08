import numpy as np
from collections import Counter

class BagOfWords:
    def __init__(self):
        self.vocabulary = []
        self.document_vectors = []
        self.filenames = []

    def fit_transform(self, docs_dict):
        """
        Menerima input dictionary {filename: list_of_tokens}
        Contoh: {'venti.txt': ['archon', 'anemo', 'mondstadt'], ...}
        """
        self.filenames = list(docs_dict.keys())
        all_tokens = docs_dict.values()
        
        # 1. Bangun Vocabulary (Daftar semua kata unik di korpus)
        unique_words = sorted(list(set(word for tokens in all_tokens for word in tokens)))
        self.vocabulary = unique_words
        
        # 2. Hitung frekuensi kata untuk setiap dokumen
        vectors = []
        for filename in self.filenames:
            tokens = docs_dict[filename]
            counts = Counter(tokens)
            
            # Buat vektor berdasarkan urutan vocabulary
            vector = [counts.get(word, 0) for word in self.vocabulary]
            vectors.append(vector)
            
        self.document_vectors = np.array(vectors)
        return self.document_vectors

    def transform_query(self, query_tokens):
        """Mengubah query menjadi vektor BoW berdasarkan vocabulary yang ada"""
        counts = Counter(query_tokens)
        return np.array([counts.get(word, 0) for word in self.vocabulary])

    def rank_documents(self, query_tokens):
        """Melakukan ranking dokumen berdasarkan Dot Product (kemunculan terbanyak)"""
        if not self.vocabulary:
            return []

        query_vector = self.transform_query(query_tokens)
        
        # Hitung skor (menggunakan dot product untuk menghitung kemunculan kata query di dokumen)
        # Semakin banyak kata query muncul di dokumen, semakin tinggi skornya
        scores = np.dot(self.document_vectors, query_vector)
        
        # Satukan dengan nama file dan urutkan
        results = zip(self.filenames, scores)
        return sorted(results, key=lambda x: x[1], reverse=True)