import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Pastikan resource NLTK terdownload
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    def __init__(self):
        # Inisialisasi Sastrawi Stemmer
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        # Inisialisasi Stopwords Indonesia
        self.stop_words = set(stopwords.words("indonesian"))

    def preprocess_text(self, text):
        # 1. Case Folding
        text_lower = text.lower()

        # 2. Tokenization
        tokens = word_tokenize(text_lower)

        # 3. Stopword & Punctuation Removal
        clean_tokens = [
            token for token in tokens 
            if token not in string.punctuation and token not in self.stop_words
        ]

        # 4. Stemming
        stemmed_tokens = [self.stemmer.stem(word) for word in clean_tokens]

        # Mengembalikan semua tahap (sesuai request kamu)
        return text_lower, tokens, clean_tokens, stemmed_tokens