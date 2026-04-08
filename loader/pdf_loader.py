import os
import pdfplumber

class PDFLoader:

    def load(self, folder_path):
        """
        Melakukan looping ke folder dataset, membaca setiap file PDF,
        dan mengembalikan dictionary {nama_file: isi_teks}.
        """
        documents = {}

        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' tidak ditemukan.")
            return documents

        # Iterasi setiap file di dalam folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)
                
                try:
                    text_content = []
                    
                    # Membuka file PDF
                    with pdfplumber.open(file_path) as pdf:
                        # Looping setiap halaman dalam satu PDF
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text_content.append(page_text)
                    
                    # Gabungkan teks dari semua halaman dan simpan ke dictionary
                    documents[filename] = "\n".join(text_content)
                    
                except Exception as e:
                    print(f"Gagal membaca file {filename}: {str(e)}")

        return documents

# Contoh penggunaan (opsional):
# if __name__ == "__main__":
#     loader = PDFLoader()
#     data = loader.load("dataset/")
#     print(data)