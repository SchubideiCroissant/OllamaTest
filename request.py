# ================================
# Local Knowledge Base mit Ollama + Chroma
#
# Funktionsweise:
#   - Kommuniziert mit Ollama welches Nativ installiert ist
#   - Liest automatisch PDFs und Code aus angegebenen Ordnern
#   - Speichert Inhalte persistent in Chroma
#   - Beantwortet Fragen über Model (Ollama)
# ================================

from pathlib import Path
import os
from chromadb import PersistentClient
from chromadb.config import Settings
from PyPDF2 import PdfReader
import ollama

# ------------------------------
# EINSTELLUNGEN
# ------------------------------
PERSIST_DIR = Path("F:/Code/OllamaTest/chroma_db") # Speicherort der Datenbank
PDF_DIR = "F:/Code/OllamaTest/docs"                 # Ordner für PDFs
CODE_DIR = "F:/Code/OllamaTest/src"                 # Ordner für Code-Dateien
MODEL_NAME = "llama3"                               # Ollama-Modell, Ilama für GPU Unterstützung

# ------------------------------
# CHROMA INITIALISIEREN
# ------------------------------
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection("local_knowledge")
print(f"Datenbankpfad: {PERSIST_DIR}")
print(f"Vorhandene Collections: {client.list_collections()}")

# ------------------------------
# FUNKTIONEN ZUM DATEIEN LADEN
# ------------------------------
def load_pdf_text(pdf_path):
    """Extrahiert Text aus einer PDF-Datei."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        try:
            text += page.extract_text() or ""
        except Exception:
            pass
    return text.strip()

def load_code_text(folder):
    """Liest Code-Dateien (C, C++, H, PY) rekursiv ein."""
    text_data = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith((".c", ".cpp", ".h", ".py")):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text_data.append(f.read())
                except Exception:
                    pass
    return "\n".join(text_data)

# ------------------------------
# INHALTE LADEN UND SPEICHERN
# ------------------------------
def index_files():
    docs = []
    ids = []

    # PDFs
    if os.path.isdir(PDF_DIR):
        for file in os.listdir(PDF_DIR):
            if file.lower().endswith(".pdf"):
                path = os.path.join(PDF_DIR, file)
                print(f"Lade PDF: {file}")
                text = load_pdf_text(path)
                if text:
                    docs.append(text)
                    ids.append(f"pdf_{file}")

    # Code
    if os.path.isdir(CODE_DIR):
        print(f"Lade Code aus {CODE_DIR}")
        code_text = load_code_text(CODE_DIR)
        if code_text:
            docs.append(code_text)
            ids.append("code_data")

    if docs:
        collection.add(documents=docs, ids=ids)
        print(f"{len(docs)} Dateien zur Datenbank hinzugefügt.")
    else:
        print("Keine neuen Dateien gefunden.")

# ------------------------------
# FRAGEN AN MODEL
# ------------------------------
def ask(question):
    """Durchsucht die DB und fragt Model."""
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n".join([doc for docs in results["documents"] for doc in docs])

    if not context:
        print("Keine passenden Informationen gefunden.")
        return


    prompt = (
        f"Kontext:\n{context}\n\nFrage: {question}"
)
    system_content =(
        f"Du bist ein prägnanter technischer Assistent. "
        "Antworte immer kurz, klar und ohne Ausschmückungen. "
        "Antworte ausschließlich auf Deutsch."
    )
    print("\n--- Model antwortet ---\n")

    """
                           options={
                                "num_predict": 300,   
                                "temperature": 0.3
                            },    
    """  
    for chunk in ollama.chat(
        model=MODEL_NAME,
        messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
        ],
        stream=True
    ):
        print(chunk["message"]["content"], end="", flush=True) # Changed to flush for shorter waiting times

if __name__ == "__main__":
    print("=== Lokale Knowledge Base ===")
    print("Erstelle bzw. lade Datenbank...")
    index_files()

    while True:
        frage = input("\nFrage an deine Knowledge Base ('exit' zum Beenden): ")
        if frage.lower() in ("exit", "quit"):
            break
        ask(frage)
