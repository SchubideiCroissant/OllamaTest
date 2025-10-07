# ================================
# Local Knowledge Base mit Ollama + Chroma
#
# Funktionsweise:
#   - Kommuniziert mit Ollama welches Nativ installiert ist
#   - Liest automatisch PDFs und Code aus angegebenen Ordnern
#   - Speichert Inhalte persistent in Chroma
#   - Beantwortet Fragen über Model (Ollama)
# ================================

import textwrap, shutil
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
CODE_DIR = "F:/Code/OllamaTest/code"                 # Ordner für Code-Dateien
MODEL_NAME = "llama3"                               # Ollama-Modell, Ilama für GPU Unterstützung

# ------------------------------
# CHROMA INITIALISIEREN
# ------------------------------
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection("local_knowledge")
print(f"Datenbankpfad: {PERSIST_DIR}")
print(f"Vorhandene Collections: {client.list_collections()}")

def split_text(text, size=500, overlap=100):
    """Teilt Text in überlappende Chunks."""
    text = text.replace("\n", " ").strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def process_pdf(file_path, chunk_size, overlap):
    """Extrahiert Text aus einer PDF und chunkt ihn."""
    from PyPDF2 import PdfReader
    docs, ids, metas = [], [], []
    reader = PdfReader(file_path)
    filename = os.path.basename(file_path)

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        if text.strip():
            page_chunks = split_text(text, chunk_size, overlap)
            for j, chunk in enumerate(page_chunks):
                docs.append(chunk)
                ids.append(f"pdf_{filename}_p{page_num}_c{j}")
                metas.append({
                    "filename": filename,
                    "page": page_num,
                    "chunk": j,
                    "type": "pdf"
                })
    return docs, ids, metas

def process_code(file_path, chunk_size, overlap):
    """Liest Code-Dateien ein und chunkt sie."""
    docs, ids, metas = [], [], []
    filename = os.path.basename(file_path)
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text:
            chunks = split_text(text, chunk_size, overlap)
            for j, chunk in enumerate(chunks):
                docs.append(chunk)
                ids.append(f"code_{filename}_c{j}")
                metas.append({
                    "filename": filename,
                    "chunk": j,
                    "type": "code"
                })
    except Exception as e:
        print(f"Fehler beim Laden von {filename}: {e}")
    return docs, ids, metas

def add_new_documents(collection, docs, ids, metadatas):
    """Fügt nur neue Dokumente zur Datenbank hinzu."""
    print(f"{len(docs)} Chunks vorbereitet. Überprüfe bestehende Datenbankeinträge ...")

    existing_data = collection.get()
    existing_ids = set(existing_data.get("ids", []))

    new_docs, new_ids, new_metas = [], [], []
    for d, i, m in zip(docs, ids, metadatas):
        if i not in existing_ids:
            new_docs.append(d)
            new_ids.append(i)
            new_metas.append(m)

    if new_docs:
        print(f"{len(new_docs)} neue Chunks werden hinzugefügt ...")
        collection.add(documents=new_docs, ids=new_ids, metadatas=new_metas)
        print("Datenbank erfolgreich aktualisiert.")
    else:
        print("Keine neuen Chunks gefunden.")

def index_files(chunk_size=500, overlap=100):
    """Liest PDFs und Code-Dateien, chunkt und speichert sie in Chroma."""
    docs, ids, metadatas = [], [], []

    # PDFs verarbeiten
    if os.path.isdir(PDF_DIR):
        for file in os.listdir(PDF_DIR):
            if file.lower().endswith(".pdf"):
                path = os.path.join(PDF_DIR, file)
                print(f"Lade PDF: {file}")
                d, i, m = process_pdf(path, chunk_size, overlap)
                docs += d; ids += i; metadatas += m

    # Code verarbeiten
    if os.path.isdir(CODE_DIR):
        print(f"Lade Code aus {CODE_DIR}")
        for root, _, files in os.walk(CODE_DIR):
            for file in files:
                if file.endswith((".c", ".cpp", ".h", ".py")):
                    path = os.path.join(root, file)
                    d, i, m = process_code(path, chunk_size, overlap)
                    docs += d; ids += i; metadatas += m

    # Neue Dokumente hinzufügen
    if docs:
        add_new_documents(collection, docs, ids, metadatas)
    else:
        print("Keine Dateien gefunden.")

def show_chunks(limit = 1000):
    """Zeigt gespeicherte Chunks in der Chroma-Datenbank (mit Metadaten und Vorschau)."""
    print("\n=== Gespeicherte Chunks ===")
    try:
        # Alle Daten abrufen
        data = collection.get(include=["metadatas", "documents"])
        ids = data.get("ids", [])
        print(f"Gesamt: {len(ids)} Chunks in der Collection '{collection.name}'\n")

        for i, (cid, meta, doc) in enumerate(zip(ids, data["metadatas"], data["documents"])):
            print(f"[{i+1}] ID: {cid}")
            if meta:
                print(f"   Datei: {meta.get('filename', '?')}")
                if meta.get("page"):
                    print(f"   Seite: {meta['page']}")
                print(f"   Typ: {meta.get('type', '?')}")
            snippet = doc[:200].replace("\n", " ") + ("..." if len(doc) > 200 else "")
            print(f"   Inhalt: {snippet}\n")

            if i + 1 >= limit:
                print(f"--- Ausgabe auf {limit} Chunks begrenzt ---")
                break

    except Exception as e:
        print(f"Fehler beim Laden der Chunks: {e}")

def filter_chunks(question):
    where_filter = {}
    q_lower = question.lower()

    # Automatische Dokumentauswahl
    if "pdf" in q_lower:
        where_filter = {"type": {"$eq": "pdf"}}
    elif "code" in q_lower:
        where_filter = {"type": {"$eq": "code"}}
    else:
        where_filter = {}  # keine Einschränkung → alle durchsuchen
    
    return where_filter
# ------------------------------
# FRAGEN AN MODEL
# ------------------------------
def ask(question):
    """Durchsucht die DB und fragt Model."""

    results = collection.query(
        query_texts=[question],
        n_results=6,
        where=filter_chunks(question),
        include=["documents", "metadatas"]
    )

    if not results["documents"] or not results["documents"][0]:
        print("Keine passenden Informationen gefunden.")
        return

    # Kontext aus den besten Treffern zusammensetzen
    context = "\n".join(results["documents"][0])

    # Quellen anzeigen
    print("\n=== Gefundene Quellen ===")
    for i, meta in enumerate(results["metadatas"][0]):
        cid = results["ids"][0][i] if "ids" in results else f"chunk_{i}"
        info = f"→ {cid}"
        if meta:
            if meta.get("filename"):
                info += f" | Datei: {meta['filename']}"
            if meta.get("page"):
                info += f" | Seite: {meta['page']}"
        print(info)
    print("==========================\n")



    prompt = (
    "Nutze ausschließlich die folgenden Informationen, um die Frage zu beantworten.\n"
    "Wenn du im Kontext keine direkte Antwort findest, gib eine plausible Zusammenfassung der gefundenen Inhalte wieder.'\n\n"
    f"--- KONTEXT START ---\n{context}\n--- KONTEXT ENDE ---\n\n"
    f"FRAGE: {question}"
)

    system_content =(
        f"Du bist ein technischer Assistent. "
        "Antworte klar, strukturiert und in vollständigen Sätzen. "
        "Sei präzise, aber liefere genügend Kontext, um die Antwort verständlich zu machen. "
        "Antworte ausschließlich auf Deutsch."
        "Verwende ausschließlich Informationen aus dem gegebenen Kontext."
    )
    

    """
                           options={
                                "num_predict": 300,   
                                "temperature": 0.3
                            },    
    """  
    term_width = shutil.get_terminal_size((100, 20)).columns
    buf = ""
    print("\n--- Antwort ---\n")
    # Ausgabe erfolgt Stück für Stück
    for chunk in ollama.chat(
        model=MODEL_NAME,
        messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
        ],
        stream=True
    ):
        buf += chunk["message"]["content"]
        # sobald genug Text oder ein Zeilenumbruch da ist → formatiert ausgeben
        if len(buf) > 400 or "\n" in buf:
            parts = buf.split("\n")
            for line in parts[:-1]:
                print(textwrap.fill(line, width=term_width))
            buf = parts[-1]  # Rest im Puffer lassen

# Rest flushen
    if buf:
        print(textwrap.fill(buf, width=term_width))


if __name__ == "__main__":
    print("=== Lokale Knowledge Base ===")
    print("Erstelle bzw. lade Datenbank...")
    index_files()
    show_chunks()

    while True:
        frage = input("\nFrage an deine Knowledge Base ('exit' zum Beenden): ")
        if frage.lower() in ("exit", "quit"):
            break
        ask(frage)
