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
import os, sys
from chromadb import PersistentClient
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nomic import embed
from PyPDF2 import PdfReader
import ollama
import json
import re
from tool_registry import TOOLS, format_output, generate_tool_descriptions

# ------------------------------
# EINSTELLUNGEN
# ------------------------------
current_mode = "auto"

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


def print_help():
    print("""
Verfügbare Befehle:
  tool   → Schaltet in den Tool-Modus (GitHub-Tools)
  rag    → Schaltet in den Wissensdatenbank-Modus (Chroma)
  auto   → Automatische Erkennung (Standard)
  status → Zeigt den aktuellen Modus
  help   → Zeigt diese Hilfe
  exit   → Beendet das Programm
    """)

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

def process_pdf(path, chunk_size=500, overlap=100):
    """Liest eine PDF vollständig ein, chunked seitenübergreifend und behält Seiteninfos."""
    reader = PdfReader(path)

    full_text = ""
    page_map = []  # (start_char, end_char, page_number)

    # 1. Gesamten Text + Seitenbereiche erfassen
    char_index = 0
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        start = char_index
        end = char_index + len(text)
        page_map.append((start, end, page_num))
        full_text += text + "\n"
        char_index = end + 1

    # 2. Mit RecursiveCharacterTextSplitter aufteilen
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(full_text)

    # 3. IDs, Metadaten und Dokumente erzeugen
    docs, ids, metadatas = [], [], []
    base_name = os.path.basename(path)

    for idx, chunk in enumerate(chunks):
        # Finde zugehörige Seiten basierend auf Characterposition
        start_char = full_text.find(chunk)
        end_char = start_char + len(chunk)
        pages_in_chunk = [
            p for (s, e, p) in page_map if not (e < start_char or s > end_char)
        ]
        page_info = f"{min(pages_in_chunk)}-{max(pages_in_chunk)}" if len(pages_in_chunk) > 1 else str(pages_in_chunk[0])

        docs.append(chunk)
        ids.append(f"{base_name}_chunk_{idx}")
        metadatas.append({
            "source": base_name,
            "pages": page_info,
            "path": path
        })

    print(f"{len(chunks)} Chunks aus {base_name} erzeugt.")
    return docs, ids, metadatas

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

import os

import os

def show_chunks(limit=1000):
    """Zeigt gespeicherte Chunks in der Chroma-Datenbank (mit Metadaten und Vorschau)."""
    print("\n=== Gespeicherte Chunks ===")
    try:
        # Gesamtzahl abrufen (neuere Versionen unterstützen .count())
        try:
            total = collection.count()
        except Exception:
            total = None

        # Alle Daten abrufen (ohne 'ids' im include!)
        data = collection.get(include=["metadatas", "documents"])
        ids = data.get("ids", [])
        metas = data.get("metadatas", [])
        docs  = data.get("documents", [])

        if total:
            print(f"Gesamt: {total} Chunks in Collection '{collection.name}' (zeige max. {limit})\n")
        else:
            print(f"Gesamt: {len(ids)} Chunks in Collection '{collection.name}'\n")

        # Iteration über limit
        for i, (cid, meta, doc) in enumerate(zip(ids, metas, docs)):
            if i >= limit:
                print(f"--- Ausgabe auf {limit} Chunks begrenzt ---")
                break

            meta = meta or {}

            # Einheitliche Metadaten auslesen
            filename = meta.get("filename") or meta.get("source")
            if not filename and meta.get("path"):
                filename = os.path.basename(meta["path"])
            filename = filename or "?"

            pages = meta.get("page") or meta.get("pages")
            typ = meta.get("type")
            if not typ and filename and "." in filename:
                typ = os.path.splitext(filename)[1].lstrip(".").lower()

            # Ausgabe
            print(f"[{i+1}] ID: {cid}")
            print(f"   Datei: {filename}")
            if pages:
                print(f"   Seiten: {pages}")
            if typ:
                print(f"   Typ: {typ}")

            snippet = (doc or "").replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            print(f"   Inhalt: {snippet}\n")

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
        where_filter = None  # keine Einschränkung → alle durchsuchen
    
    return where_filter

def handle_command(cmd: str):
    global current_mode

    cmd = cmd.strip().lower()

    if cmd == "tool":
        current_mode = "tool"
        print("Modus geändert zu: Tool")
    elif cmd == "rag":
        current_mode = "rag"
        print("Modus geändert zu: Wissensdatenbank (RAG)")
    elif cmd == "auto":
        current_mode = "auto"
        print("Modus geändert zu: Automatisch")
    elif cmd == "status":
        print(f"Aktueller Modus: {current_mode}")
    elif cmd == "help":
        print_help()
    elif cmd in ("exit", "quit"):
        print("Programm wird beendet.")
        sys.exit(0)
    else:
        # An dieser Stelle wird dein bisheriger Code eingebunden:
        if current_mode == "tool":
            print(f"[TOOL] Anfrage: {cmd}")
            ask_with_tools(cmd)
            
        elif current_mode == "rag":
            print(f"[RAG] Anfrage: {cmd}")
            ask_rag(cmd)

        else:  # auto
            print(f"[AUTO] Anfrage: {cmd}")
            if any(x in cmd for x in ["git","github","repo","repository", "commit", "issue", "fork", "sterne", "pull request"]):
                print("Tool-Mode")
                return ask_rag(cmd)

            else:
                print("Rag-Mode")
                return ask_with_tools(cmd)


def extract_json(text: str):
    """Versucht, eingebettetes JSON aus einem Text zu extrahieren.
    Gibt den JSON-String zurück oder wirft eine Exception, wenn keins gefunden wird."""
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        print("Fehler "+text)
        raise ValueError("Kein gültiges JSON im Text gefunden.")
    
    json_str = match.group(0)

    try:
        # Prüfen, ob das gefundene JSON tatsächlich gültig ist
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        raise ValueError(f"Ungültiges JSON im Text gefunden: {e}")

def ask_with_tools(question: str):
    """Verarbeitet Fragen, die Tools (z. B. GitHub) benötigen."""
    tool_descriptions = generate_tool_descriptions(TOOLS)

    system_prompt = f"""
                        Du bist ein KI-Assistent mit Zugriff auf externe Tools.
                        Verfügbare Tools:
                        {tool_descriptions}

                        Wenn du erkennst, dass eine der Funktionen gemeint ist(auch bei Tippfehlern oder ähnlichen Formulierungen),
                        fordere die Ausführung eines Tools außschließlich im JSON-Format an:
                        {{
                        "action": "<Funktionsname>",
                        "arguments": {{ "<parameter>": "<wert>", ... }}
                        }}
                        Beispiele:
                        - Für `get_repo_stats`: {{"repo_name": "repo_name"}}
                        - Für `list_user_repos`: {{"username": "user"}}
                        Antworte nur mit JSON, ohne weiteren Text, Markdown oder Erklärung.
                        Wenn kein passendes Tool zu finden ist, gib normalen Text zurück.
                        """
    

    messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    # Modell wählt Tool-Funktion aus
    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages
    )
    content = response["message"]["content"].strip()
    

    try:
        data = json.loads(extract_json(content))
        action = data.get("action")
        args = data.get("arguments", {})

        if action in TOOLS:
            func = TOOLS[action]["function"]
            print(f"\n[Tool-Auswahl] Modell ruft auf: {action} mit Argumenten: {args}\n")
            result = func(**args)
            print("\n--- Ergebnis (Tool) ---\n")
            result_text = format_output(result)
            print(result_text)

            answer_prompt = f"""
                Du bist jetzt im Antwortmodus.

                Analysiere und fasse das Tool-Ergebnis nur anhand der angezeigten Daten zusammen.
                Verwende keine eigenen Zusatzinformationen oder externes Wissen.

                Antworte auf Deutsch in vollständigen, gut lesbaren Sätzen.
                Wenn es sich um viele Daten handelt, kannst du sie analysieren wenn die Frage danach fragt.
                --- TOOL-ERGEBNIS ---
                {result_text}
                --- ENDE ---

                Frage: {question}
                """

            # Modell antwortet auf Grundlage des Tool-Outputs
            final = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": answer_prompt}])
            print("\n--- Antwort (nach Tool-Call) ---\n")
            print(final["message"]["content"])
            return

        else:
            print(f"Unbekannte Aktion: {action}")

    except json.JSONDecodeError:
        print("\n JSON Fehler ---\n")
        print(content)

    print("\n--- Antwort (Text) ---\n")
    print(content)

def ask_rag(question: str):
    """Durchsucht die lokale Wissensdatenbank (Chroma) und fragt das Modell."""

    results = collection.query(
        query_texts=[question],
        n_results=4,
        where=filter_chunks(question),
        include=["documents", "metadatas"]
    )

    if not results["documents"] or not results["documents"][0]:
        print("Keine passenden Informationen gefunden.")
        return

    # Kontext aus besten Treffern
    context = "\n".join(results["documents"][0])

    # Quellenanzeige
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
        "Wenn du im Kontext keine direkte Antwort findest, gib eine plausible Zusammenfassung der gefundenen Inhalte wieder.\n\n"
        f"--- KONTEXT START ---\n{context}\n--- KONTEXT ENDE ---\n\n"
        f"FRAGE: {question}"
    )

    system_content = (
        "Du bist ein technischer Assistent. "
        "Antworte klar, strukturiert und in vollständigen Sätzen. "
        "Sei präzise, aber liefere genügend Kontext, um die Antwort verständlich zu machen. "
        "Antworte ausschließlich auf Deutsch. "
        "Verwende ausschließlich Informationen aus dem gegebenen Kontext."
    )

    term_width = shutil.get_terminal_size((100, 20)).columns
    buf = ""
    print("\n--- Antwort ---\n")

    for chunk in ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ],
        stream=True
    ):
        buf += chunk["message"]["content"]
        if len(buf) > 400 or "\n" in buf:
            parts = buf.split("\n")
            for line in parts[:-1]:
                print(textwrap.fill(line, width=term_width))
            buf = parts[-1]

    if buf:
        print(textwrap.fill(buf, width=term_width))

if __name__ == "__main__":
    print("Standard Rag oderr Tool-Use mit: repo, github, commit, issue, fork, sterne, pull request")
    print("Erstelle bzw. lade Datenbank...")
    index_files()
    show_chunks()

    while True:
        frage = input(f"\n[{current_mode.upper()}] Frage('help' für Hilfe): ")
        handle_command(frage)
        
