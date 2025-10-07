# Ollama Knowledge Base
## Ziel
Lokaler Agent agiert als Wissensdatenbank mit **Chroma** und **Ollama**.  
Liest automatisch PDFs und Code aus den Ordnern `docs/` und `src/`  
und beantwortet Fragen mit dem Modell **LLaMA 3** mit GPU-Unterstützung.

Unterstützt außerdem einige Github-API Aufrufe wenn Keywords wie 
"repo, github, commit, issue, fork, sterne, pull request" eingegeben werden wechselt es in den Tool-Modus.


## Setup
Ollama Nativ installieren mit beliebigem Model z.B:  
```ollama run llama3```
Cuda installieren für GPU Unterstützung:
```nvidia-smi```
Python-Setup:
```install chromadb ollama PyPDF2```
