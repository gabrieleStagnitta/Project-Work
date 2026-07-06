# Project Work: Triage Automatico dei Ticket aziendali con Machine Learning

Benvenuti nel repository del Project Work dedicato allo sviluppo di un sistema end-to-end di **Natural Language Processing (NLP)** e **Machine Learning** per l'automazione del triage del supporto clienti. Creato da Gabriele Stagnitta (matricola 0312401645).

Il progetto affronta l'intero ciclo di vita del dato in **3 fasi sequenziali**: dalla generazione quantitativa del dataset testuale, passando per la costruzione e valutazione della pipeline di Machine Learning, fino alla messa in produzione tramite una Web Application interattiva dotata di funzionalità di **Explainable AI (XAI)**.

---

## Architettura del Progetto

Il repository è organizzato in tre moduli principali numerati in base al flusso logico di elaborazione:

```text
📦 Project-Work
 ┣ 📂 01. Generatore di dataset/    # Creazione e simulazione dei dati di supporto
 ┣ 📂 02. Pipeline ML/              # Preprocessing, training, valutazione e salvataggio modelli
 ┗ 📂 03. Dashboard/                # Web Application interattiva (NiceGUI) e Explainability
```
---
## Dettagli dei moduli

### Generatore di dataset
La prima fase del progetto si concentra sulla creazione di una base dati di test strutturata e realistica, fondamentale per addestrare i modelli predittivi in assenza di uno storico aziendale accessibile.
All'interno di questa cartella sono presenti due Notebook Jupyter dedicati:
- **Generazione Dataset Elementare**
  - Si crea in maniera pseudocasuale un dataset utilizzando un vocabolario ed estraendo casualmente le parole da utilizzare per formare una frase
- **Generazione Dataset con Gemma**
  - Si utilizza *Gemma4:e2b* (modello di Intelligenza Artificiale open-source sviluppata da Google) per generare un dataset creando in un ciclo un prompt con parametri casuali che chieda allo LLM di scrivere un ticket con quelle caratteristiche.
### Pipeline ML
Il cuore del progetto. Qui risiede il notebook che implementa la pipeline completa di Natural Language Processing e Machine Learning:

- Preprocessing Testuale: Pulizia delle stringhe tramite Regular Expression (rimozione caratteri speciali, normalizzazione in lower-case, rimozione spazi ridondanti).

- Feature Extraction (TF-IDF): Trasformazione del testo in matrici numeriche tramite TfidfVectorizer, con analisi di unigrammi e bigrammi per catturare espressioni composte di business (es. "rinnovo contratto", "errore server").

- Addestramento e Valutazione: Analisi comparativa tra diversi algoritmi di classificazione supervisionata (es. Random Forest Classifier e LinearSVC).

- Serializzazione dei Modelli: Selezione e salvataggio (tramite joblib) dei migliori modelli predittivi per le due task di triage parallele:
  - v_categoria.pkl & m_categoria.pkl (Vettorizzatore e Modello per la Categoria)
  - v_priorita.pkl & m_priorita.pkl (Vettorizzatore e Modello per la Priorità)
### Dashboard
L'ultima sezione è impiegata per il deployment del sistema attraverso un'interfaccia grafica moderna, reattiva e accessibile a dei potenziali operatori aziendali, realizzata in Python tramite il framework NiceGUI (basato su Quasar e Tailwind CSS).
- Analisi Singolo Ticket in Real-Time: Inserendo un ticket nell'interfaccia, il sistema calcola istantaneamente la Categoria e la Priorità suggerita.
  - Explainable AI: L'interfaccia non si limita alla predizione "black-box", ma interroga direttamente le Feature Importances e i coefficienti matematici del modello per mostrare a schermo le Top 5 Parole più influenti della decisione.
  - Classifica Mista e Color-Coding: I chip delle parole sono colorati dinamicamente per specificare se il termine ha influenzato maggiormente la Categoria (verde brillante) o la Priorità (arancione), con tooltip interattivi.
- Elaborazione Batch (CSV): Sezione dedicata all'upload massivo di file .csv non classificati. La dashboard processa l'intero file in background e restituisce il download istantaneo di un nuovo CSV arricchito con le colonne delle predizioni.
