# Project Work - Machine Learning applicato allo Smart Ticketing: dalla generazione dati tramite LLM allo smistamento automatico dei ticket 

Project Work di Stagnitta Gabriele (mat. 0312401645) dedicato allo sviluppo di un sistema end-to-end di **Natural Language Processing (NLP)** e **Machine Learning** per l'automazione del triage di ticket aziendali con Machine Learning.

Il progetto affronta l'intero ciclo di vita del dato in **3 fasi sequenziali**: dalla generazione di dataset artificiali (partendo da algoritmi elementari ed arrivando a generazioni con LLM utilizzando prompt parametrici), passando per la costruzione e valutazione della pipeline di Machine Learning, fino alla messa in produzione tramite una Web Application interattiva dotata di funzionalità di **Explainable AI (XAI)**.

---

## Architettura del Progetto

Il repository è così organizzato:

```text
📦 Project-Work
 ┣ 📂 dataset/            # Al suo interno sono presenti tutti i dataset generati
 ┣ 📂 elaborato/          # Al suo interno sono presente l'elaborato in versione PDF. Nello specifico, oltre la versione consegnata, è presente una versione realizzata in Light Mode
 ┣ 📂 input/              # File utilizzati per la generazione elementare + batch di esempio per elaborazione ticket
 ┣ 📂 modelli/            # Modelli e Vettorizzatori utili al funzionamento della dashboard realizzati durante la Pipeline ML
 ┣ 🗄️01_Generazione_Dataset_Elementare.ipynb
 ┣ 🗄️02_Generazione_Dataset_Avanzato_Con_Gemma.ipynb
 ┣ 🗄️03_Pipeline_Machine_Learning.ipynb
 ┗ 🗄️04_Dashboard.py
```
---
## Dettagli del progetto

### Generatore di dataset
La prima fase del progetto si basa su diversi approcci per la creazione di dataset strutturati e realistici, fondamentali per addestrare i modelli predittivi.
All'interno della repo sono presenti due Notebook Jupyter dedicati:
- **01 Generazione Dataset Elementare**
  - Crea in maniera pseudocasuale un dataset utilizzando un vocabolario ed estraendo casualmente le parole da utilizzare per formare una frase
- **02 Generazione Dataset Avanzato con Gemma**
  - Utilizzando *Gemma4:e2b* (modello di Intelligenza Artificiale open-source sviluppata da Google), genera un dataset creando in un ciclo un prompt con parametri casuali che chiede allo LLM di scrivere un ticket con quelle caratteristiche.
### Pipeline ML
Il cuore del progetto. Qui risiede il notebook che implementa la pipeline completa di Natural Language Processing e Machine Learning:

- Preprocessing Testuale: Pulizia delle stringhe tramite Regex.

- Feature Extraction (TF-IDF): Trasformazione del testo in matrici numeriche tramite TfidfVectorizer.

- Addestramento e Valutazione: Analisi comparativa tra diversi algoritmi di classificazione supervisionata (Random Forest Classifier e LinearSVC) con un approccio gerarchico per la valutazione della priorità.

- Serializzazione dei Modelli: Selezione e salvataggio (tramite joblib) dei migliori modelli predittivi per la dashboard. Verranno conservati dentro la cartella *modelli*
  - v_categoria.pkl & m_categoria.pkl (Vettorizzatore e Modello per la Categoria)
  - v_priorità.pkl & m_priorità.pkl (Vettorizzatore e Modello per la Priorità)
### Dashboard
L'ultima sezione è impiegata per il deployment del sistema attraverso un'interfaccia grafica moderna, reattiva e accessibile a dei potenziali operatori aziendali, realizzata in Python tramite il framework NiceGUI (basato su Quasar e Tailwind CSS). Vi è la possibilità di vederla all'opera al seguente link: https://project-work-vdxq.onrender.com
- Analisi Singolo Ticket in Real-Time
- Elaborazione Batch (CSV)