from nicegui import ui
import pandas as pd
import io
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import asyncio
# =====================================================================
#funzioni di supporto e preprocessing
def clean_text(text):
    #funzione di pulizia base (la stessa usata in addestramento)
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_top_words(text, vectorizer, model, top_n=5):
    #trasforma il testo in vettore TF-IDF (array 1D)
    tfidf_matrix = vectorizer.transform([text]).toarray()[0]
    #si controlla che tipo di modello si ha per evitare problemi con strutture diverse
    if hasattr(model, 'coef_'):
        #se è un modello lineare
        #se ha più classi (Tecnico, Commerciale, Amm.), prendiamo i coefficienti della classe predetta
        if model.coef_.ndim == 2:
            classe_predetta = model.predict(vectorizer.transform([text]))[0]
            idx_classe = list(model.classes_).index(classe_predetta)
            pesi_modello = model.coef_[idx_classe]
        else:
            pesi_modello = model.coef_[0]
            
        if hasattr(pesi_modello, 'toarray'):
            pesi_modello = pesi_modello.toarray().flatten()
            
    elif hasattr(model, 'feature_importances_'):
        #se è un modello random forest
        pesi_modello = model.feature_importances_
    else:
        pesi_modello = 1
        
    #moltiplica le parole presenti (TF-IDF) per i pesi reali del modello
    punteggio_combinato = tfidf_matrix * pesi_modello
    
    #ottiene i nomi delle parole/bigrammi dal vocabolario
    feature_names = vectorizer.get_feature_names_out()
    
    #ordina in base al punteggio combinato dal più alto al più basso
    sorted_indices = punteggio_combinato.argsort()[::-1]
    
    #filtra prendendo le prime top_n parole che hanno dato un contributo positivo alla classe
    top_words = [feature_names[i] for i in sorted_indices[:top_n] if punteggio_combinato[i] > 0]
    
    return top_words
# =====================================================================
#caricamento modelli
try:
    vec_categoria = joblib.load('modelli/v_categoria.pkl')
    vec_priorita = joblib.load('modelli/v_priorità.pkl')
    model_cat = joblib.load('modelli/m_categoria.pkl')
    model_pri = joblib.load('modelli/m_priorità.pkl')    
    print("Modelli caricati con successo!")
#in caso di errore (file non trovati), vengono creati dei modelli fittizzi per non creare problemi 
except FileNotFoundError:
    print("Modelli non trovati. Creazione modelli fittizi per non esplodere...")
    dummy_texts = ["il server è offline", "fattura errata", "licenza commerciale bloccante", "router rotto", "sconto"]
    
    vec_categoria = TfidfVectorizer().fit(dummy_texts)
    vec_priorita = TfidfVectorizer().fit(dummy_texts)
    
    model_cat = RandomForestClassifier().fit(vec_categoria.transform(dummy_texts), ["Tecnico", "Amministrazione", "Commerciale", "Tecnico", "Commerciale"])
    model_pri = RandomForestClassifier().fit(vec_priorita.transform(dummy_texts), ["Alta", "Media", "Alta", "Bassa", "Bassa"])

# =====================================================================
#logica dell'analisi del singolo ticket
def analizza_singolo():
    titolo = input_titolo.value
    corpo = input_corpo.value
    
    if not titolo and not corpo:
        ui.notify('Inserisci testo nel ticket!', type='warning')
        return

    #preprocessing
    testo_pulito = clean_text(titolo + " " + corpo)
    
    
    X_cat = vec_categoria.transform([testo_pulito])
    X_pri = vec_priorita.transform([testo_pulito])
    
    #predizioni separate
    categoria = model_cat.predict(X_cat)[0]
    priorita = model_pri.predict(X_pri)[0]
    
    
    #aggiornamento UI
    lbl_categoria.set_text(categoria.upper())
    lbl_priorita.set_text(priorita.upper())

    parole_cat = get_top_words(testo_pulito, vec_categoria, model_cat)
    parole_pri = get_top_words(testo_pulito, vec_priorita, model_pri)
    #aggiorna la lista delle parole influenti con due colori distinti (in base al gruppo di appartenenza)
    container_parole.clear()
    with container_parole:
        for p in parole_cat:
            ui.chip(p, color='primary', text_color='white')
            
        for p in parole_pri:
            ui.chip(p, color='orange-7', text_color='white')
            
    ui.notify('Analisi completata!', type='positive')
    
#analisi del batch dei dati
async def processa_batch(e):
    try:
        file_obj = getattr(e, 'content', getattr(e, 'file', None))
        #nel caso in cui non viene trovato il file
        if not file_obj:
            ui.notify('Errore: impossibile trovare i dati del file caricato.', type='negative')
            return
        
        #lettura del file
        dati_file = file_obj.read()
        
        if asyncio.iscoroutine(dati_file):
            dati_file = await dati_file
            
        #legge il file caricato
        df = pd.read_csv(io.BytesIO(dati_file))
        #nel caso di file nel formato errato
        if 'title' not in df.columns or 'body' not in df.columns:
            ui.notify('Il CSV deve contenere le colonne "title" e "body"', type='negative')
            return
        #in caso positivo
        ui.notify('Elaborazione batch in corso...', type='info')
        
        #preprocessing testuale
        df['testo_pulito'] = (df['title'].fillna('') + " " + df['body'].fillna('')).apply(clean_text)
        
        X_cat = vec_categoria.transform(df['testo_pulito'])
        X_pri = vec_priorita.transform(df['testo_pulito'])
        
        df['Categoria_Prevista'] = model_cat.predict(X_cat)
        df['Priorita_Prevista'] = model_pri.predict(X_pri)
        
        #rimuove la colonna utilizzata per l'elaborazione
        df = df.drop(columns=['testo_pulito'])
        
        #salva in locale e avvia il download
        output_file = 'ticket_predetti_batch.csv'
        df.to_csv(output_file, index=False)
        ui.download(output_file)
        
        ui.notify('Batch elaborato e scaricato!', type='positive')
    except Exception as ex:
        ui.notify(f'Errore: {str(ex)}', type='negative')

# =====================================================================
#interfaccia grafica

#cambio colori principali e font utilizzato
ui.colors(primary='#2ca25f', secondary='#99d8c9', accent='#e5f5e0')
#importa il font da Google Fonts
ui.add_head_html('<link href="https://fonts.googleapis.com/css2?family=Arimo:ital,wght@0,400..700;1,400..700&family=Fira+Sans:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&family=Unica+One&display=swap" rel="stylesheet">')

ui.add_css('''
    body {
        font-family: "Arimo", sans-serif;
    }
''')
#importo le icon
ui.add_head_html('<link href="https://cdn.jsdelivr.net/themify-icons/0.1.2/css/themify-icons.css" rel="stylesheet" />')


#header
with ui.header(elevated=True).style('background-color: #006d2c'):
    ui.label('Smart Ticketing').classes('text-2xl font-bold p-4 text-white w-full text-center')

#tabs
with ui.tabs().classes('w-full') as tabs:
    tab_singolo = ui.tab('Analisi Ticket', icon='bar_chart')
    tab_batch = ui.tab('Elaborazione Batch', icon='cloud_upload')

#panels
with ui.tab_panels(tabs, value=tab_singolo).classes('w-full max-w-4xl mx-auto mt-8'):
    
    #tab per analisi ticket
    with ui.tab_panel(tab_singolo):
        with ui.row().classes('w-full gap-8'):
            
            #form di Input
            with ui.card().classes('w-1/2 p-6 border-t-4').style('border-color: #2ca25f'):
                ui.label('Ticket:').classes('text-xl font-bold mb-4').style('color: #006d2c')
                input_titolo = ui.input(label='Oggetto (Titolo)').classes('w-full mb-2')
                input_corpo = ui.textarea(label='Corpo del Ticket (Descrizione del problema)').classes('w-full mb-4')
                ui.button('Analizza Ticket', on_click=analizza_singolo).classes('w-full mt-4')
            
            #pannello di output
            with ui.card().classes('w-2/5 p-6 bg-gray-50 border-t-4').style('border-color: #99d8c9'):
                ui.label('Risultato analisi del ticket').classes('text-xl font-bold mb-4').style('color: #006d2c')
                
                ui.label('Categoria prevista:').classes('text-sm text-gray-500')
                lbl_categoria = ui.label('-').classes('text-2xl font-bold mb-4 text-primary')
                
                ui.label('Priorità suggerita:').classes('text-sm text-gray-500')
                lbl_priorita = ui.label('-').classes('text-2xl font-bold mb-4').style('color: #00441b')
                
                ui.label('Top 5 parole più influenti (categoria e priorità):').classes('text-sm text-gray-500 mb-2')
                container_parole = ui.row().classes('w-full gap-2')

    #tab per analisi batch
    with ui.tab_panel(tab_batch):
        with ui.card().classes('w-full p-8 items-center border-t-4').style('border-color: #2ca25f'):
            ui.icon('cloud_upload', size='4rem').style('color: #99d8c9').classes('mb-4')
            ui.label('Carica un file CSV per processare multipli ticket contemporaneamente').classes('text-lg mb-4 text-gray-800')
            ui.label('IMPORTANTE: Il file deve contenere almeno le colonne "title" e "body".').classes('text-sm text-red-500 mb-6')
            ui.upload(label='Trascina qui il CSV o clicca il tasto +', 
                      auto_upload=True, 
                      on_upload=processa_batch).classes('w-full max-w-md')

ui.run(title="Smart Ticketing", port=8080)