import tkinter as tk
from tkinter import messagebox, ttk
import re
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =====================================================================
# CARICAMENTO MODELLI E FUNZIONI LOGICHE (VERSIONE STANDARD NON GERARCHICA)
# =====================================================================
# Carica i modelli creati
m_category = joblib.load('modelli/m_categoria.pkl')
v_category = joblib.load('modelli/v_categoria.pkl')
m_priority = joblib.load('modelli/m_priorità.pkl')
v_priority = joblib.load('modelli/v_priorità.pkl')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_category(ticket):
    ticket_pulito = clean_text(ticket)
    testo_vettorizzato = v_category.transform([ticket_pulito])
    categoria_predetta = m_category.predict(testo_vettorizzato)[0]
    return categoria_predetta

def predict_priority(ticket):
    ticket_pulito = clean_text(ticket)
    testo_vettorizzato = v_priority.transform([ticket_pulito])
    priorità_predetta = m_priority.predict(testo_vettorizzato)[0]
    return priorità_predetta

def get_top_5_words(ticket):
    ticket_pulito = clean_text(ticket)
    # Estraiamo le feature basandoci sul vocabolario del modello di priorità globale
    testo_vettorizzato = v_priority.transform([ticket_pulito])
    feature_index = testo_vettorizzato.nonzero()[1]
    
    # Mappa ogni parola attiva nel testo con il rispettivo peso TF-IDF assegnato
    words_weights = [(v_priority.get_feature_names_out()[idx], testo_vettorizzato.data[i]) for i, idx in enumerate(feature_index)]
    top_5 = sorted(words_weights, key=lambda x: x[1], reverse=True)[:5]
    return top_5

# =====================================================================
# INTERFACCIA GRAFICA PROFESSIONALE (DUE COLONNE)
# =====================================================================
class ProfessionalDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Ticketing Dashboard - Prototipo di Tesi")
        self.root.geometry("1100x600")
        self.root.configure(bg="#f4f6f9")
        
        # Stile globale dei Widget
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TLabel", background="#f4f6f9", font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        
        # --- HEADER SUPERIORE ---
        header = tk.Frame(self.root, bg="#1e293b", height=60)
        header.pack(fill="x", side="top")
        lbl_titolo = tk.Label(header, text="🎫 Smart Ticketing Classifier", fg="white", bg="#1e293b", font=("Segoe UI", 16, "bold"))
        lbl_titolo.pack(side="left", padx=20, pady=15)
        
        lbl_sub = tk.Label(header, text="NLP Pipeline Lineare (LinearSVC)", fg="#94a3b8", bg="#1e293b", font=("Segoe UI", 10, "italic"))
        lbl_sub.pack(side="right", padx=20, pady=20)

        # --- CONTENITORE PRINCIPALE A DUE COLONNE ---
        main_container = tk.Frame(self.root, bg="#f4f6f9")
        main_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Configurazione proporzioni colonne (Colonna 0: Input, Colonna 1: Output/Grafici)
        main_container.columnconfigure(0, weight=1, minsize=450)
        main_container.columnconfigure(1, weight=1, minsize=550)
        main_container.rowconfigure(0, weight=1)
        
        # =====================================================================
        # COLONNA SINISTRA: INPUT UTENTE
        # =====================================================================
        col_sinistra = tk.Frame(main_container, bg="#f4f6f9")
        col_sinistra.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        box_manuale = tk.LabelFrame(col_sinistra, text=" Analisi Singolo Ticket ", bg="white", font=("Segoe UI", 10, "bold"), bd=1, relief="solid")
        box_manuale.pack(fill="both", expand=True, pady=5, padx=5)
        
        tk.Label(box_manuale, text="Oggetto / Titolo del Ticket:", bg="white", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=15, pady=(15, 2))
        self.entry_titolo = ttk.Entry(box_manuale, font=("Segoe UI", 11))
        self.entry_titolo.pack(fill="x", padx=15, pady=5)
        
        tk.Label(box_manuale, text="Corpo del Messaggio:", bg="white", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=15, pady=(10, 2))
        
        # Campo di testo con scrollbar integrata
        text_frame = tk.Frame(box_manuale, bg="white")
        text_frame.pack(fill="both", expand=True, padx=15, pady=5)
        self.text_corpo = tk.Text(text_frame, font=("Segoe UI", 10), bd=1, relief="solid", wrap="word")
        scrollbar = ttk.Scrollbar(text_frame, command=self.text_corpo.yview)
        self.text_corpo.configure(yscrollcommand=scrollbar.set)
        self.text_corpo.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bottone di classificazione istantanea
        btn_classifica = ttk.Button(box_manuale, text="⚡ Esegui Classificazione", command=self.classifica_singolo)
        btn_classifica.pack(anchor="e", padx=15, pady=15)

        # =====================================================================
        # COLONNA DESTRA: RISULTATI (KPI BOXES) & EXPLAINABLE AI
        # =====================================================================
        col_destra = tk.Frame(main_container, bg="#f4f6f9")
        col_destra.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        box_risultati = tk.LabelFrame(col_destra, text=" Dashboard di Output ed Esplicabilità (XAI) ", bg="white", font=("Segoe UI", 10, "bold"), bd=1, relief="solid")
        box_risultati.pack(fill="both", expand=True, padx=5)
        
        # --- SOTTO-PANNELLO DEI KPI BOX (LAYOUT AFFIANCATO) ---
        kpi_frame = tk.Frame(box_risultati, bg="white")
        kpi_frame.pack(fill="x", padx=15, pady=15)
        kpi_frame.columnconfigure(0, weight=1)
        kpi_frame.columnconfigure(1, weight=1)
        
        # Box Categoria (Verde)
        self.card_cat = tk.Frame(kpi_frame, bg="#f0fdf4", bd=1, relief="solid", highlightthickness=2, highlightbackground="#bbf7d0")
        self.card_cat.grid(row=0, column=0, sticky="nsew", padx=(0, 10), ipady=10)
        tk.Label(self.card_cat, text="CATEGORIA PREVISTA", bg="#f0fdf4", fg="#166534", font=("Segoe UI", 9, "bold")).pack(pady=(5, 0))
        self.val_cat = tk.Label(self.card_cat, text="-", bg="#f0fdf4", fg="#15803d", font=("Segoe UI", 16, "bold"))
        self.val_cat.pack(pady=5)
        
        # Box Priorità (Grigio neutro all'avvio)
        self.card_prio = tk.Frame(kpi_frame, bg="#f8fafc", bd=1, relief="solid", highlightthickness=2, highlightbackground="#e2e8f0")
        self.card_prio.grid(row=0, column=1, sticky="nsew", padx=(10, 0), ipady=10)
        tk.Label(self.card_prio, text="PRIORITÀ SUGGERITA", bg="#f8fafc", fg="#475569", font=("Segoe UI", 9, "bold")).pack(pady=(5, 0))
        self.val_prio = tk.Label(self.card_prio, text="-", bg="#f8fafc", fg="#334155", font=("Segoe UI", 16, "bold"))
        self.val_prio.pack(pady=5)
        
        # --- SEZIONE GRAFICO XAI ---
        self.graph_frame = tk.Frame(box_risultati, bg="white")
        self.graph_frame.pack(fill="both", expand=True, padx=15, pady=(10, 15))
        
        self.lbl_placeholder_grafico = tk.Label(self.graph_frame, text="Inserisci un ticket e clicca su Classifica\nper visualizzare l'importanza dei termini.", bg="white", fg="#94a3b8", font=("Segoe UI", 10, "italic"))
        self.lbl_placeholder_grafico.pack(expand=True)
        
        self.canvas_plot = None

    # =====================================================================
    # LOGICA DI AZIONE AL CLICK DEL BOTTONE
    # =====================================================================
    def classifica_singolo(self):
        titolo = self.entry_titolo.get().strip()
        corpo = self.text_corpo.get("1.0", tk.END).strip()
        
        if not titolo or not corpo:
            messagebox.showwarning("Dati Mancanti", "Per favore, compila sia il titolo che il corpo del ticket.")
            return
            
        testo_completo = f"{titolo} {corpo}"
        
        # Richiamo delle tue funzioni adattate senza l'albero gerarchico
        cat_predetta = predict_category(testo_completo)
        prio_predetta = predict_priority(testo_completo)
        top_5_parole = get_top_5_words(testo_completo)
        
        # =====================================================================
        # AGGIORNAMENTO AUTOMATICO DEL LAYOUT GRAFICO
        # =====================================================================
        # 1. Aggiornamento UI Box Categoria
        self.val_cat.config(text=str(cat_predetta).capitalize())
        
        # 2. Aggiornamento UI Box Priorità in base al livello (Colori Semaforici)
        prio_stringa = str(prio_predetta).capitalize()
        if prio_stringa == "Alta":
            self.card_prio.config(bg="#fef2f2", highlightbackground="#fca5a5")
            self.val_prio.config(text=prio_stringa, bg="#fef2f2", fg="#991b1b")
        elif prio_stringa == "Media":
            self.card_prio.config(bg="#fffbeb", highlightbackground="#fde68a")
            self.val_prio.config(text=prio_stringa, bg="#fffbeb", fg="#9a3412")
        else: # Bassa
            self.card_prio.config(bg="#f0fdf4", highlightbackground="#bbf7d0")
            self.val_prio.config(text=prio_stringa, bg="#f0fdf4", fg="#166534")
            
        # 3. Aggiornamento e disegno del Grafico Matplotlib delle parole
        if self.canvas_plot:
            self.canvas_plot.get_tk_widget().destroy()
        self.lbl_placeholder_grafico.pack_forget()
        
        if top_words := top_5_parole:
            parole = [x[0] for x in top_words]
            pesi = [x[1] for x in top_words]
            
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#f8fafc')
            
            ax.barh(parole, pesi, color='#3b82f6', edgecolor='#2563eb', height=0.5)
            ax.invert_yaxis()  # Mette il termine più influente in alto
            ax.set_title("Feature Importance: Top 5 parole influenti della decisione", fontsize=10, fontweight='bold', color='#1e293b', pad=10)
            ax.tick_params(axis='both', labelsize=9, labelcolor='#475569')
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(False)
                
            plt.tight_layout()
            
            self.canvas_plot = FigureCanvasTkAgg(fig, master=self.graph_frame)
            self.canvas_plot.draw()
            self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)
            plt.close()
        else:
            self.lbl_placeholder_grafico.config(text="Nessun termine significativo intercettato nel vocabolario.")
            self.lbl_placeholder_grafico.pack(expand=True)

# =====================================================================
# AVVIO APPLICAZIONE
# =====================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = ProfessionalDashboard(root)
    root.mainloop()