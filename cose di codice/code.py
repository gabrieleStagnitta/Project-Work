import random
import pandas as pd

# --- CARICAMENTO E PULIZIA DATI ---
# Nota: Assicurati che i file 'nomi_f.csv', 'nomi_m.csv' e 'cognomi.csv' siano nella stessa cartella
dfFemale = pd.read_csv(
    "nomi_f.csv", comment="#", header=None, names=["Nome", "Rarità"]
)
dfMale = pd.read_csv(
    "nomi_m.csv", comment="#", header=None, names=["Nome", "Rarità"]
)
dfSurname = pd.read_csv("cognomi.csv", comment="#", header=None, names=["Cognome"])
df_nomi = pd.concat([dfFemale, dfMale], ignore_index=True)

categorie_nomi = [1, 2, 3, 4]
pesi_categorie_nomi = [0.85, 0.11, 0.03, 0.01]
domini = ["gmail.com", "outlook.it", "hotmail.com", "libero.it", "yahoo.com"]
pesi_domini = [0.55, 0.15, 0.13, 0.13, 0.04]


def genera_anno_realistico():
    anno = random.randint(1965, 2007)
    return str(anno)[2:]


# =====================================================================
# STEP 1: GENERIAMO IL POOL DI 125 UTENTI UNICI (Con Nome ed Email)
# =====================================================================
numero_utenti_unici = 125
pool_utenti = []

for _ in range(numero_utenti_unici):
    cat_nome = random.choices(categorie_nomi, weights=pesi_categorie_nomi)[0]
    df_nomi_filtrato = df_nomi[df_nomi["Rarità"] == cat_nome]
    if df_nomi_filtrato.empty:
        df_nomi_filtrato = df_nomi

    # Conserviamo una versione "pulita" con le iniziali maiuscole
    nome_reale = df_nomi_filtrato["Nome"].sample(n=1).values[0].title().strip()
    cognome_reale = dfSurname["Cognome"].sample(n=1).values[0].title().strip()
    nome_completo = f"{nome_reale} {cognome_reale}"

    # Versione minuscola e senza spazi per l'indirizzo email
    nome_email = nome_reale.lower().replace(" ", "").replace("'", "")
    cognome_email = cognome_reale.lower().replace(" ", "").replace("'", "")

    dominio = random.choices(domini, weights=pesi_domini)[0]
    anno_nascita = genera_anno_realistico()

    formati = [
        f"{nome_email}.{cognome_email}",
        f"{nome_email}.{cognome_email}{anno_nascita}",
        f"{nome_email[0]}.{cognome_email}",
        f"{nome_email[0]}.{cognome_email}{anno_nascita}",
        f"{cognome_email}.{nome_email}",
        f"{nome_email}_{cognome_email}",
        f"{nome_email}{cognome_email}{anno_nascita}",
    ]
    pesi_formati = [0.20, 0.40, 0.10, 0.10, 0.10, 0.05, 0.05]
    struttura_email = random.choices(formati, weights=pesi_formati)[0]
    email_finale = f"{struttura_email}@{dominio}"

    pool_utenti.append({"user_name": nome_completo, "user_email": email_finale})


# =====================================================================
# STEP 2: FUNZIONE FUNZIONE GENERAZIONE TICKET INTEGRATA CON UTENTI
# =====================================================================
def genera_ticket_sintetici_con_utenti(pool_utenti, num_ticket=300):
    # Definiamo il lessico tipico per ogni categoria
    lessico = {
        "Amministrazione": {
            "verbi": [
                "sollecitare",
                "verificare",
                "restituire",
                "rettificare",
                "inviare",
            ],
            "sostantivi": [
                "la fattura",
                "il pagamento",
                "la nota di credito",
                "lo storno",
                "il rimborso",
            ],
            "dettagli": [
                "del mese scorso",
                "per il servizio cloud",
                "con importo errato",
                "non ancora pervenuto",
            ],
        },
        "Tecnico": {
            "verbi": [
                "riscontrare",
                "segnalare",
                "risolvere",
                "riavviare",
                "ottimizzare",
            ],
            "sostantivi": [
                "un errore 500",
                "un bug bloccante",
                "il crash dell'app",
                "il server offline",
                "il database lento",
            ],
            "dettagli": [
                "dopo l'ultimo aggiornamento",
                "sulla macchina di produzione",
                "durante il login",
                "in fase di caricamento",
            ],
        },
        "Commerciale": {
            "verbi": [
                "richiedere",
                "valutare",
                "attivare",
                "rinnovare",
                "modificare",
            ],
            "sostantivi": [
                "un preventivo",
                "lo sconto commerciale",
                "il piano di abbonamento",
                "la licenza aggiuntiva",
                "le condizioni contrattuali",
            ],
            "dettagli": [
                "per la nuova filiale",
                "in vista della scadenza",
                "per il pacchetto Enterprise",
                "per un potenziale cliente",
            ],
        },
    }

    categorie = list(lessico.keys())
    dataset = []

    for i in range(1, num_ticket + 1):
        # 2a. Scegliamo un utente casuale dal pool (generando ripetizioni realistiche)
        utente_estratto = random.choice(pool_utenti)

        # 2b. Generiamo i dettagli del ticket
        cat = random.choice(categorie)
        v = random.choice(lessico[cat]["verbi"])
        s = random.choice(lessico[cat]["sostantivi"])
        d = random.choice(lessico[cat]["dettagli"])

        title = f"{v.capitalize()} {s}"
        body = f"Salve, si richiede di {v} {s} {d}. Restiamo in attesa di un riscontro urgente."

        # Logica di assegnazione priorità
        testo_completo = (title + " " + body).lower()
        if any(w in testo_completo for w in ["bloccante", "offline", "crash"]):
            priority = "Alta"
        elif any(w in testo_completo for w in ["errore", "errato", "sollecitare"]):
            priority = "Media"
        else:
            priority = "Bassa"

        # 2c. Uniamo i dati dell'utente e del ticket nello stesso record
        dataset.append(
            {
                "id": f"TKT-{i:04d}",
                "user_name": utente_estratto["user_name"],
                "user_email": utente_estratto["user_email"],
                "title": title,
                "body": body,
                "category": cat,
                "priority": priority,
            }
        )

    # 3. Trasformazione in DataFrame ed Esportazione in un unico file CSV
    df = pd.DataFrame(dataset)
    df.to_csv("ticket_utenti_simulati.csv", index=False, encoding="utf-8-sig")

    print(
        f"🎉 Successo! Creato il file 'ticket_utenti_simulati.csv' con {num_ticket} righe totali."
    )


# =====================================================================
# ESECUZIONE
# =====================================================================
genera_ticket_sintetici_con_utenti(pool_utenti, num_ticket=200)