import pandas as pd
dataset = "ticket_utenti_simulati.csv"
df = pd.read_csv(dataset, lineterminator='\n')
df.sample(10)