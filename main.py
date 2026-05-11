import os
import pandas as pd
from src.fetch_data import fetch_fixtures, fetch_odds, save_fixtures_parquet
from src.preprocess import clean_and_merge
from src.train import train
from src.predict import load_model, predict
from src.telegram_bot import send_message

def main():
    print("Descargando fixtures y odds...")
    fixtures = fetch_fixtures()
    odds = fetch_odds()

    print("Guardando fixtures raw como parquet...")
    save_fixtures_parquet(fixtures)

    print("Procesando y limpiando datos...")
    df_new = clean_and_merge(fixtures, odds)

    os.makedirs('data/processed', exist_ok=True)
    training_data_path = 'data/processed/training_data.csv'

    if os.path.exists(training_data_path):
        df_train = pd.read_csv(training_data_path)
        df_train = pd.concat([df_train, df_new], ignore_index=True).drop_duplicates(subset=['home_team', 'away_team', 'home_goals', 'away_goals'])
    else:
        df_train = df_new

    df_train.to_csv(training_data_path, index=False)

    if not os.path.exists('models/model.pkl') or df_new.shape[0] > 50:
        print("Entrenando modelo...")
        model = train(df_train)
    else:
        print("Cargando modelo entrenado...")
        model = load_model()

    print("Generando predicciones...")
    pred_df = predict(model, df_new)

    pred_path = 'data/processed/predictions.csv'
    pred_df.to_csv(pred_path, index=False)

    msg = "*Predicciones deportivas actuales*\n"
    for _, row in pred_df.head(10).iterrows():
        equipo = row['home_team']
        visitante = row['away_team']
        prob = row['predicted_proba']
        pred = "Victoria local" if row['prediction'] == 1 else "No victoria local"
        msg += f"{equipo} vs {visitante}: {pred} ({prob:.2%})\n"

    print("Enviando predicciones por Telegram...")
    send_message(msg)
    print("Proceso finalizado.")

if __name__ == "__main__":
    main()
