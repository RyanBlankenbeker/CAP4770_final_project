# main.py
# CAP 4770 Final Project – Flight Delay Prediction
# Group: Ryan Blankenbeker, Kenyen Hast-Otero, John WG Wallace

from data_loader import load_flight_data
from preprocessing import preprocess_data, create_delay_label
from plots import run_eda
from models import train_models, evaluate_models

def main():

    print("\n=== LOADING DATA ===")
    df = load_flight_data("flight_delays_2024.csv")

    print("\n=== PREPROCESSING ===")
    df = preprocess_data(df)
    df = create_delay_label(df)

    print("\n=== EXPLORATORY DATA ANALYSIS (EDA) ===")
    run_eda(df)

    print("\n=== TRAINING MODELS ===")
    models, X_test, y_test = train_models(df)

    print("\n=== EVALUATION ===")
    evaluate_models(models, X_test, y_test)

if __name__ == "__main__":
    main()
