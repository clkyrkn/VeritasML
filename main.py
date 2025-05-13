from src.data_loader import load_data, clean_data
from src.features import preprocess_text, vectorize_text
from src.train import train_model
from src.evaluate import evaluate_model
import pandas as pd

def main():
    print("Uploading data...")
    df = load_data("data/raw/news.csv")
    
    print("Cleaning operations...")
    df = clean_data(df)
    
    print("Pre-processing of texts...")
    df['text'] = df['text'].apply(preprocess_text)

    print("TF-IDF vectorization...")
    X, vectorizer = vectorize_text(df['text'])
    y = df['label']

    print("The model is being trained...")
    model, X_val, y_val = train_model(X, y)

    print("The model is being evaluated...")
    evaluate_model(model, X_val, y_val)

    print("Completed!")

if __name__ == "__main__":
    main()