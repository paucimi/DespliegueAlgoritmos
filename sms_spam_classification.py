# sms_spam_classification.py

import pandas as pd
import re
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import argparse

def cargar_y_limpiar_datos(ruta_archivo):
    """
    Carga el archivo CSV y realiza una limpieza básica del texto.
    """
    df = pd.read_csv(ruta_archivo, encoding="latin-1")
    df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df[["label", "text"]]

    def limpiar_texto(texto):
        texto = texto.lower()
        texto = re.sub(r"[^a-zA-Z\s]", "", texto)
        return texto
    #Generamos la columna text_clean
    df["text_clean"] = df["text"].apply(limpiar_texto)
    #Forzamos a cadena, en caso de que haya valores no textuales
    df["text_clean"] = df["text_clean"].astype(str)

    return df

def entrenar_y_evaluar_modelo(df, n_estimators):
    """
    Entrena un modelo de Random Forest y lo evalúa.
    """
    # Dividir los datos
    X = df["text_clean"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Crear el pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=42))
    ])

    # Entrenar el modelo
    pipeline.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]

    # Mostrar resultados
    print(f"\n=== Resultados para n_estimators={n_estimators} ===")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Retornamos también X_test para usarlo al generar la firma
    return pipeline, precision, recall, f1, X_test

def guardar_modelo_en_mlflow(pipeline, X_test, n_estimators, precision, recall, f1):
    """
    Guardo el modelo y las métricas en MLflow, incluyendo un input_example y signature.
    """
    with mlflow.start_run():
        # Log de parámetros y métricas
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Crear un ejemplo de entrada (una sola muestra del X_test).
        # Asumiendo que X_test es un pd.Series de texto.
        input_example = [X_test.iloc[0]]  # lista con 1 SMS de ejemplo
        prediction_example = pipeline.predict(input_example)

        # Inferir la firma del modelo
        signature = infer_signature(input_example, prediction_example)

        # Registrar el modelo con la firma y el ejemplo de entrada
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path=f"modelo_sms_spam_rf_n{n_estimators}",
            input_example=input_example,
            signature=signature
        )
        print(f"Modelo y métricas registrados en MLflow para n_estimators={n_estimators}.")

def main():
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description="Clasificación de SMS Spam usando Random Forest.")
    parser.add_argument("--ruta_archivo", type=str, required=True, help="Ruta del archivo CSV con los datos")
    parser.add_argument("--n_estimators", type=int, default=100, help="Número de estimadores para Random Forest")
    
    # Parsear los argumentos
    args = parser.parse_args()

    # Cargar y limpiar los datos
    df = cargar_y_limpiar_datos(args.ruta_archivo)

    # Entrenar y evaluar el modelo
    pipeline, precision, recall, f1, X_test = entrenar_y_evaluar_modelo(df, args.n_estimators)

    # Guardar el modelo en MLflow con firma y ejemplo de entrada
    guardar_modelo_en_mlflow(pipeline, X_test, args.n_estimators, precision, recall, f1)

# Ejecutar el bloque main si el script se ejecuta directamente
if __name__ == "__main__":
    main()