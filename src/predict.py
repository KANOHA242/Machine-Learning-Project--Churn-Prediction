from train import LogisticRegressionModel, RandomForestModel, XGBoostModel, sauvegarde_model
from data_cleaning import DataCleaning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
class Predict:
    def load_model(self, file_path):
        try:
            import joblib
            model = joblib.load(file_path)
            print(f"Modèle chargé depuis {file_path}")
            return model
        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")
            return None

    def predict(self, model, X):
        try:
            predictions = model.predict(X)
            print("Prédictions effectuées.")
            return predictions
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            return None


    def evaluate_metrics(self, y_true, y_pred):
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

        except Exception as e:
            print(f"Erreur lors de l'évaluation des métriques : {e}")
    
    def save_predictions(self, predictions, file_path):
        try:
            import pandas as pd
            pd.DataFrame(predictions, columns=["Predictions"]).to_csv(file_path, index=False)
            print(f"Prédictions sauvegardées dans {file_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des prédictions : {e}")

if __name__ == "__main__":
 
    predict = Predict()
    #Importation des modèles
    model_random_forest = predict.load_model("src/best_random_forest_model.pkl")
    model_logistic_regression = predict.load_model("src/best_logistic_regression_model.pkl")
    model_xgboost = predict.load_model("src/best_xgboost_model.pkl")

    #Importation des données de test
    X_test  = pd.read_csv("data/processed/X_test.csv")
    X_test_scaled = pd.read_csv("data/processed/X_test_scaled.csv")
    y_test  = pd.read_csv("data/processed/y_test.csv").values.ravel()

    #Prédictions avec les modèles
    predictions_random_forest = predict.predict(model_random_forest, X_test)
    predictions_xgboost = predict.predict(model_xgboost, X_test)
    predictions_logistic_regression = predict.predict(model_logistic_regression, X_test_scaled)

    #Evaluation des métriques avec les modèles
    print("Métriques du modèle Random Forest :")
    predict.evaluate_metrics(y_test, predictions_random_forest)
    print("Métriques du modèle XGBoost :")
    predict.evaluate_metrics(y_test, predictions_xgboost)
    print("Métriques du modèle Logistic Regression :")
    predict.evaluate_metrics(y_test, predictions_logistic_regression)

    #predict.save_predictions(predictions, "src/predictions.csv")