from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from data_cleaning import DataCleaning
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd

class LogisticRegressionModel:

    def train_model(self, X_train, y_train):

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        print("Modèle de régression logistique entraîné.")
        return model
    
    def tuning_model(self, X_train, y_train):

        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }

        grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        print(f"Meilleurs paramètres : {grid_search.best_params_}")
        return grid_search.best_estimator_
    
class RandomForestModel:
    def train_model(self, X_train, y_train):

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        print("Modèle de forêt aléatoire entraîné.")
        return model
    
    def tuning_model(self, X_train, y_train):

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"]
        }

        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        print(f"Meilleurs paramètres : {grid_search.best_params_}")
        return grid_search.best_estimator_

class XGBoostModel:
    def train_model(self, X_train, y_train):

        model = XGBClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        print("Modèle XGBoost entraîné.")
        return model
    
    def tuning_model(self, X_train, y_train):

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10, 20],
            'learning_rate': [0.01, 0.1]
        }

        grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        print(f"Meilleurs paramètres : {grid_search.best_params_}")
        return grid_search.best_estimator_      
    
class sauvegarde_model:
   
    def save_model(self, model, file_path):
        try:
            joblib.dump(model, file_path)
            print(f"Modèle sauvegardé dans {file_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du modèle : {e}")
    
if __name__ == "__main__":
    X_train= pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    X_train_scaled = pd.read_csv("data/processed/X_train_scaled.csv")


    model_logistic_regression = LogisticRegressionModel()
    model_logistic_regression.train_model(X_train_scaled, y_train)
    model_logistic_regression.tuning_model(X_train_scaled, y_train)

    model_random_forest = RandomForestModel()
    model_random_forest.train_model(X_train, y_train)
    model_random_forest.tuning_model(X_train, y_train)

    model_xgboost = XGBoostModel()
    model_xgboost.train_model(X_train, y_train)
    model_xgboost.tuning_model(X_train, y_train)

    #Récupérer les meilleurs modèles après tuning
    best_lr  = model_logistic_regression.tuning_model(X_train_scaled, y_train)
    best_rf  = model_random_forest.tuning_model(X_train, y_train)
    best_xgb = model_xgboost.tuning_model(X_train, y_train)
    print("Tuning terminé")

    # Sauvegarde des best modeles
    sauvegarde = sauvegarde_model()
    sauvegarde.save_model(best_lr,  "src/best_logistic_regression_model.pkl")
    sauvegarde.save_model(best_rf,  "src/best_random_forest_model.pkl")
    sauvegarde.save_model(best_xgb, "src/best_xgboost_model.pkl")
    print("Sauvegarde terminée")
