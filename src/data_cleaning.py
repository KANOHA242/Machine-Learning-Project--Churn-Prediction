import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class DataCleaning:

    def load_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            print("Données chargées avec succès.")
            return data
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            return None


    def clean_data(self, data):

        print("Valeurs manquantes par colonne :")
        print(data.isna().sum())

        # convertir TotalCharges en numérique
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

        # supprimer les NaN
        data = data.dropna()

        print("NaN restants :", data.isna().sum().sum())

        # supprimer l'identifiant
        if "customerID" in data.columns:
            data = data.drop(columns=["customerID"])

        print("Dataset nettoyé.")
        return data
    
    def delete_features(self, data, features_to_delete):

        data = data.drop(columns=[col for col in features_to_delete if col in data.columns])

        print(f"Features supprimées : {features_to_delete}")

        return data

    def encoding(self, data):

        # One Hot Encoding
        data = pd.get_dummies(data, drop_first=True)

        print("Encodage terminé.")
        return data


    def separation(self, data):

        X = data.drop("Churn_Yes", axis=1)
        y = data["Churn_Yes"].values.ravel()

        print("Séparation X / y réussie.")
        return X, y


    def split_data(self, X, y, test_size=0.2, random_state=42):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print("Train/Test split effectué.")
        return X_train, X_test, y_train, y_test


    def equilibration(self, X_train, y_train):

        smote = SMOTE(random_state=42)

        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        print("Équilibrage SMOTE effectué.")

        return X_train_resampled, y_train_resampled


    def standardisation(self, X_train, X_test):

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        print("Standardisation effectuée.")

        return X_train_scaled, X_test_scaled


    def sauvegarde_csv(self, data, file_path):

        try:
            if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
                data.to_csv(file_path, index=False)
            else:
                pd.DataFrame(data).to_csv(file_path, index=False)

            print(f"Données sauvegardées dans {file_path}")

        except Exception as e:
            print(f"Erreur lors de la sauvegarde : {e}")


if __name__ == "__main__":

    cleaner = DataCleaning()

    # 1 Charger
    data = cleaner.load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    if data is not None:

        # 2 Nettoyage
        data = cleaner.clean_data(data)

         # 3 Suppression des features inutiles
        features_to_drop = [
            "gender",
            "PhoneService",
            "StreamingTV",
            "StreamingMovies",
            "TotalCharges"
        ]

        data = cleaner.delete_features(data, features_to_drop)

        # 4 Encodage
        data = cleaner.encoding(data)

        # 5 Séparation
        X, y = cleaner.separation(data)

        # 6 Split
        X_train, X_test, y_train, y_test = cleaner.split_data(X, y)

        # 7 SMOTE (train uniquement)
        X_train, y_train = cleaner.equilibration(X_train, y_train)

        # 8 Standardisation
        X_train_scaled, X_test_scaled = cleaner.standardisation(X_train, X_test)

        # 9 Sauvegarde
        cleaner.sauvegarde_csv(X_train, "data/processed/X_train.csv")
        cleaner.sauvegarde_csv(X_test, "data/processed/X_test.csv")
        cleaner.sauvegarde_csv(y_train, "data/processed/y_train.csv")
        cleaner.sauvegarde_csv(y_test, "data/processed/y_test.csv")
        cleaner.sauvegarde_csv(X_train_scaled, "data/processed/X_train_scaled.csv")
        cleaner.sauvegarde_csv(X_test_scaled, "data/processed/X_test_scaled.csv")