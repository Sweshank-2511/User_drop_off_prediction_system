import sys
import os
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# Configuration Class
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    #  Create preprocessing pipeline
    def get_data_transformer_object(self):
        try:
            logging.info("Creating preprocessing pipeline")

            # 🔶 Numerical Features (from YOUR dataset)
            numerical_features = [
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]

            # Categorical Features 
            categorical_features = [
                "Contract",
                "InternetService",
                "OnlineSecurity",
                "TechSupport"
            ]

            #  Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            # 🔶 Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
                ]
            )

            #  Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            logging.info("Preprocessing pipeline created successfully")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # Apply transformation
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation")

            #  Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded")

            #  Convert target variable
            train_df["Churn"] = train_df["Churn"].map({"Yes": 1, "No": 0})
            test_df["Churn"] = test_df["Churn"].map({"Yes": 1, "No": 0})

            #  Define target
            target_column = "Churn"

            #  Split features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info("Split into input and target features")

            # Get preprocessor
            preprocessor = self.get_data_transformer_object()

            # Fit + Transform
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Data transformation completed")

            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Preprocessor saved successfully")

            return X_train_transformed, X_test_transformed, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)