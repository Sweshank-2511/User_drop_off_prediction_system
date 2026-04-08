import sys
import os
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


#  Configuration Class
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Starting model training")

            # 🔶 Define models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier()
            }

            best_model = None
            best_score = 0
            best_model_name = ""

            #  Train and evaluate models
            for name, model in models.items():
                logging.info(f"Training {name}")

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                score = accuracy_score(y_test, y_pred)

                logging.info(f"{name} Accuracy: {score}")
                print(f"{name} Accuracy: {score}")

                # Select best model
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name

            logging.info(f"Best Model: {best_model_name} with accuracy {best_score}")
            print(f"Best Model: {best_model_name}")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Model saved successfully")

            return best_model

        except Exception as e:
            raise CustomException(e, sys)
        
if  __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr, test_arr, _,_= data_transformation.initiate_data_transformation(train_data, test_data)
    
    X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

modeltrainer = ModelTrainer()
print(modeltrainer.initiate_model_trainer(X_train, X_test, y_train, y_test))