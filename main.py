from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, auc
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def preprocess_data(dataframe = pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Process the input DataFrame to prepare for machine learning training.

    Parameters:
    - data (DataFrame): The input data containing the features and label.

    Returns:
    - Tuple[DataFrame, pd.Series]: A tuple where the first element is the DataFrame of features and
      the second element is the Series containing the labels.
    """

    # Handling missing values if necessary
    dataframe.fillna(method = 'ffill', inplace = True)

    # Feature Scaling for 'Amount' and 'Time'
    scaler = StandardScaler()
    dataframe['NormalizedAmount'] = scaler.fit_transform(dataframe['Amount'].values.reshape(-1, 1))
    dataframe['NormalizedTime'] = scaler.fit_transform(dataframe['Time'].values.reshape(-1, 1))

    # Drop the original 'Time' and 'Amount' columns
    data = dataframe.drop(['Time', 'Amount'], axis=1)

    # Splitting the data into features (X) and labels (y)
    X = dataframe.drop('Class', axis=1)
    y = data['Class']

    return X, y

def evaluate_model(model: BaseEstimator, y_test: pd.Series, y_pred: np.ndarray, X_test: pd.DataFrame) -> None:
    
    # Evaluating the model
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
    pr_auc = auc(recall, precision)
    print("AUC-PR:", pr_auc)

    plt.figure()
    plt.plot(recall, precision, label='AUC-PR curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

def main() -> None:
    dataframe = pd.read_csv("FinalProj/creditcard.csv")

    # Preprocess the data using the preprocess function
    X, y = preprocess_data(dataframe)

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y) 

    # Creating the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Training the model
    model.fit(X_train, y_train) 

    # Predicting the Test set results
    y_pred = model.predict(X_test)

    # Evaluating our model
    evaluate_model(model, y_test, y_pred, X_test)  


if __name__ == "__main__":
    main()
    
