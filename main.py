from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, auc
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def preprocess_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Process the input DataFrame to prepare for machine learning training.

    Parameters:
    - dataframe (DataFrame): The input data containing the features and label.

    Returns:
    - Tuple[DataFrame, pd.Series]: A tuple where the first element is the DataFrame of features and
      the second element is the Series containing the labels.
    """
    # Handling missing values if necessary
    dataframe.fillna(method='ffill', inplace=True)

    # Feature Scaling for 'Amount' and 'Time'
    scaler = StandardScaler()
    dataframe['NormalizedAmount'] = scaler.fit_transform(dataframe['Amount'].values.reshape(-1, 1))
    dataframe['NormalizedTime'] = scaler.fit_transform(dataframe['Time'].values.reshape(-1, 1))

    # Drop the original 'Time' and 'Amount' columns
    data = dataframe.drop(['Time', 'Amount'], axis=1)

    # Splitting the data into features (X) and labels (y)
    X = data.drop('Class', axis=1)
    y = data['Class']

    return X, y

def evaluate_model(model: BaseEstimator, y_test: pd.Series, y_pred: np.ndarray, X_test: pd.DataFrame) -> None:
    """
    Evaluate the performance of a machine learning model on the test set and plot a Precision-Recall curve.

    Parameters:
    - model (BaseEstimator): The trained machine learning model to be evaluated.
    - y_test (pd.Series): The true labels (ground truth) for the test set.
    - y_pred (np.ndarray): The predicted labels or probabilities for the test set.
    - X_test (pd.DataFrame): The feature matrix of the test set.

    Returns:
    - None

    This function prints out the confusion matrix and gives a window that provides a plot of the data

    This function evaluates the performance of the specified model on the provided test set by printing
    a confusion matrix, classification report, and accuracy score. Additionally, it calculates the 
    Area Under the Precision-Recall Curve (AUC-PR) and plots the Precision-Recall curve.

    The Precision-Recall curve is particularly useful for evaluating models on imbalanced datasets
    like fraud detection, where traditional metrics such as accuracy may be misleading.

    Example usage:
    evaluate_model(model, y_test, y_pred, X_test)
    """

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
    # Read in the dataframe via the Pandas Library
    dataframe = pd.read_csv("FinalProj/creditcard.csv")

    # Preprocess the data using the preprocess function
    X, y = preprocess_data(dataframe)

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

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
    
