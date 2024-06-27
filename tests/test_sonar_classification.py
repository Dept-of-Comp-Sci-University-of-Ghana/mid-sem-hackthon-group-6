# test_sonar_classification.py

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Sample test data (replace with actual data loading)
def load_test_data(file_path='/workspaces/mid-sem-hackthon-group-6/data/sonar.all-data.csv'):
    # ... Load your test data here ...
    data = pd.DataFrame(file_path)
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    Y = np.where(Y == 'M', 1, 0)
    return X, Y

# Test data loading
def test_data_loading():
    X, Y = load_test_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(Y, np.ndarray)

# Test feature scaling
def test_feature_scaling():
    X, _ = load_test_data()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    assert X_scaled.min() >= 0
    assert X_scaled.max() <= 1

# Test model training and evaluation
def test_model_training():
    X, Y = load_test_data()
    X_scaled = MinMaxScaler().fit_transform(X)
    
    lr_model = LogisticRegression()
    svm_model = SVC(C=1.0, kernel='rbf', probability=True)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    ensemble = VotingClassifier(estimators=[('lr', lr_model), ('svm', svm_model), ('knn', knn_model)], voting='soft')

    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    accuracies = []
    for train_index, test_index in kfold.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        ensemble.fit(X_train, Y_train)
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(Y_test, y_pred)
        accuracies.append(accuracy)
    
    assert np.mean(accuracies) > 0  # Check if mean accuracy is reasonable

# Run tests with pytest
if __name__ == '__main__':
    pytest.main()