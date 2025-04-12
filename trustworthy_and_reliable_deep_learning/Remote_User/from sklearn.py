from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Train the model
def train_model(request):
    # Load dataset
    dataset = pd.read_csv('Datasets.csv', encoding='latin-1')

    # Preprocessing
    X = dataset[['Fid', 'Packet_Size']].values
    y = dataset['Label'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree Classifier
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_predictions = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_predictions)

    # Train PRU (Deep Learning Model)
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save results
    detection_accuracy.objects.create(names='Decision Tree', ratio=dt_accuracy * 100)
    return render(request, 'SProvider/train_model.html', {'accuracy': dt_accuracy})