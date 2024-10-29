import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def trainer(embeddings,labels):
    EMBEDDED_X=embeddings
    Y=labels
    encoder=LabelEncoder()
    encoder.fit(Y)
    Y_encoded=encoder.transform(Y)

    model=SVC(kernel='linear',probability=True)
    model.fit(EMBEDDED_X,Y_encoded)
    Y=list(set(Y))
    Y.sort()
    students=Y.copy()
    attendence_matrix=[]
    for student in students:
        x=[]
        x.append(student)
        x.append('A')
        attendence_matrix.append(x)
    return model,encoder,attendence_matrix

if __name__=='__main__':
     # Generate synthetic data for standalone training and testing
    num_samples = 50   # Number of data points (embeddings)
    embedding_size = 128  # Size of each embedding vector
    num_classes = 10  # Number of unique labels (students)

    # Generate random embeddings and labels
    embeddings = np.random.rand(num_samples, embedding_size)
    labels = [f"Student_{i % num_classes}" for i in range(num_samples)]

    # Train the model using the synthetic data
    model, encoder, attendance_matrix = trainer(embeddings, labels)

    # Split the data into training and testing sets for validation
    X_train, X_test, Y_train, Y_test = train_test_split(embeddings, labels, shuffle=True, random_state=17)

    # Predict on both training and testing data
    ypreds_train = model.predict(X_train)
    ypreds_test = model.predict(X_test)

    # Decode predictions to get actual labels
    Y_train_encoded = encoder.transform(Y_train)
    Y_test_encoded = encoder.transform(Y_test)
    
    # Evaluate model accuracy on training and testing data
    train_accuracy = accuracy_score(Y_train_encoded, ypreds_train)
    test_accuracy = accuracy_score(Y_test_encoded, ypreds_test)

    print(f"Training accuracy: {train_accuracy}")
    print(f"Testing accuracy: {test_accuracy}")
