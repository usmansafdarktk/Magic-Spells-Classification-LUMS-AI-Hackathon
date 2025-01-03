import pickle
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(base_dir):
    data = []
    labels = []
    for spell_folder in sorted(os.listdir(base_dir)):
        spell_path = os.path.join(base_dir, spell_folder)
        
        if not os.path.isdir(spell_path):
            continue
        
        for csv_file in os.listdir(spell_path):
            file_path = os.path.join(spell_path, csv_file)
            
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                data.append(df)
                labels.append(spell_folder)
    
    return data, labels

def preprocess_coordinates(df):
    df['x'] = df['x'] - df['x'].mean()
    df['y'] = df['y'] - df['y'].mean()
    df['dx'] = df['x'].diff().fillna(0)
    df['dy'] = df['y'].diff().fillna(0)
    df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['angle'] = np.arctan2(df['dy'], df['dx']).fillna(0)
    return df[['dx', 'dy', 'speed', 'angle']]

def extract_features(data_list):
    feature_set = []
    for df in data_list:
        features = {
            'dx_mean': df['dx'].mean(),
            'dx_std': df['dx'].std(),
            'dy_mean': df['dy'].mean(),
            'dy_std': df['dy'].std(),
            'speed_mean': df['speed'].mean(),
            'speed_std': df['speed'].std(),
            'angle_mean': df['angle'].mean(),
            'angle_std': df['angle'].std(),
        }
        feature_set.append(features)
    return pd.DataFrame(feature_set)

def train_random_forest(X_train, X_test, y_train, y_test, le):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, le.classes_)
    return rf_model

def train_svm(X_train, X_test, y_train, y_test, le):
    svm_model = SVC(random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, le.classes_)
    return svm_model

def train_dnn(X_train, X_test, y_train, y_test, le):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(le.classes_), activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("DNN Accuracy:", accuracy_score(y_test, y_pred_classes))
    print(classification_report(y_test, y_pred_classes, target_names=le.classes_))
    cm = confusion_matrix(y_test, y_pred_classes)
    plot_confusion_matrix(cm, le.classes_)
    return model

def train_cnn(X_train, X_test, y_train, y_test, le):
    X_train_cnn = X_train[..., np.newaxis]  # Add a channel dimension for CNN
    X_test_cnn = X_test[..., np.newaxis]
    
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=X_train_cnn.shape[1:]))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(le.classes_), activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_cnn, y_train, epochs=20, batch_size=32, validation_data=(X_test_cnn, y_test))
    
    y_pred = model.predict(X_test_cnn)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("CNN Accuracy:", accuracy_score(y_test, y_pred_classes))
    print(classification_report(y_test, y_pred_classes, target_names=le.classes_))
    cm = confusion_matrix(y_test, y_pred_classes)
    plot_confusion_matrix(cm, le.classes_)
    return model

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def save_model(model, label_encoder, scaler, model_name, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{model_name}_model.pkl"), "wb") as model_file:
        pickle.dump(model, model_file)
    with open(os.path.join(save_dir, f"{model_name}_label_encoder.pkl"), "wb") as le_file:
        pickle.dump(label_encoder, le_file)
    with open(os.path.join(save_dir, f"{model_name}_scaler.pkl"), "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

def main(base_dir="Synthetic All Data"):
    # Load data
    data_list, labels = load_data(base_dir)
    
    # Preprocess data
    processed_data = [preprocess_coordinates(df) for df in data_list]
    X = extract_features(processed_data)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Train and save all models
    rf_model = train_random_forest(X_train, X_test, y_train, y_test, le)
    save_model(rf_model, le, scaler, "rf")
    
    svm_model = train_svm(X_train, X_test, y_train, y_test, le)
    save_model(svm_model, le, scaler, "svm")
    
    dnn_model = train_dnn(X_train, X_test, y_train, y_test, le)
    save_model(dnn_model, le, scaler, "dnn")
    
    cnn_model = train_cnn(X_train, X_test, y_train, y_test, le)
    save_model(cnn_model, le, scaler, "cnn")
    
    print("All models have been trained and saved.")

if __name__ == "__main__":
    main()
