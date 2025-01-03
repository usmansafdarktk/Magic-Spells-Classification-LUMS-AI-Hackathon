import pickle
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(base_dir):
    data = []
    labels = []
    
    # Iterate through each spell folder in the base directory
    for spell_folder in sorted(os.listdir(base_dir)):
        spell_path = os.path.join(base_dir, spell_folder)
        if not os.path.isdir(spell_path):
            continue  # Skip non-folder files
        
        # Iterate through CSV files in the spell folder
        for csv_file in os.listdir(spell_path):
            file_path = os.path.join(spell_path, csv_file)
            if file_path.endswith('.csv'):
                # Load the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                
                # Append the loaded dataframe to data list
                data.append(df)
                
                # Append the folder name as the label
                labels.append(spell_folder)
    
    # Concatenate all dataframes in the list into a single dataframe
    data = pd.concat(data, ignore_index=True)
    
    return data, labels

def preprocess_coordinates(df):
    # Normalize coordinates
    df['x'] = df['x'] - df['x'].mean()
    df['y'] = df['y'] - df['y'].mean()
    # Convert to relative movements
    df['dx'] = df['x'].diff().fillna(0)
    df['dy'] = df['y'].diff().fillna(0)
    
    # Calculate additional features
    df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['angle'] = np.arctan2(df['dy'], df['dx']).fillna(0)
    
    # Drop non-feature columns
    return df[['dx', 'dy', 'speed', 'angle']]

def extract_features(data_list):
    feature_set = []
    for df in data_list:
        # Compute statistical features for each data file
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

def train_random_forest(base_dir):
    # Load data and labels
    data_list, labels = load_data(base_dir)
    
    # Preprocess data
    processed_data = [preprocess_coordinates(df) for df in data_list]
    
    # Extract features
    X = extract_features(processed_data)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Evaluate model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    return model, le, scaler


base_dir = 'Synthetic All Data'
model, label_encoder, scaler = train_random_forest(base_dir)


def save_model(model, label_encoder, scaler, save_dir="rf2_all_data_model_files"):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "model.pkl"), "wb") as model_file:
        pickle.dump(model, model_file)
    with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as le_file:
        pickle.dump(label_encoder, le_file)
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

# Save the trained model
save_model(model, label_encoder, scaler)
print("Model saved") 

#A2
# import pickle
# import os
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load data
# def load_data(base_dir):
#     data = []
#     labels = []
#     for spell_folder in sorted(os.listdir(base_dir)):
#         spell_path = os.path.join(base_dir, spell_folder)
#         if not os.path.isdir(spell_path):
#             continue
#         for csv_file in os.listdir(spell_path):
#             file_path = os.path.join(spell_path, csv_file)
#             if file_path.endswith('.csv'):
#                 df = pd.read_csv(file_path)
#                 data.append(df)
#                 labels.append(spell_folder)
#     return data, labels

# # Preprocess data
# def preprocess_coordinates(df):
#     df['x'] = df['x'] - df['x'].mean()
#     df['y'] = df['y'] - df['y'].mean()
#     df['dx'] = df['x'].diff().fillna(0)
#     df['dy'] = df['y'].diff().fillna(0)
#     df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)
#     df['angle'] = np.arctan2(df['dy'], df['dx']).fillna(0)
#     df['acceleration'] = df['speed'].diff().fillna(0)
#     return df[['dx', 'dy', 'speed', 'angle', 'acceleration']]

# # Extract features
# def extract_features(data_list):
#     feature_set = []
#     for df in data_list:
#         features = {
#             'dx_mean': df['dx'].mean(),
#             'dx_std': df['dx'].std(),
#             'dy_mean': df['dy'].mean(),
#             'dy_std': df['dy'].std(),
#             'speed_mean': df['speed'].mean(),
#             'speed_std': df['speed'].std(),
#             'angle_mean': df['angle'].mean(),
#             'angle_std': df['angle'].std(),
#             'acceleration_mean': df['acceleration'].mean(),
#             'acceleration_std': df['acceleration'].std(),
#         }
#         feature_set.append(features)
#     return pd.DataFrame(feature_set)

# # Train Random Forest
# def train_random_forest(base_dir):
#     data_list, labels = load_data(base_dir)
#     processed_data = [preprocess_coordinates(df) for df in data_list]
#     X = extract_features(processed_data)
#     le = LabelEncoder()
#     y = le.fit_transform(labels)
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
#     # Hyperparameter tuning
#     param_grid = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [None, 10, 20, 30],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     }
#     rf = RandomForestClassifier(random_state=42)
#     grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
#     grid_search.fit(X_train, y_train)
    
#     # Best model
#     best_model = grid_search.best_estimator_
#     y_pred = best_model.predict(X_test)
    
#     # Evaluation
#     print("Best Parameters:", grid_search.best_params_)
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print(classification_report(y_test, y_pred, target_names=le.classes_))
    
#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title('Confusion Matrix')
#     plt.show()
    
#     return best_model, le, scaler

# # Save model
# def save_model(model, label_encoder, scaler, save_dir="rf_all_data_model_files"):
#     os.makedirs(save_dir, exist_ok=True)
#     with open(os.path.join(save_dir, "model.pkl"), "wb") as model_file:
#         pickle.dump(model, model_file)
#     with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as le_file:
#         pickle.dump(label_encoder, le_file)
#     with open(os.path.join(save_dir, "scaler.pkl"), "wb") as scaler_file:
#         pickle.dump(scaler, scaler_file)

# # Train and save
# base_dir = 'Synthetic All Data'
# model, label_encoder, scaler = train_random_forest(base_dir)
# save_model(model, label_encoder, scaler)
# print("Model saved")