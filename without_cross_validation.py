import pickle
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

def load_data(base_dir):
    data = []
    labels = []
    
    # Iterate through each spell folder
    for spell_folder in sorted(os.listdir(base_dir)):
        spell_path = os.path.join(base_dir, spell_folder)
        
        if not os.path.isdir(spell_path):
            continue  # Skip non-folder files
        
        # Iterate through CSV files in the spell folder
        for csv_file in os.listdir(spell_path):
            file_path = os.path.join(spell_path, csv_file)
            
            if file_path.endswith('.csv'):
                # Load the CSV file
                df = pd.read_csv(file_path)
                data.append(df)
                labels.append(spell_folder)  # Use folder name as label
    
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

def train_random_forest_on_all_data(base_dir):
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
    
    # Train Random Forest model on all data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print("Model trained on all data.")
    return model, le, scaler

base_dir = 'Synthetic All Data'
model, label_encoder, scaler = train_random_forest_on_all_data(base_dir)

def save_model(model, label_encoder, scaler, save_dir="rf_all_data_model_files"):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "model.pkl"), "wb") as model_file:
        pickle.dump(model, model_file)
    with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as le_file:
        pickle.dump(label_encoder, le_file)
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

# Save the trained model
save_model(model, label_encoder, scaler)
print("Model saved!!")
