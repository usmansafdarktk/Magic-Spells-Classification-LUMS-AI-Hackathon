import pickle
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_saved_model(model_dir="rf_all_data_model_files"):
    """Load the saved model, label encoder and scaler"""
    with open(os.path.join(model_dir, "model.pkl"), "rb") as model_file:
        model = pickle.load(model_file)
    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as le_file:
        label_encoder = pickle.load(le_file)
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, label_encoder, scaler

def preprocess_coordinates(df):
    """Preprocess the coordinate data"""
    df['x'] = df['x'] - df['x'].mean()
    df['y'] = df['y'] - df['y'].mean()
    df['dx'] = df['x'].diff().fillna(0)
    df['dy'] = df['y'].diff().fillna(0)
    df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['angle'] = np.arctan2(df['dy'], df['dx']).fillna(0)
    return df[['dx', 'dy', 'speed', 'angle']]

def extract_features(df):
    """Extract features from preprocessed data"""
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
    return pd.DataFrame([features])

def prediction_to_one_hot(pred_class, num_classes):
    """Convert prediction class to one-hot encoded list"""
    one_hot = [0] * num_classes
    one_hot[pred_class] = 1
    return one_hot

def predict_test_data(test_dir, output_file="predictions.csv"):
    """Process test data and generate predictions"""
    # Load the saved model
    model, label_encoder, scaler = load_saved_model()
    num_classes = len(label_encoder.classes_)
    
    predictions = []
    
    # Process each CSV file in the test directory
    for csv_file in sorted(os.listdir(test_dir)):
        if csv_file.endswith('.csv'):
            file_path = os.path.join(test_dir, csv_file)
            
            # Load and preprocess the data
            df = pd.read_csv(file_path)
            processed_data = preprocess_coordinates(df)
            features = extract_features(processed_data)
            
            # Scale the features
            scaled_features = scaler.transform(features)
            
            # Make prediction
            pred_class = model.predict(scaled_features)[0]
            print("pred:", pred_class)
            one_hot_pred = prediction_to_one_hot(pred_class, num_classes)
            print("one hot:", one_hot_pred)
            # Store prediction with filename
            predictions.append({
                'file_name': csv_file,
                'labels': str(one_hot_pred)  # Convert list to string for CSV storage
            })
    
    # Create and save predictions DataFrame
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    # Specify your test data directory
    test_directory = r"E:\Full Dataset\All Data\2"  
    predict_test_data(test_directory)