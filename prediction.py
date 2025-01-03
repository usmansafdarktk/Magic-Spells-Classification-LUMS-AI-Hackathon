# import pickle
# import pandas as pd
# import numpy as np
# import os
# def load_model(save_dir="rf_all_data_model_files"):
#     with open(os.path.join(save_dir, "model.pkl"), "rb") as model_file:
#         model = pickle.load(model_file)
#     with open(os.path.join(save_dir, "label_encoder.pkl"), "rb") as le_file:
#         label_encoder = pickle.load(le_file)
#     with open(os.path.join(save_dir, "scaler.pkl"), "rb") as scaler_file:
#         scaler = pickle.load(scaler_file)
#     return model, label_encoder, scaler

# def preprocess_new_data(df):
#     # Apply the same preprocessing as training
#     df['x'] = df['x'] - df['x'].mean()
#     df['y'] = df['y'] - df['y'].mean()
#     df['dx'] = df['x'].diff().fillna(0)
#     df['dy'] = df['y'].diff().fillna(0)
#     df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)
#     df['angle'] = np.arctan2(df['dy'], df['dx']).fillna(0)
#     return df[['dx', 'dy', 'speed', 'angle']]

# def extract_new_features(df):
#     features = {
#         'dx_mean': df['dx'].mean(),
#         'dx_std': df['dx'].std(),
#         'dy_mean': df['dy'].mean(),
#         'dy_std': df['dy'].std(),
#         'speed_mean': df['speed'].mean(),
#         'speed_std': df['speed'].std(),
#         'angle_mean': df['angle'].mean(),
#         'angle_std': df['angle'].std(),
#     }
#     return pd.DataFrame([features])

# def make_prediction(file_path, save_dir="rf_all_data_model_files"):
#     # Load saved model components
#     model, label_encoder, scaler = load_model(save_dir)

#     # Load and preprocess new data
#     df = pd.read_csv(file_path)
#     preprocessed_df = preprocess_new_data(df)
#     features = extract_new_features(preprocessed_df)

#     # Scale features
#     scaled_features = scaler.transform(features)

#     # Predict
#     prediction = model.predict(scaled_features)
#     predicted_label = label_encoder.inverse_transform(prediction)

#     return predicted_label[0]

# # Example usage
# new_file_path = r"E:\Full Dataset\All Data\2\96.csv"
# predicted_label = make_prediction(new_file_path)
# print("Predicted Label:", predicted_label)












# import pickle
# import pandas as pd
# import numpy as np
# import os
# from tqdm import tqdm

# def load_model(save_dir="svm model"):
#     with open(os.path.join(save_dir, "svm_model.pkl"), "rb") as model_file:
#         model = pickle.load(model_file)
#     with open(os.path.join(save_dir, "svm_label_encoder.pkl"), "rb") as le_file:
#         label_encoder = pickle.load(le_file)
#     with open(os.path.join(save_dir, "svm_scaler.pkl"), "rb") as scaler_file:
#         scaler = pickle.load(scaler_file)
#     return model, label_encoder, scaler

# def preprocess_new_data(df):
#     # Apply the same preprocessing as training
#     df['x'] = df['x'] - df['x'].mean()
#     df['y'] = df['y'] - df['y'].mean()
#     df['dx'] = df['x'].diff().fillna(0)
#     df['dy'] = df['y'].diff().fillna(0)
#     df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)
#     df['angle'] = np.arctan2(df['dy'], df['dx']).fillna(0)
#     return df[['dx', 'dy', 'speed', 'angle']]

# def extract_new_features(df):
#     features = {
#         'dx_mean': df['dx'].mean(),
#         'dx_std': df['dx'].std(),
#         'dy_mean': df['dy'].mean(),
#         'dy_std': df['dy'].std(),
#         'speed_mean': df['speed'].mean(),
#         'speed_std': df['speed'].std(),
#         'angle_mean': df['angle'].mean(),
#         'angle_std': df['angle'].std(),
#     }
#     return pd.DataFrame([features])

# def create_tensor(predicted_label):
#     """
#     Create a tensor of zeros with 1 at the predicted label index
#     Returns a string representation without spaces after commas
#     """
#     tensor = [0] * 25  # Create tensor with 25 zeros
#     try:
#         if isinstance(predicted_label, (int, np.integer)):
#             tensor[predicted_label] = 1
#     except (IndexError, TypeError):
#         # Return all zeros for error cases
#         pass
#     # Convert to string and remove spaces after commas
#     return '[' + ','.join(map(str, tensor)) + ']'

# def process_folder(folder_path, save_dir="svm model", output_file="submis.csv"):
#     """
#     Process all CSV files in the given folder and its subfolders to make predictions.
    
#     Args:
#         folder_path (str): Path to the folder containing CSV files
#         save_dir (str): Directory containing the model files
#         output_file (str): Name of the CSV file to save predictions
    
#     Returns:
#         pd.DataFrame: DataFrame containing file paths and label tensors
#     """
#     # Load model components once
#     model, label_encoder, scaler = load_model(save_dir)
    
#     results = []
    
#     # Get all CSV files in the folder and subfolders
#     csv_files = []
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith('.csv'):
#                 csv_files.append(os.path.join(root, file))
    
#     # Process each CSV file with a progress bar
#     for file_path in tqdm(csv_files, desc="Processing files"):
#         try:
#             # Load and preprocess data
#             df = pd.read_csv(file_path)
#             preprocessed_df = preprocess_new_data(df)
#             features = extract_new_features(preprocessed_df)
            
#             # Scale features
#             scaled_features = scaler.transform(features)
            
#             # Predict
#             prediction = model.predict(scaled_features)
#             predicted_label = label_encoder.inverse_transform(prediction)[0]
            
#             # Create tensor for the prediction
#             tensor = create_tensor(int(predicted_label)-1)
            
#             # Get just the filename without extension
#             file_name = os.path.splitext(os.path.basename(file_path))[0]
            
#             # Store results
#             results.append({
#                 'file_name': file_name,
#                 'labels': tensor
#             })
            
#         except Exception as e:
#             print(f"Error processing {file_path}: {str(e)}")
#             error_tensor = '[' + ','.join(['0']*25) + ']'
#             file_name = os.path.splitext(os.path.basename(file_path))[0]
#             results.append({
#                 'file_name': file_name,
#                 'labels': error_tensor
#             })
    
#     # Create DataFrame with results
#     results_df = pd.DataFrame(results)
#     # Save predictions to CSV
#     results_df.to_csv(output_file, index=False)
#     print(f"\nPredictions saved to {output_file}")
    
#     return results_df
# # Example usage
# if __name__ == "__main__":
#     folder_path = r"E:\Full Dataset\Final Dataset"  # Replace with your folder path
#     predictions_df = process_folder(folder_path)
#     # Print sample of predictions
#     print("\nSample of predictions:")
#     print(predictions_df.head())


import os
import pickle
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
def load_saved_model_and_components(model_name, save_dir="models"):
    with open(os.path.join(save_dir, f"{model_name}_model.pkl"), "rb") as model_file:
        model = pickle.load(model_file)
    with open(os.path.join(save_dir, f"{model_name}_label_encoder.pkl"), "rb") as le_file:
        label_encoder = pickle.load(le_file)
    with open(os.path.join(save_dir, f"{model_name}_scaler.pkl"), "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, label_encoder, scaler
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
def predict_with_cnn(model, label_encoder, scaler, folder_path):
    predictions = []
    file_names = []
    
    for csv_file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, csv_file)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            processed_df = preprocess_coordinates(df)
            features = extract_features([processed_df])
            scaled_features = scaler.transform(features)
            
            # Add channel dimension
            scaled_features_cnn = scaled_features[..., np.newaxis]
            
            # Predict
            pred_probs = model.predict(scaled_features_cnn)
            pred_class = np.argmax(pred_probs, axis=1)
            pred_label = label_encoder.inverse_transform(pred_class)[0]
            
            predictions.append(pred_label)
            file_names.append(csv_file)
    
    return pd.DataFrame({"File": file_names, "Prediction": predictions})

# Predicting on new data
def main_predict(base_dir=r"E:\Full Dataset\Final Dataset", model_name="cnn"):
    # Load the CNN model and preprocessing components
    cnn_model, le, scaler = load_saved_model_and_components(model_name)
    
    # Predict on new CSV files
    predictions_df = predict_with_cnn(cnn_model, le, scaler, base_dir)
    print(predictions_df)
    predictions_df.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    main_predict()
