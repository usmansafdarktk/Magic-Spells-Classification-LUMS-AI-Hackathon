import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from threading import Thread
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pickle
import os

# Global variables
tracking = False
frame_data = []
cap = None
camera_label = None
prediction_label = None

def find_most_prominent_green(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_green = np.array([20, 100, 100])
    # upper_green = np.array([40, 255, 255])
    lower_green = np.array([20, 85, 140])
    upper_green = np.array([29, 155, 235])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] > 0:
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])
            return (x, y, largest_contour)
    return None, None, None

def update_camera_feed():
    global cap, tracking, camera_label, frame_data

    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Flip the frame horizontally for more intuitive display
            frame = cv2.flip(frame, 1)

            x, y, largest_contour = find_most_prominent_green(frame)
            if largest_contour is not None:
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                if x is not None and y is not None:
                    if tracking:
                        timestamp = len(frame_data)
                        frame_data.append({'timestamp': timestamp, 'x': x, 'y': y})

                    # Draw the current position
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"({x}, {y})", (x + 10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # Draw the trace (path of movement)
                    for i in range(1, len(frame_data)):
                        pt1 = (frame_data[i - 1]['x'], frame_data[i - 1]['y'])
                        pt2 = (frame_data[i]['x'], frame_data[i]['y'])
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

            # Convert frame to RGB for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.imgtk = imgtk
            camera_label.configure(image=imgtk)

    window.after(10, update_camera_feed)


def initialize_camera():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access the webcam.")
        return False
    return True

def toggle_tracking():
    global tracking, frame_data
    tracking = not tracking

    if tracking:
        frame_data = []
        start_button.config(text="Stop Tracking")
        plot_button.config(state=tk.DISABLED)
    else:
        start_button.config(text="Track Again" if frame_data else "Start Tracking")
        plot_button.config(state=tk.NORMAL)


def normalize_coordinates(data):
    x_values = data[:, 0]
    y_values = data[:, 1]

    # Min-Max Normalization to [0, 1]
    x_min, x_max = np.min(x_values), np.max(x_values)
    y_min, y_max = np.min(y_values), np.max(y_values)

    x_normalized = (x_values - x_min) / (x_max - x_min)
    y_normalized = (y_values - y_min) / (y_max - y_min)

    return np.column_stack((x_normalized, y_normalized))

def save_and_plot_data():
    global frame_data
    if not frame_data:
        messagebox.showerror("Error", "No data captured to save.")
        return

    df = pd.DataFrame(frame_data)
    data_array = df[['x', 'y']].to_numpy()
    num_points = len(data_array)

    if num_points > 100:
        indices = np.linspace(0, num_points - 1, 100, dtype=int)
        sampled_data = data_array[indices]
    elif num_points < 100:
        x_original = np.arange(num_points)
        x_target = np.linspace(0, num_points - 1, 100)

        interpolator_x = interp1d(x_original, data_array[:, 0], kind='linear')
        interpolator_y = interp1d(x_original, data_array[:, 1], kind='linear')
        sampled_data = np.column_stack((interpolator_x(x_target), interpolator_y(x_target)))
    else:
        sampled_data = data_array

    # Normalize the sampled data
    normalized_data = normalize_coordinates(sampled_data)

    sampled_df = pd.DataFrame(normalized_data, columns=['x', 'y'])
    sampled_df['Frame'] = range(1, 101)
    sampled_df = sampled_df[['Frame', 'x', 'y']]
    sampled_df.to_csv('tracked_data.csv', index=False)
    print("Coordinates saved to tracked_data.csv")

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(range(len(normalized_data)), normalized_data[:, 0], 'b-', label='X coordinate')
    plt.plot(range(len(normalized_data)), normalized_data[:, 1], 'r-', label='Y coordinate')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Coordinate Value')
    plt.title('Normalized Coordinate Values over Samples')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(normalized_data[:, 0], normalized_data[:, 1], 'g-', label='Movement Path')
    plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=np.arange(len(normalized_data)),
                cmap='viridis', label='Points', alpha=0.5)
    plt.xlabel('Normalized X Coordinate')
    plt.ylabel('Normalized Y Coordinate')
    plt.title('Movement Path')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Call the prediction logic after saving the data
    predicted_label = make_prediction('tracked_data.csv')
    print("Predicted Label:", predicted_label)

    # Display prediction on GUI
    prediction_label.config(text=f"Prediction: {predicted_label}")

def quit_application():
    global cap
    if cap is not None:
        cap.release()
    window.quit()

# Prediction Logic
def load_model(save_dir="rf_all_data_model_files"):
    with open(os.path.join(save_dir, "model.pkl"), "rb") as model_file:
        model = pickle.load(model_file)
    with open(os.path.join(save_dir, "label_encoder.pkl"), "rb") as le_file:
        label_encoder = pickle.load(le_file)
    with open(os.path.join(save_dir, "scaler.pkl"), "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, label_encoder, scaler

def preprocess_new_data(df):
    df['x'] = df['x'] - df['x'].mean()
    df['y'] = df['y'] - df['y'].mean()
    df['dx'] = df['x'].diff().fillna(0)
    df['dy'] = df['y'].diff().fillna(0)
    df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['angle'] = np.arctan2(df['dy'], df['dx']).fillna(0)
    return df[['dx', 'dy', 'speed', 'angle']]

def extract_new_features(df):
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

def make_prediction(file_path, save_dir="rf_all_data_model_files"):
    model, label_encoder, scaler = load_model(save_dir)
    df = pd.read_csv(file_path)
    preprocessed_df = preprocess_new_data(df)
    features = extract_new_features(preprocessed_df)

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    predicted_label = label_encoder.inverse_transform(prediction)

    return predicted_label[0]

# GUI Setup
window = tk.Tk()
window.title("Green Object Tracker")
window.geometry("800x600")
window.configure(bg="#f0f0f0")

style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=10)
style.configure("TFrame", background="#f0f0f0")
style.configure("TLabel", font=("Helvetica", 10), background="#f0f0f0")

camera_frame = ttk.Frame(window)
camera_frame.pack(padx=10, pady=10)

camera_label = tk.Label(camera_frame, bg="#333", width=640, height=480, borderwidth=2, relief="groove")
camera_label.pack()

button_frame = ttk.Frame(window)
button_frame.pack(pady=20)

start_button = ttk.Button(button_frame, text="Start Tracking", command=toggle_tracking)
start_button.grid(row=0, column=0, padx=10)

plot_button = ttk.Button(button_frame, text="Plot Data", command=save_and_plot_data)
plot_button.grid(row=0, column=1, padx=10)
plot_button.config(state=tk.DISABLED)

quit_button = ttk.Button(button_frame, text="Quit", command=quit_application)
quit_button.grid(row=0, column=2, padx=10)

# Prediction label setup
prediction_label = tk.Label(window, text="Prediction: None", font=("Helvetica", 14), bg="#f0f0f0")
prediction_label.pack(pady=10)

if initialize_camera():
    update_camera_feed()

window.mainloop()
