import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load Models
gb_model = joblib.load("gb_model.pkl")
cnn_model = load_model("cnn_model.keras")
tokenizer = joblib.load("tokenizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Predict Function
def classify_url(url):
    # Preprocess for CNN
    url_cnn = tokenizer.texts_to_sequences([url])
    url_cnn = pad_sequences(url_cnn, maxlen=50, padding='post')

    # Preprocess for GB
    url_features = {
        'Querylength': len(url),
        'domain_token_count': url.count('.'),
        'path_token_count': url.count('/'),
        'avgdomaintokenlen': np.mean([len(t) for t in url.split('.') if t]),
        'longdomaintokenlen': max([len(t) for t in url.split('.') if t]),
        'avgpathtokenlen': np.mean([len(t) for t in url.split('/') if t]),
    }
    url_gb = np.array([list(url_features.values())])

    # Predictions
    cnn_pred = np.argmax(cnn_model.predict(url_cnn), axis=1)[0]
    gb_pred = gb_model.predict(url_gb)[0]

    return label_encoder.inverse_transform([cnn_pred])[0], label_encoder.inverse_transform([gb_pred])[0]

# GUI Application
def predict_url():
    url = url_entry.get().strip()
    if not url:
        messagebox.showerror("Error", "Please enter a valid URL.")
        return

    try:
        cnn_result, gb_result = classify_url(url)
        cnn_label.config(text=f"CNN Prediction: {cnn_result}")
        gb_label.config(text=f"Gradient Boosting Prediction: {gb_result}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Tkinter GUI Setup
app = tk.Tk()
app.title("URL Classifier")
app.geometry("400x300")

# Input Field
url_label = tk.Label(app, text="Enter URL:", font=("Arial", 12))
url_label.pack(pady=10)

url_entry = tk.Entry(app, width=50, font=("Arial", 12))
url_entry.pack(pady=5)

# Predict Button
predict_button = tk.Button(app, text="Classify URL", font=("Arial", 12), command=predict_url)
predict_button.pack(pady=10)

# Output Labels
cnn_label = tk.Label(app, text="CNN Prediction: ", font=("Arial", 12))
cnn_label.pack(pady=5)

gb_label = tk.Label(app, text="Gradient Boosting Prediction: ", font=("Arial", 12))
gb_label.pack(pady=5)

# Run the application
app.mainloop()
