# Required Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import joblib

# ------------------------------
# Load Dataset
# ------------------------------
file_path = "ISCX-URL2016_All.csv"  # Replace with your dataset path
df = pd.read_csv(file_path)

# Select relevant columns
df = df[['Querylength', 'domain_token_count', 'path_token_count',
         'avgdomaintokenlen', 'longdomaintokenlen', 'avgpathtokenlen',
         'URL_Type_obf_Type']].copy()

# Handle missing values if any
df = df.dropna()

# ------------------------------
# Encode Labels
# ------------------------------
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['URL_Type_obf_Type'])

# Save label encoder for later use
joblib.dump(label_encoder, "label_encoder.pkl")

# ------------------------------
# Traditional Model Features
# ------------------------------
X = df.drop(columns=['URL_Type_obf_Type', 'label'])
y = df['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------------
# Fine-Tuned Gradient Boosting Model
# ------------------------------
gb_model = GradientBoostingClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.01]
}

grid_search = GridSearchCV(gb_model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best Model
gb_model = grid_search.best_estimator_

# Save Traditional Model
joblib.dump(gb_model, "gb_model.pkl")

# Predictions
y_pred_traditional = gb_model.predict(X_test)

# Evaluate Traditional Model
gb_accuracy = accuracy_score(y_test, y_pred_traditional)
print("Fine-Tuned Gradient Boosting Model Accuracy:", gb_accuracy)
print("Gradient Boosting Model Classification Report:")
print(classification_report(y_test, y_pred_traditional, target_names=label_encoder.classes_))

# ------------------------------
# CNN Features
# ------------------------------
tokenizer = Tokenizer(num_words=1000, char_level=True, oov_token="<UNK>")
tokenizer.fit_on_texts(df['URL_Type_obf_Type'])
X_cnn = tokenizer.texts_to_sequences(df['URL_Type_obf_Type'])
X_cnn = pad_sequences(X_cnn, maxlen=50, padding='post')

# Save tokenizer
joblib.dump(tokenizer, "tokenizer.pkl")

# Train-Test Split for CNN
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_cnn, y, test_size=0.3, random_state=42
)

# ------------------------------
# Fine-Tuned CNN Model
# ------------------------------
cnn_model = Sequential([
    Embedding(input_dim=1000, output_dim=50, input_length=50),
    Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)),
    GlobalMaxPooling1D(),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')  # Multi-class classification
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the CNN Model
history = cnn_model.fit(
    X_train_cnn, y_train_cnn,
    validation_split=0.2,
    epochs=20, batch_size=32, verbose=1,
    callbacks=[early_stopping]
)

# Save the CNN Model
cnn_model.save("cnn_model.keras")

# Predictions
y_pred_cnn = cnn_model.predict(X_test_cnn)
y_pred_cnn = np.argmax(y_pred_cnn, axis=1)

# Evaluate CNN Model
cnn_accuracy = accuracy_score(y_test_cnn, y_pred_cnn)
print("Fine-Tuned CNN Model Accuracy:", cnn_accuracy)
print("CNN Model Classification Report:")
print(classification_report(y_test_cnn, y_pred_cnn, target_names=label_encoder.classes_))

# ------------------------------
# Visualization of Accuracy
# ------------------------------
plt.bar(['Gradient Boosting', 'CNN'], [gb_accuracy, cnn_accuracy], color=['blue', 'green'])
plt.ylim(0, 1)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()
