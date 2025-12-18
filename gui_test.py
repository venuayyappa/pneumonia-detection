import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Config
# ----------------------------
IMG_SIZE = (224, 224)
MODEL_PATH = "best_simplecnn.h5"

# ----------------------------
# Load model
# ----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# ----------------------------
# Prediction function
# ----------------------------
def pick_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    # Load image
    img = Image.open(file_path)

    # Handle grayscale X-rays
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize(IMG_SIZE)

    # Preprocess
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prob = model.predict(img_array, verbose=0)[0][0]

    if prob >= 0.5:
        result = "PNEUMONIA"
        color = "red"
    else:
        result = "NORMAL"
        color = "green"

    label.config(
        text=f"{result}\nConfidence: {prob*100:.2f}%",
        fg=color
    )

# ----------------------------
# GUI
# ----------------------------
root = tk.Tk()
root.title("Pneumonia Detection")
root.geometry("300x200")
root.resizable(False, False)

btn = tk.Button(
    root,
    text="Select Chest X-ray",
    command=pick_image,
    width=20
)
btn.pack(pady=30)

label = tk.Label(root, text="", font=("Arial", 14))
label.pack()

root.mainloop()
