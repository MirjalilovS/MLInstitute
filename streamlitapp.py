# streamlit_app.py
import os
import numpy as np
import psycopg2
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

#Model import
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, 3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.drop_c  = nn.Dropout2d(0.25)
        self.drop_f  = nn.Dropout(0.5)
        self.fc1     = nn.Linear(64 * 14 * 14, 128)
        self.fc2     = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop_c(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop_f(x)
        return self.fc2(x)

@st.cache_resource(show_spinner=False)
def load_model():
    model = SimpleCNN()
    state = torch.load("mnist_cnn.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

#Database connection
@st.cache_resource(show_spinner=False)
def get_db_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", 5432),
        user=os.getenv("PGUSER", "mnist_user"),
        password=os.getenv("PGPASSWORD", "mnist_pw"),
        dbname=os.getenv("PGDATABASE", "mnist_app"),
    )

def log_prediction(pred: int, conf: float, true_digit: int | None):
    conn = get_db_conn()
    with conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO predictions (predicted, confidence, true_digit) "
            "VALUES (%s,%s,%s)",
            (pred, conf, true_digit)
        )

# UI setup
st.title("MNIST Digit Recognizer")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.confidence = None

# Buttons
col_btn1, col_btn2, col_true = st.columns([1, 1, 2])
with col_btn1:
    submit_clicked = st.button("Submit")
#with col_btn2:
 #   reset_clicked = st.button("Reset")
with col_true:
    true_digit = st.number_input(
        "True digit (0-9, optional)", min_value=0, max_value=9, step=1
    )

# Drawing canvas
canvas = st_canvas(
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}"
)

# Submit → preprocess, infer, log
if submit_clicked:
    if canvas.image_data is None:
        st.error("Draw a digit first!")
        st.stop()

    # ── 3.1 preprocess (28×28, [0,1], normalize) ──
    img = Image.fromarray(canvas.image_data.astype("uint8")).convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0                 # 0 (bg) .. 1 (stroke)
    arr = (arr - 0.1307) / 0.3081                                 # match training stats
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)      # [1,1,28,28]

    # ── 3.2 inference ──
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1)[0]
    pred   = int(probs.argmax().item())
    conf   = float(probs.max().item())

    # ── 3.3 store in session & DB ──
    st.session_state.prediction = pred
    st.session_state.confidence = conf
    #log_prediction(pred, conf, true_digit if true_digit is not None else None)

if st.session_state.prediction is not None:
    submit_true_clicked = st.button("Submit True Digit")
    if submit_true_clicked:
        # Log the prediction and provided true digit.
        # The user input from number_input is used; if not intended, adjust logic as needed.
        log_prediction(
            st.session_state.prediction,
            st.session_state.confidence,
            true_digit  # true_digit is an int from the number_input
        )
        st.experimental_rerun()
        st.success("Prediction logged with true digit.")
# ───────────────────────────────────────────
# 4.  Display result
# ───────────────────────────────────────────
if st.session_state.prediction is not None:
    st.success(
        f"**Prediction:** {st.session_state.prediction}   "
        f"**Confidence:** {st.session_state.confidence:.2%}"
    )
conn = get_db_conn()
if conn:
    import pandas as pd
    try:
        conn.commit()  # Ensure any previous transactions are committed
        df = pd.read_sql(
            "SELECT ts, predicted, confidence, true_digit "
            "FROM predictions ORDER BY ts DESC LIMIT 20;",
            conn
        )
        st.subheader("Recent predictions")
        st.dataframe(df)
    except Exception as e:
        st.warning(f"Could not fetch table:\n{e}")