# streamlit_app.py
"""
MNIST Digit Recogniser
A simple app that took me way too long using psycopg2 that lets you test out a CNN on handwritten digits.
Note: This version fixes the recursive-connection bug and disables the watchdog to avoid PyTorch issues.
"""

# ── PRE-SET ENVIRONMENT (Don’t change the order) ──────────────
import os
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"   # Disables filewatcher to prevent crashes

# ─────────────────────────── Imports ──────────────────────────────
import warnings
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
import streamlit as st  # Importing Streamlit after setting environment variables
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hiding some warnings from pandas/SQLAlchemy we don't care about
warnings.filterwarnings("ignore", category=UserWarning, module="pandas.io.sql")

st.set_page_config(page_title="MNIST Digit Recogniser", page_icon="✏️")

# ────────────────────── 1. Define the CNN model ─────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_c = nn.Dropout2d(0.25)
        self.drop_f = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.drop_c(x)
        x = torch.flatten(x, 1)
        x = self.drop_f(F.relu(self.fc1(x)))
        return self.fc2(x)

@st.cache_resource(show_spinner=False)
def load_model() -> nn.Module:
    # Load the pre-trained CNN and set it to evaluation mode
    m = SimpleCNN()
    m.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))
    m.eval()
    return m

model = load_model()

# ───────────────────── 2. PostgreSQL Utility Functions ──────────────────────
def _pg_kwargs():
    # Connection parameters for our PostgreSQL database
    return dict(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", 5432),
        user=os.getenv("PGUSER", "mnist_user"),
        password=os.getenv("PGPASSWORD", "mnist_pw"),
        dbname=os.getenv("PGDATABASE", "mnist_app"),
    )

@contextmanager
def pg_conn():
    # Using a context manager to handle our database connection neatly
    conn = psycopg2.connect(**_pg_kwargs())
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()

def log_prediction(pred: int, conf: float, true_digit: int | None) -> None:
    # Log the prediction details into the database
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO predictions (ts, predicted, confidence, true_digit)
            VALUES (%s, %s, %s, %s)
            """,
            (datetime.utcnow(), pred, conf, true_digit),
        )

@st.cache_data(show_spinner=False)
def fetch_recent(limit: int = 20) -> pd.DataFrame:
    # Retrieve the most recent predictions from our database
    with pg_conn() as conn:
        return pd.read_sql_query(
            f"""
            SELECT ts, predicted, confidence, true_digit
            FROM predictions
            ORDER BY ts DESC
            LIMIT {limit}
            """,
            conn,
        )

# ────────────────────────── 3. Build the UI ────────────────────────────────
st.title("MNIST Digit Recogniser")

# Initialize session state variables if they don't exist
st.session_state.setdefault("prediction", None)
st.session_state.setdefault("confidence", None)
st.session_state.setdefault("canvas_key", 0)

# Set up columns for button and input elements
c1, _, c3 = st.columns([1, 0.2, 2])
with c1:
    submit_clicked = st.button("Submit")
with c3:
    true_digit = st.number_input(
        "True digit (optional)", min_value=0, max_value=9, step=1, format="%d"
    )

# Create a drawable canvas for users to sketch their digit
canvas = st_canvas(
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",
)

if submit_clicked:
    if canvas.image_data is None:
        st.error("Please draw something first!")
        st.stop()

    # Process the drawn image: convert, resize, and normalize it
    img = (
        Image.fromarray(canvas.image_data.astype("uint8"))
        .convert("L")
        .resize((28, 28))
    )
    arr = (np.array(img, dtype=np.float32) / 255.0 - 0.1307) / 0.3081
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    # Make a prediction using our CNN model
    probs = torch.softmax(model(tensor)[0], dim=0)
    st.session_state.prediction = int(probs.argmax().item())
    st.session_state.confidence = float(probs.max().item())

# Show the prediction result if available
if st.session_state.prediction is not None:
    st.success(
        f"**Prediction:** {st.session_state.prediction}&nbsp;&nbsp;"
        f"**Confidence:** {st.session_state.confidence:.2%}"
    )

    # Button to log the prediction with an optional true digit
    if st.button("Submit True Digit"):
        log_prediction(
            st.session_state.prediction,
            st.session_state.confidence,
            int(true_digit) if true_digit is not None else None,
        )
        # Reset session state for a new prediction
        st.session_state.prediction = None
        st.session_state.confidence = None
        st.session_state.canvas_key += 1
        st.experimental_rerun()

# Display a table of recent predictions from the database
df = fetch_recent()
if df.empty:
    st.info("No predictions logged yet – try drawing a digit and hitting **Submit**.")
else:
    st.subheader("Recent predictions")
    st.dataframe(df, use_container_width=True)
