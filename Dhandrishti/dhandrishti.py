import streamlit as st
import cv2
import pytesseract
import numpy as np
import face_recognition
import os
import yfinance as yf
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import json
import plotly.express as px

# Load AI financial advice model
financial_model = load_model("financial_advice_model.keras")

# Cache known faces to improve efficiency
@st.cache_data
def load_known_faces():
    if os.path.exists("known_faces.json"):
        with open("known_faces.json", "r") as file:
            return json.load(file)
    return []

# Improved Fake Currency Detection with enhanced edge detection & contrast analysis
def detect_fake_currency(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contrast = np.std(gray)
    return "Fake" if np.mean(edges) < 50 or contrast < 20 else "Real"

# Improved OCR with preprocessing
def extract_text_from_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return pytesseract.image_to_string(processed)

# Stock tracking with error handling and visualization
def track_portfolio(stock_symbols):
    portfolio = {}
    prices = []
    for stock in stock_symbols:
        try:
            data = yf.Ticker(stock)
            history = data.history(period="1d")
            if "Close" in history.columns and not history.empty:
                price = history["Close"].iloc[-1]
                portfolio[stock] = price
                prices.append({"Stock": stock, "Price": price})
            else:
                portfolio[stock] = "Invalid Symbol"
        except Exception as e:
            portfolio[stock] = f"Error: {str(e)}"
    return portfolio, prices

# Optimized Face Verification
def verify_face(image, known_faces):
    unknown_encoding = face_recognition.face_encodings(np.array(image))
    if len(unknown_encoding) == 0:
        return "No face detected"
    
    for face_data in known_faces:
        known_encoding = np.array(face_data["encoding"])
        matches = face_recognition.compare_faces([known_encoding], unknown_encoding[0], tolerance=0.5)
        if True in matches:
            return f"Face Verified: {face_data['name']}"
    return "Face not recognized"

# Enhanced AI-based Financial Advice
def get_financial_advice(user_financial_data):
    try:
        input_data = tf.convert_to_tensor([user_financial_data], dtype=tf.float32)
        advice = financial_model.predict(input_data)
        return f"AI Financial Advice: {advice[0][0]:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

st.title("Dhandrishti - AI-powered Financial Assistant")

option = st.sidebar.selectbox("Select Action", [
    "Detect Fake Currency", "Scan Document", "Stock Portfolio", "Face Verification", "Get Financial Advice"
])

if option == "Detect Fake Currency":
    uploaded_file = st.file_uploader("Upload currency image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Currency", use_column_width=True)
        with st.spinner('Analyzing...'):
            result = detect_fake_currency(image)
        st.success(f"Currency Status: {result}")

elif option == "Scan Document":
    uploaded_file = st.file_uploader("Upload document image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        with st.spinner('Extracting text...'):
            text = extract_text_from_image(image)
        st.text_area("Extracted Text:", text)

elif option == "Stock Portfolio":
    symbols = st.text_input("Enter stock symbols (comma separated)", "AAPL, TSLA, GOOGL")
    if st.button("Track Portfolio"):
        with st.spinner('Fetching stock data...'):
            stock_prices, price_data = track_portfolio(symbols.replace(" ", "").split(","))
        st.write(stock_prices)
        if price_data:
            df = pd.DataFrame(price_data)
            fig = px.bar(df, x='Stock', y='Price', title="Stock Prices", text='Price', color='Stock')
            st.plotly_chart(fig)

elif option == "Face Verification":
    uploaded_file = st.file_uploader("Upload face image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Face", use_column_width=True)
        with st.spinner('Verifying face...'):
            known_faces_db = load_known_faces()
            result = verify_face(image, known_faces_db)
        st.success(result)

elif option == "Get Financial Advice":
    user_financial_data = st.text_area("Enter your financial data (comma separated)")
    if st.button("Get Advice"):
        try:
            data = list(map(float, user_financial_data.split(",")))
            with st.spinner('Processing...'):
                advice = get_financial_advice(data)
            st.success(advice)
        except ValueError:
            st.error("Please enter valid numerical data.")
