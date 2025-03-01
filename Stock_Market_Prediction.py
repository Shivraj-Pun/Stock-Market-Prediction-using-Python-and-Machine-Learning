import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

st.title('📈 Stock Price Prediction App')
st.write("Enter a stock ticker to analyze historical trends and predict future movement.")

stock_symbol = st.text_input("Enter the stock ticker symbol (e.g., AAPL, TSLA, INFY):")
training_period = st.selectbox("Select the period for training data", ["1y", "3y", "5y", "10y"])

@st.cache_data
def fetch_stock_data(symbol, period):
    try:
        stock_data = yf.download(symbol, period=period)
        return stock_data if not stock_data.empty else None
    except Exception:
        return None

if stock_symbol:
    stock_data = fetch_stock_data(stock_symbol, training_period)

    if stock_data is None:
        st.error(f"⚠️ Invalid symbol {stock_symbol}. Please enter a valid stock ticker.")
    else:
        st.write(f"### Stock Data for {stock_symbol} ({training_period} period)")
        st.dataframe(stock_data.head())

        st.write("### 📊 Closing Price Over Time")
        st.line_chart(stock_data['Close'])

        stock_data['open-close'] = stock_data['Open'] - stock_data['Close']
        stock_data['low-high'] = stock_data['Low'] - stock_data['High']
        stock_data['is_quarter_end'] = (stock_data.index.month % 3 == 0).astype(int)
        stock_data['target'] = (stock_data['Close'].shift(-1) > stock_data['Close']).astype(int)
        stock_data.dropna(inplace=True)

        features = stock_data[['open-close', 'low-high', 'is_quarter_end']]
        target = stock_data['target']
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        X_train, X_valid, Y_train, Y_valid = train_test_split(features_scaled, target, test_size=0.1, random_state=42)
        smote = SMOTE(random_state=42)
        X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

        model = XGBClassifier(eval_metric='logloss')
        model.fit(X_train_resampled, Y_train_resampled)

        y_train_pred = model.predict(X_train_resampled)
        y_valid_pred = model.predict(X_valid)

        train_accuracy = accuracy_score(Y_train_resampled, y_train_pred)
        valid_accuracy = accuracy_score(Y_valid, y_valid_pred)

        st.write("### Model Performance")
        st.write(f"✔ **Training Accuracy:** {train_accuracy * 100:.2f}%")
        st.write(f"✔ **Validation Accuracy:** {valid_accuracy * 100:.2f}%")

        st.write("#### 📌 Classification Report")
        st.write(pd.DataFrame(classification_report(Y_valid, y_valid_pred, output_dict=True)).transpose())

        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sb.heatmap(confusion_matrix(Y_valid, y_valid_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        last_day_features = scaler.transform(stock_data[['open-close', 'low-high', 'is_quarter_end']].iloc[-1:].values)
        next_day_prediction = model.predict(last_day_features)[0]

        prediction_text = "📈 Stock Price is likely to go **UP!**" if next_day_prediction == 1 else "📉 Stock Price is likely to go **DOWN!**"
        st.write(f"### 🔮 Prediction for the Next Day: {prediction_text}")

else:
    st.info("🔎 Please enter a stock ticker symbol to begin analysis.")
