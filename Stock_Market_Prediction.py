import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from imblearn.over_sampling import SMOTE

# Streamlit app setup
st.title('Stock Price Prediction App')
st.write("Select a stock and predict future trends based on historical data.")

# Stock selection with validation
stock_symbol = st.text_input("Enter the stock ticker symbol (e.g., AAPL, TSLA, INFY):")
# Period for checking if stock exists (1 year for validation)
period_check = "1y"

# Period for training and validation data (customizable by user)
training_period = st.selectbox("Select the period for training data", ["1y", "3y", "5y", "10y"])

# Function to validate stock symbol
def fetch_stock_data(symbol, period):
    try:
        stock_data = yf.download(symbol, period=period)
        if stock_data.empty:
            return None  # If the data is empty, return None
        return stock_data
    except Exception as e:
        return None  # If there is an error fetching data, return None

if stock_symbol:
    # Fetch stock data for symbol with fixed 1 year to check existence
    stock_data_check = fetch_stock_data(stock_symbol, period_check)

    if stock_data_check is None:
        st.error(f"Error fetching data for {stock_symbol}. Please enter a valid stock ticker symbol.")
    else:
        # If the stock symbol is valid, proceed with the training period
        st.write(f"Displaying data for {stock_symbol} ({training_period} for training)")
        
        # Fetch stock data for training with the user-defined period
        stock_data = fetch_stock_data(stock_symbol, training_period)

        if stock_data is None:
            st.error(f"Error fetching data for {stock_symbol} with period {training_period}. Please try another period.")
        else:
            st.dataframe(stock_data.head())

            # Visualize stock price
            st.write("### Closing Price Over Time")
            st.line_chart(stock_data['Close'])

            # Feature engineering
            stock_data['day'] = stock_data.index.day
            stock_data['month'] = stock_data.index.month
            stock_data['year'] = stock_data.index.year
            stock_data['open-close'] = stock_data['Open'] - stock_data['Close']
            stock_data['low-high'] = stock_data['Low'] - stock_data['High']
            stock_data['is_quarter_end'] = np.where(stock_data['month'] % 3 == 0, 1, 0)
            stock_data['target'] = np.where(stock_data['Close'].shift(-1) > stock_data['Close'], 1, 0)

            # Drop rows with NaN values
            stock_data = stock_data.dropna()

            # Define features and target
            features = stock_data[['open-close', 'low-high', 'is_quarter_end']]
            target = stock_data['target']

            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Split data into train and validation sets
            X_train, X_valid, Y_train, Y_valid = train_test_split(features_scaled, target, test_size=0.1, random_state=2022)

            # Handle class imbalance with SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

            # Train model using XGBoost
            model = XGBClassifier(eval_metric='logloss')
            model.fit(X_train_resampled, Y_train_resampled)

            # Evaluate model
            y_train_pred = model.predict(X_train_resampled)
            y_valid_pred = model.predict(X_valid)

            # Calculate accuracy for training and validation sets
            train_accuracy = accuracy_score(Y_train_resampled, y_train_pred)
            valid_accuracy = accuracy_score(Y_valid, y_valid_pred)

            st.write("### Model Performance")

            # Display total accuracy for training and validation sets
            st.write(f"**Training Accuracy**: {train_accuracy * 100:.2f}%")
            st.write(f"**Validation Accuracy**: {valid_accuracy * 100:.2f}%")

            # Create a summary of the metrics for training and validation
            # Training performance
            training_report = classification_report(Y_train_resampled, y_train_pred, output_dict=True)
            validation_report = classification_report(Y_valid, y_valid_pred, output_dict=True)

            # Convert the classification report to DataFrames
            training_df = pd.DataFrame(training_report).transpose()
            validation_df = pd.DataFrame(validation_report).transpose()

            # Display training and validation results as tables
            st.write("#### Training Performance")
            st.table(training_df)

            st.write("#### Validation Performance")
            st.table(validation_df)

            # Display confusion matrix manually
            st.write("### Confusion Matrix")
            cm = confusion_matrix(Y_valid, y_valid_pred)
            cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            cm_display.plot(cmap='Blues')
            st.pyplot(plt)

            # Make prediction for the next day's stock movement
            last_day_features = stock_data[['open-close', 'low-high', 'is_quarter_end']].iloc[-1:].values
            last_day_features_scaled = scaler.transform(last_day_features)
            
            next_day_prediction = model.predict(last_day_features_scaled)[0]
            
            if next_day_prediction == 1:
                st.write("### Prediction for the Next Day: Stock Price will go **Up**!")
            else:
                st.write("### Prediction for the Next Day: Stock Price will go **Down**!")

else:
    st.write("Please enter a stock ticker symbol to begin.")
