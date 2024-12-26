import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def fetch_stock_data(symbol, start_date, end_date):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {symbol}. Please check the stock symbol and try again.")
        return data
    except Exception as e:
        raise ValueError(f"Failed to fetch data for {symbol}. Error: {str(e)}")

def add_technical_indicators(data):
    # Adding Simple Moving Average
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    
    # Adding Relative Strength Index
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Adding MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data

def prepare_data(data, look_back, future_steps):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x, y = [], []
    for i in range(look_back, len(data) - future_steps + 1):
        x.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i:i+future_steps, 0])  # Predicting only the 'Close' price
    
    return np.array(x), np.array(y), scaler

def create_model(input_shape, output_shape):
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(100)),
        Dropout(0.3),
        Dense(100, activation='relu'),
        Dense(output_shape)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def plot_predictions(actual, predicted, dates, symbol):
    plt.figure(figsize=(15, 7))
    plt.plot(dates, actual, label='Actual Prices', color='blue')
    plt.plot(dates, predicted, label='Predicted Prices', color='red')
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Main execution
if __name__ == "__main__":
    symbol = input("Enter Stock Symbol (e.g., AAPL for Apple): ")
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime.now()

    # Fetch and prepare data
    data = fetch_stock_data(symbol, start_date, end_date)
    data = add_technical_indicators(data)
    data = data.dropna()  # Remove any NaN values

    features = ['Close', 'Volume', 'SMA20', 'SMA50', 'RSI', 'MACD', 'Signal_Line']
    look_back = 60
    future_steps = 5  # Predicting 5 days into the future

    X, y, scaler = prepare_data(data[features], look_back, future_steps)

    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Create and train model
    model = create_model((X.shape[1], X.shape[2]), future_steps)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], len(features)-1))), axis=1))[:, 0]
    actual = scaler.inverse_transform(np.concatenate((y_test, np.zeros((y_test.shape[0], len(features)-1))), axis=1))[:, 0]

    # Plot results
    plot_dates = data.index[split+look_back:split+look_back+len(predictions)]
    plot_predictions(actual, predictions, plot_dates, symbol)

    # Print performance metrics
    mse = mean_squared_error(actual, predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(actual, predictions)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")

    # Predict next 5 days
    last_60_days = data[features].values[-60:]
    last_60_days_scaled = scaler.transform(last_60_days)
    X_predict = np.array([last_60_days_scaled])
    prediction = model.predict(X_predict)
    prediction = scaler.inverse_transform(np.concatenate((prediction, np.zeros((prediction.shape[0], len(features)-1))), axis=1))[:, 0]

    print("\nPredictions for the next 5 days:")
    for i, price in enumerate(prediction[0]):
        print(f"Day {i+1}: ${price:.2f}")
