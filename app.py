import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy.interpolate import UnivariateSpline
from keras.models import Sequential
from keras.layers import LSTM, Dense,Input
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt


# Custom SplineImputer class
class SplineImputer:
    def __init__(self):
        self.spline = None

    def fit_transform(self, X):
        n_samples = X.shape[0]
        x = np.arange(n_samples)
        y = X.flatten()

        # Mask for non-missing values
        mask = ~np.isnan(y)
        
        # Fit spline to the non-missing values
        self.spline = UnivariateSpline(x[mask], y[mask], k=3, s=0)
        
        # Fill missing values using the fitted spline
        y[~mask] = self.spline(x[~mask])
        
        return y.reshape(-1, 1)
    
# File upload
st.title("crop-price-forecaster")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "xlsx"])

if uploaded_file:
    # Read the uploaded file
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.write("Data Preview:", data.head())
    
    # Select columns for date, price, and market
    date_col = st.selectbox("Select Date Column", data.columns)
    price_col = st.selectbox("Select Price Column", data.columns)
    market_col = st.selectbox("Select Market Column", data.columns)
    
    data = data[[date_col,price_col,market_col]]
    # Convert date column to datetime
    data[date_col] = pd.to_datetime(data[date_col])
    
   # Filter by market
    markets = data[market_col].unique()
    selected_market = st.selectbox("Select Market", markets)
    filtered_data = data[data[market_col] == selected_market]

    # Convert date column to datetime
    filtered_data[date_col] = pd.to_datetime(filtered_data[date_col])

    # Get start and end dates for filtering
    start_date = st.date_input('Start date', value=pd.to_datetime('2020-01-01').date())
    end_date = st.date_input('End date', value=pd.to_datetime('today').date())

    # Convert start and end dates to datetime format
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter data by selected date range
    filtered_data = filtered_data[(filtered_data[date_col] >= start_date) & (filtered_data[date_col] <= end_date)]
    
    # Set date column as index
    filtered_data.set_index(date_col, inplace=True)
    
    # Display filtered data
    st.write("Filtered Data:", filtered_data.head())

    
    filtered_data = filtered_data[[price_col]]
    # User selects resampling frequency or defaults to the data's frequency
    resample_options = {"Original": None, "Weekly": 'W', "Monthly": 'M'}
    resample_freq = st.selectbox("Select Resampling Frequency", list(resample_options.keys()), index=0)
    
    if resample_options[resample_freq] is not None:
        # Resample the data based on the selected frequency
        resampled_data = filtered_data.resample(resample_options[resample_freq]).mean()
    else:
        resampled_data = filtered_data
    # Handle missing values with Spline Imputer after resampling
    imputer = SplineImputer()
    data_array = imputer.fit_transform(resampled_data.values)
    #data_array = data.values
    data_array = data_array.astype('float32')
    st.write(data_array)
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data_array)

    # Splitting data into training and testing sets
    train_size = int(len(normalized_data) * 0.70)
    train, test = normalized_data[:train_size], normalized_data[train_size:]

    # Convert to sequences
    def to_sequences(dataset, seq_size=1):
        x = []
        y = []
        for i in range(len(dataset)-seq_size-1):
        #print(i)
            window = dataset[i:(i+seq_size), 0]
            x.append(window)
            y.append(dataset[i+seq_size, 0])

        return np.array(x),np.array(y)

    seq_size = st.slider("Select Sequence Size", 1, 30, 5)
    trainX, trainY = to_sequences(train, seq_size)
    testX, testY = to_sequences(test, seq_size)

    # Reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0],1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1,testX.shape[1]))

    # Model configuration
    print('Single LSTM with hidden Dense...')
    model = Sequential()
    # Add Input layer to define the input shape
    model.add(Input(shape=(None, seq_size)))
    # Add the LSTM layer
    model.add(LSTM(64))
    # Add Dense layers
    model.add(Dense(32))
    model.add(Dense(1))
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    # Print the model summary
    model.summary()

    # Train the model
    model.fit(trainX, trainY, validation_data=(testX, testY),verbose=2, epochs=100)

    # Predictions
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)
    
    # Inverse transform to get actual values
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    actual_train = scaler.inverse_transform([trainY])
    actual_test = scaler.inverse_transform([testY])

    # Evaluation
    train_rmse = np.sqrt(mean_squared_error(actual_train[0], train_predict[:,0]))
    test_rmse = np.sqrt(mean_squared_error(actual_test[0], test_predict[:,0]))
    train_mae = mean_absolute_error(actual_train[0], train_predict[:,0])
    test_mae = mean_absolute_error(actual_test[0], test_predict[:,0])
    train_mape = mean_absolute_percentage_error(actual_train[0], train_predict[:,0])
    test_mape = mean_absolute_percentage_error(actual_test[0], test_predict[:,0])

    st.write(f"Train RMSE: {train_rmse}")
    st.write(f"Test RMSE: {test_rmse}")
    st.write(f"Train MAE: {train_mae}")
    st.write(f"Test MAE: {test_mae}")
    st.write(f"Train MAPE: {train_mape}")
    st.write(f"Test MAPE: {test_mape}")

    # shift train predictions for plotting
    #we must shift the predictions so that they align on the x-axis with the original dataset. 
    trainPredictPlot = np.empty_like(normalized_data)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[seq_size:len(train_predict)+seq_size, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(normalized_data)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(seq_size*2)+1:len(normalized_data)-1, :] = test_predict
    
    ##Plot baseline and predictions
    f_data= scaler.inverse_transform(normalized_data)
    train = trainPredictPlot
    test = testPredictPlot
    
    # Plotting the results
    st.subheader("Actual vs Predicted Prices")
    plt.figure(figsize=(12, 6))
    plt.plot(resampled_data.index,f_data, label="Actual")
    plt.plot(resampled_data.index,test, label="Prediction")
    plt.legend()
    st.pyplot(plt)
