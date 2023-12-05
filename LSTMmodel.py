import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def clean(df, startDateStr=None, endDateStr=None):

    # filter dates and set as index
    df["Date"] = pd.to_datetime(df["Date"], format='%Y-%m-%d')
    df.index = df["Date"]
    
    minDate = df["Date"].min()
    minDateStr = minDate.date().strftime('%m/%d/%Y')
    
    maxDate = df["Date"].max()
    maxDateStr = maxDate.date().strftime('%m/%d/%Y')
    
    startDate = pd.to_datetime(startDateStr,format='%m/%d/%Y')
    endDate = pd.to_datetime(endDateStr,format='%m/%d/%Y')
    
    startDate = startDate or minDate
    endDate = endDate or maxDate

    if startDate < minDate:
        raise ValueError("Starting date is not valid in this dataset. This dataset ranges in-between " + minDateStr + " and " + maxDateStr + ".")
    elif endDate > maxDate:
        raise ValueError("Ending date is not valid in this dataset. This dataset ranges in-between " + minDateStr + " and " + maxDateStr + ".")
    else:
        df = df[(df["Date"] >= startDate) & (df["Date"] <= endDate)]    

    # only keep "Close" column as a feature
    df = df.loc[:, ["Date", "Close"]]
    df.dropna(inplace=True)

    # scale values
    sc = MinMaxScaler(feature_range = (0,1))
    df["Close"] = sc.fit_transform(df.loc[:, ["Close"]])
    
    return df, sc

def timestep(df, stepCount = 60):
    X, y = [], []

    df = np.array(df)

    for i in range(stepCount, df.shape[0]):
        row = []
        row.append(df[i, 0])
        values = df[i-stepCount:i, 1]
        for j in range(values.size):
            row.append(values[j])
        X.append(row)
        y.append(df[i, 1])
    
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    return X, y

def trainModel(csvFile, epochCount, timestepCount, startDate, endDate):
    
    df = pd.read_csv(csvFile)

    # clean up dataset and scale features ("Close") between 0-1
    df, sc = clean(df,startDate,endDate)

    # set up backtracking timesteps in dataset
    X, y = timestep(df, timestepCount)
    
    # train-validation-test split = 80-10-10
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.20, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, shuffle=False)

    X_train, X_val, X_test = np.array(X_train), np.array(X_val), np.array(X_test)
    y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)

    dates_train, dates_val, dates_test = X_train[:,0], X_val[:,0], X_test[:,0]
    X_train, X_val, X_test = X_train[:,1:], X_val[:,1:], X_test[:,1:]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))    
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))    
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))   
    X_train, X_val, X_test = X_train.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32)

    # model setup 
    model = Sequential()
    
    model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer=Adam(learning_rate=0.001),loss='mean_squared_error')
    model.fit(X_train,y_train,epochs=epochCount,batch_size=32)

    
    return (dates_train, dates_val, dates_test,
           X_train, y_train, 
           X_val, y_val, 
           X_test, y_test, 
           model, sc)

# settings
epochCount = 100
timestepCount = 60
startDate = "01/02/2017"
endDate = None

# USA Model
US_dates_train, US_dates_val, US_dates_test, US_X_train, US_y_train, US_X_val, US_y_val, US_X_test, US_y_test, US_model, US_sc = trainModel("^NDXT.csv", epochCount, timestepCount, startDate, endDate)
# India Model
IN_dates_train, IN_dates_val, IN_dates_test, IN_X_train, IN_y_train, IN_X_val, IN_y_val, IN_X_test, IN_y_test, IN_model, IN_sc = trainModel("^CNXIT.csv", epochCount, timestepCount, startDate, endDate)

# Plotting settings
plt.rcParams['font.size'] = 9

# Training data
plt.figure(1)
plt.suptitle("Technology Sector: Training Data Results")

plt.subplot(2,1,1)
plt.title("USA")
US_train_predictions = US_model.predict(US_X_train)
US_train_predictions = US_sc.inverse_transform(US_train_predictions)
US_y_train = US_sc.inverse_transform(US_y_train)
plt.plot(US_dates_train, US_train_predictions, label="Model Predictions")
plt.plot(US_dates_train, US_y_train, label="True Values")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
max = US_train_predictions.max() or US_y_train.max()
plt.ylim(0,max * 1.1)

plt.subplot(2,1,2)
plt.title("India")
IN_train_predictions = IN_model.predict(IN_X_train)
IN_train_predictions = IN_sc.inverse_transform(IN_train_predictions)
IN_y_train = IN_sc.inverse_transform(IN_y_train)
plt.plot(IN_dates_train, IN_train_predictions, label="Model Predictions")
plt.plot(IN_dates_train, IN_y_train, label="True Values")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price (INR)")
max = IN_train_predictions.max() or IN_y_train.max()
plt.ylim(0, max * 1.1)

plt.show()
input("Enter any button to continue")
plt.close()

# Validation data
plt.figure(2)
plt.suptitle("Technology Sector: Validation Data Results")

plt.subplot(2,1,1)
plt.title("USA")
US_val_predictions = US_model.predict(US_X_val)
US_val_predictions = US_sc.inverse_transform(US_val_predictions)
US_y_val = US_sc.inverse_transform(US_y_val)
plt.plot(US_dates_val, US_val_predictions, label="Model Predictions")
plt.plot(US_dates_val, US_y_val, label="True Values")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
max = US_val_predictions.max() or US_y_val.max()
plt.ylim(0, max * 1.1)

plt.subplot(2, 1, 2)
plt.title("India")
IN_val_predictions = IN_model.predict(IN_X_val)
IN_val_predictions = IN_sc.inverse_transform(IN_val_predictions)
IN_y_val = IN_sc.inverse_transform(IN_y_val)
plt.plot(IN_dates_val, IN_val_predictions, label="Model Predictions")
plt.plot(IN_dates_val, IN_y_val, label="True Values")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price (INR)")
max = IN_val_predictions.max() or IN_y_val.max()
plt.ylim(0, max * 1.1)

plt.show()
input("Enter any button to continue ")
plt.close()

# Testing data
plt.figure(3)
plt.suptitle("Technology Sector: Testing Data Results")

plt.subplot(2,1,1)
plt.title("USA")
US_test_predictions = US_model.predict(US_X_test)
US_test_predictions = US_sc.inverse_transform(US_test_predictions)
US_y_test = US_sc.inverse_transform(US_y_test)
plt.plot(US_dates_test, US_test_predictions, label="Model Predictions")
plt.plot(US_dates_test, US_y_test, label="True Values")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
max = US_test_predictions.max() or US_y_test.max()
plt.ylim(0, max * 1.1)

plt.subplot(2, 1, 2)
plt.title("India")
IN_test_predictions = IN_model.predict(IN_X_test)
IN_test_predictions = IN_sc.inverse_transform(IN_test_predictions)
IN_y_test = IN_sc.inverse_transform(IN_y_test)
plt.plot(IN_dates_test, IN_test_predictions, label="Model Predictions")
plt.plot(IN_dates_test, IN_y_test, label="True Values")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price (INR)")
max = IN_test_predictions.max() or IN_y_test.max()
plt.ylim(0, max * 1.1)

plt.show()
input("Enter any button to continue ")
plt.close()
