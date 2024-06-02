import marimo

__generated_with = "0.6.6"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import altair as alt
    import pandas as pd
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    import csv
    from datetime import datetime

    return alt, csv, datetime, mo, np, pd, plt, tf


@app.cell
def __(pd):
    df = pd.read_csv("BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv")
    return df,


@app.cell
def __(df):
    df.head()
    return


@app.cell
def __(df):
    df_long = df.melt(("Date", "Currency"), var_name="Type", value_name="Price")

    # Determine type of each column
    df_long.dtypes
    return df_long,


@app.cell
def __(alt, df_long):
    chart = alt.Chart(df_long).mark_point().encode(
        x = "Date:T",
        y = "Price:Q",
        color="Type:N"
    ).interactive()
    chart
    return chart,


@app.cell
def __(csv, datetime):


    timesteps = []
    btc_price = []
    with open("BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv", "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        next(csv_reader) # skips first line (header)
        for line in csv_reader:
            timesteps.append(datetime.strptime(line[1], "%Y-%m-%d")) # date from csv
            btc_price.append(float(line[2])) # closing price as float
    # View first 10 of each
    timesteps[:10], btc_price[:10]
            
    return btc_price, csv_reader, f, line, timesteps


@app.cell
def __(btc_price, plt, timesteps):
    # Plot from CSV
    plt.figure(figsize=(10,7))
    plt.plot(timesteps, btc_price)
    plt.ylabel("BTC Price")
    plt.xlabel("Date")
    plt.title("Price of Bitcoin from 1 Oct 2013 to 18 May 2021")
    return


@app.cell
def __(mo):
    mo.md("###Creating train and test sets with time series data (the wrong way)")
    return


@app.cell
def __(df):
    df.head()
    return


@app.cell
def __(df, pd):
    #Get bitcoin date array
    bitcoin_prices = pd.DataFrame((df["Date"], df["Closing Price (USD)"])).rename({"Closing Price (USD)":"Price"}).transpose().set_index("Date")
    bitcoin_prices.head()
    return bitcoin_prices,


@app.cell
def __(bitcoin_prices):
    timesteps_series = bitcoin_prices.index.to_numpy()
    prices = bitcoin_prices["Price"].to_numpy()
    timesteps_series[:10], prices[:10]
    return prices, timesteps_series


@app.cell
def __(prices, timesteps_series):
    # Wrong way to make train/test sets for time series data
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(timesteps_series, prices, test_size=0.2, random_state=42)
    X_train.shape, X_test.shape, y_train.shape, y_test.shape
    return X_test, X_train, train_test_split, y_test, y_train


@app.cell
def __(X_test, X_train, plt, y_test, y_train):
    # Let's plot wrong train and test splits
    plt.figure(figsize=(10,7))
    plt.scatter(X_train, y_train, s=5, label="Train Data")
    plt.scatter(X_test, y_test, s=5, label="Test Data")
    plt.xlabel("Date")
    plt.ylabel("BTC Price")
    plt.xticks([])
    plt.legend(fontsize=14)
    plt.show()
    return


@app.cell
def __(mo):
    mo.md("###Create train & test sets for time series (the right way)")
    return


@app.cell
def __(prices, timesteps):
    # Create and train test splits the right way
    split_size = int(0.8 * len(prices)) # 80% train

    # Create train data splits (everythin before the split)
    X_train_c, y_train_c = timesteps[:split_size], prices[:split_size]

    # Create test data splits 
    X_test_c, y_test_c = timesteps[split_size:], prices[split_size:]

    len(X_train_c), len(X_test_c), len(y_train_c), len(y_test_c)
    return X_test_c, X_train_c, split_size, y_test_c, y_train_c


@app.cell
def __(mo):
    mo.md(
        """
        ##Create a Plotting Function
        Typing plotting code is tedious
        """
    )
    return


@app.cell
def __(np, plt):
    def plot_time_series(timesteps:np.array, values:np.array, format=".", start=0, end=None, label=None):
        """
        Plots timesteps against values

        Parameters
        ----------
        timesteps : array of timestep values
        values : array of values across time
        format : style of plot, default "."
        start : where to start the plot
        end : where to end the plot
        label : label to show on plot about values
        
        """
        plt.plot(timesteps[start:end], values[start:end], format, label=label)
        plt.xlabel("Time")
        plt.ylabel("BTC Price")
        if label:
            plt.legend(fontsize=14)
        plt.grid(True)
        
    return plot_time_series,


@app.cell
def __(X_test_c, X_train_c, plot_time_series, plt, y_test_c, y_train_c):
    plt.figure(figsize=(10,7))
    plot_time_series(X_train_c, y_train_c, label="Train data")
    plot_time_series(X_test_c, y_test_c, label="Test data")
    plt.show()
    return


@app.cell
def __(mo):
    mo.md(
        """
        ##Modeling Experiments
        **Horizon** = number of timesteps into the future we're going to predict\n
        **Window Size**= number of timesteps we're going to use to predict horizon\n
                | Model Number | Model Type | Horizon size | Window size | Extra data 
                | ----- | ----- | ----- | ----- | ----- |
                | 0 | Naïve model (baseline) | NA | NA | NA |
                | 1 | Dense model | 1 | 7 | NA |
                | 2 | Same as 1 | 1 | 30 | NA | 
                | 3 | Same as 1 | 7 | 30 | NA |
                | 4 | Conv1D | 1 | 7 | NA |
                | 5 | LSTM | 1 | 7 | NA |
                | 6 | Same as 1 (but with multivariate data) | 1 | 7 | Block reward size |
                | 7 | [N-BEATs Algorithm](https://arxiv.org/pdf/1905.10437.pdf) | 1 | 7 | NA |
                | 8 | Ensemble (multiple models optimized on different loss functions) | 1 | 7 | NA | 
                | 9 | Future prediction model (model to predict future values) | 1 | 7 | NA| 
                | 10 | Same as 1 (but with turkey 🦃 data introduced) | 1 | 7 | NA |
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        #Model 0: Naive Forcast
        $$\hat{y}_{t} = y_{t-1}$$

        In English:
        > The prediction at timestep t (y_hat) is equal to the value at timestep t-1 (previous timestep) - this is for a horizon of 1.
        """
    )
    return


@app.cell
def __(y_test_c):
    # Create a naive forcast
    naive_forcast = y_test_c[:-1]

    naive_forcast[-10:]
    return naive_forcast,


@app.cell
def __(y_test_c):
    y_test_c[-10:]
    return


@app.cell
def __(X_test_c, naive_forcast, plot_time_series, plt, y_test_c):
    plt.figure(figsize=(10,7))
    #plot_time_series(X_train_c, y_train_c, label="Training data")
    plot_time_series(X_test_c, y_test_c, start=300, format="-", label="Test_data")
    plot_time_series(X_test_c[1:], naive_forcast, start=300, format="-", label="Naive forcast")
    plt.show()
    return


@app.cell
def __(mo):
    mo.md("""
    ##Evaluating a times series model\n
    Form of regression problem\n
    Example of regression metrics: \n
    * MAE
    * MSE
    * Huber Loss (Combination of MSE and MAE)
    * RMSE
    * Mean absolute percentage error (MAPE/sMAPE)
    * Mean absolute scaled error (MASE)
    """)

    return


@app.cell
def __(tf):
    # MASE implementation
    def mean_absolute_scaled_error(y_true, y_pred):
        """
        Implement MASE (assuming no seasonality of data)
        """
        mae = tf.reduce_mean(tf.abs(y_true-y_pred))

        # Find MAE of naive forcast
        mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))

        return mae / mae_naive_no_season
    return mean_absolute_scaled_error,


@app.cell
def __(mean_absolute_scaled_error, naive_forcast, np, y_test_c):
    mean_absolute_scaled_error(np.array(y_test_c[1:], dtype=np.float32), np.array(naive_forcast, dtype=np.float32)).numpy()
    return


@app.cell
def __(naive_forcast, np, y_test_c, y_train_c):
    ytrain = np.array(y_train_c, dtype=np.float32)
    ytest = np.array(y_test_c, dtype=np.float32)
    naiveforcast = np.array(naive_forcast, dtype=np.float32)
    return naiveforcast, ytest, ytrain


@app.cell
def __(tf):
    print(dir(tf.keras.metrics))
    return


@app.cell
def __(mean_absolute_scaled_error, np, tf):
    # create function to return all evaluation metrics
    def evaluate_preds(y_true: np.array, y_pred: np.array)-> dict:
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # mae
        mae = tf.keras.metrics.MAE(y_true, y_pred)

        # mse
        mse = tf.keras.metrics.MSE(y_true, y_pred)

        # rmse
        rmse = tf.sqrt(mse)

        # mape
        mape = tf.keras.metrics.MAPE(y_true, y_pred)

        # mase
        mase = mean_absolute_scaled_error(y_true, y_pred)

        # Acount for different sized metrics (for longer horizons, reduce metrics to single value)
        if mae.ndim > 0:
            mae = tf.reduce_mean(mae)
            mse = tf.reduce_mean(mse)
            rmse = tf.reduce_mean(rmse)
            mape = tf.reduce_mean(mape)
            mase = tf.reduce_mean(mase)

        return {"mae" : mae.numpy(),
               "mse": mse.numpy(),
               "rmse" : rmse.numpy(),
               "mape" : mape.numpy(),
               "mase" : mase.numpy()}
    return evaluate_preds,


@app.cell
def __(evaluate_preds, naiveforcast, ytest):
    naive_results = evaluate_preds(ytest[1:], naiveforcast)
    naive_results
    return naive_results,


@app.cell
def __(tf, ytest):
    tf.reduce_mean(ytest)
    return


@app.cell
def __(tf, ytest):
    tf.reduce_min(ytest), tf.reduce_max(ytest)
    return


@app.cell
def __(naive_results):
    naive_results.keys()
    return


@app.cell
def __(naive_results, plt):
    import random
    fig, ax = plt.subplots()
    ax.bar(naive_results.keys(), naive_results.values(), color=[(random.choice(range(255))/255,random.choice(range(255))/255,random.choice(range(255))/255) for _ in range(len(naive_results.keys()))])
    ax.set_yscale('log')
    ax.set_xlabel("Evaluation Metrics")
    ax.set_ylabel("Error")
    ax.set_title("Bitcoin Forcast Evaluation Metrics")
    plt.show()

    return ax, fig, random


@app.cell
def __(mo):
    mo.md(
        """
        ##Format Data Part 2: Windowing our dataset

        We window our time series dataset to turn our data into a supervised learning problem\n
        Windowing for one week\n
        ```python
        [0, 1, 2, 3, 4, 5, 6] -> [7]
        [1, 2, 3, 4, 5, 6, 7] -> [8]
        [2, 3, 4, 5, 6, 7, 8] -> [9]
        ```
        """
    )
    return


@app.cell
def __(ytrain):
    len(ytrain)
    return


@app.cell
def __(btc_price):
    btc_price[:7], btc_price[7]
    return


@app.cell
def __():
    # Need to write a windowing function 
    # Let's setup global variable for window and horizon size
    HORIZON = 1 # predict next day
    WINDOW_SIZE = 7 # use past 7 days to make prediction
    return HORIZON, WINDOW_SIZE


@app.cell
def __(HORIZON):
    # Create a function to label windowed data
    def get_labeled_window(x, horizon=HORIZON):
        """
        Create labels for windowed dataset
        e.g. if horizon = 1
        Input: [0, 1, 2, 3, 4, 5, 6, 7] -> Ouput: ([0, 1, 2, 3, 4, 5, 6], [7])
        """

        return x[:,:-horizon], x[:, -horizon:]
    return get_labeled_window,


@app.cell
def __(get_labeled_window, tf):
    # Test window labeling function
    test_window, test_label = get_labeled_window(tf.expand_dims(tf.range(8), axis=0))
    print(f"Window: {tf.squeeze(test_window).numpy()} -> Label: {tf.squeeze(test_label).numpy()}")
    return test_label, test_window


@app.cell
def __(mo):
    mo.md(
        """
        We've got a way to label our windowed data. \n However, this only works on a small scale. \n We could do this with for loops, but that's quite slow\n Let's use numpy indexing \n
        Our function will: \n
        1. Create a window step of specific window size \n
        2. Use NumPy indexing to create a 2D array of multiple window steps \n
        3. Uses the 2D array of multiple window steps to index on target series \n
        4. Uses our `get_labeled_window()` function to create training and target sets
        """
    )
    return


@app.cell
def __(HORIZON, WINDOW_SIZE, get_labeled_window, np):
    # Create function to view NumPy arrays as windows
    def make_windows(x, window_size=WINDOW_SIZE, horizon=HORIZON, printinternals=False):
        """
        Turns a 1D array into 2D array of sequential labeled windows of window_size with horizon size labels
        """
        # 1. Create window of specific window_size
        window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)
        # 2. Use Numpy index to create 2D array of multiple window steps
        window_indexes = window_step + np.expand_dims(np.arange(len(x) - (window_size+horizon-1)), axis=0).T # create 2D array of window size

         # 3. Index on the target array (a time series) with 2D array of multiple window steps
        windowed_array = x[window_indexes]

        # 4. Get the labeled windows
        windows, labels = get_labeled_window(windowed_array, horizon=horizon)
        
        if printinternals:
            print(f"Window Step: {window_step, window_step.shape}")
            print(f"Second half: {np.expand_dims(np.arange(len(x) - (window_size+horizon-1)), axis=0).T, np.expand_dims(np.arange(len(x) - (window_size+horizon-1)), axis=0).T.shape}")
            print(f"Window indexes:\n {window_indexes, window_indexes.shape}")
            print(f"Windowed array: {windowed_array}")
        return windows, labels
        
    return make_windows,


@app.cell
def __(HORIZON, WINDOW_SIZE, make_windows, prices):
    full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON, printinternals=True)
    return full_labels, full_windows


@app.cell
def __(full_labels, full_windows):
    len(full_windows), len(full_labels)
    return


@app.cell
def __(full_labels, full_windows):
    # view the first 3 windows/labels
    for i in range(3):
        print(f"Window: {full_windows[i]} -> Label: {full_labels[i]}")
    return i,


@app.cell
def __(mo):
    mo.md("> You can try tf.keras.preprocessing.timeseries_dataset_from_array")
    return


@app.cell
def __(mo):
    mo.md("##Turning windows into training and test sets")
    return


@app.cell
def __(np):
    # make the train/test splits
    def make_train_test_splits(windows, labels, test_split=0.2):
        """
        Splits matching paris of windows and labels into train test splits
        """
        split_size = int(len(windows)*(1-test_split))
        train_windows = np.array(windows[:split_size], dtype=np.float32)
        train_labels = np.array(labels[:split_size], dtype=np.float32)
        test_windows = np.array(windows[split_size:], dtype=np.float32)
        test_labels = np.array(labels[split_size:], dtype=np.float32)
        return train_windows, test_windows, train_labels, test_labels
    return make_train_test_splits,


@app.cell
def __(full_labels, full_windows, make_train_test_splits):
    # Create train and test window
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
    len(train_windows), len(test_windows), len(train_labels), len(test_labels)
    return test_labels, test_windows, train_labels, train_windows


@app.cell
def __(HORIZON, WINDOW_SIZE, np, train_labels, ytrain):
    np.array_equal(np.squeeze(train_labels[:-HORIZON-1]), ytrain[WINDOW_SIZE:])
    return


@app.cell
def __(mo):
    mo.md("##Make a modelling checkpoint")
    return


@app.cell
def __():
    from tensorflow.keras.callbacks import ModelCheckpoint
    return ModelCheckpoint,


@app.cell
def __(ModelCheckpoint):
    help(ModelCheckpoint)
    return


@app.cell
def __(ModelCheckpoint):
    import os # create folder for models saved via checkpoint

    # Create a function to implement a ModelCheckpoint callback with specific filename
    def create_model_checkpoint(model_name, save_path="model_experiments"):
        return ModelCheckpoint(filepath=os.path.join(save_path, model_name)+".keras", verbose=0, save_best_only=True)
    return create_model_checkpoint, os


@app.cell
def __(mo):
    mo.md("##Model 0: Dense model (window = 7, horizon = 1)")
    return


@app.cell
def __(
    HORIZON,
    create_model_checkpoint,
    test_labels,
    test_windows,
    tf,
    train_labels,
    train_windows,
):
    from tensorflow.keras import layers

    # Set random seed for a reproducible results as possible
    tf.random.set_seed(42)

    # 1. Construct model
    model_0 = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(HORIZON, activation="linear")
    ], name="model_0_dense")

    # 2. Compile Model
    model_0.compile(loss=tf.keras.losses.MAE,
                   optimizer=tf.keras.optimizers.Adam(),
                   metrics=["mae", "mse"])

    # 3. Fit the model
    model_0.fit(x=train_windows,
               y=train_labels,
               epochs=100,
               verbose=1,
               batch_size=128,
               validation_data=(test_windows, test_labels),
               callbacks=[create_model_checkpoint(model_name=model_0.name)])
    return layers, model_0


@app.cell
def __(model_0, test_labels, test_windows):
    # Evaluate model on test data
    model_0.evaluate(test_windows, test_labels)
    return


@app.cell
def __(test_labels, test_windows, tf):
    # load in best performing model
    model_0_best = tf.keras.models.load_model("model_experiments/model_0_dense.keras")
    model_0_best.evaluate(test_windows, test_labels)
    return model_0_best,


@app.cell
def __(mo):
    mo.md("##Making forcasts with the model (on the test dataset")
    return


@app.cell
def __(tf):
    def make_preds(model, input_data):
        """
        Uses model to make predictions on input_data
        """
        forcast = model.predict(input_data)
        return tf.squeeze(forcast)
    return make_preds,


@app.cell
def __(make_preds, model_0, test_windows):
    # Make predictions using model_1 on the test dataset 
    model_0_preds = make_preds(model_0, test_windows)
    return model_0_preds,


@app.cell
def __(model_0_preds):
    len(model_0_preds), model_0_preds[:10]
    return


@app.cell
def __(evaluate_preds, model_0_preds, test_labels, tf):
    model_0_results = evaluate_preds(tf.squeeze(test_labels), model_0_preds)
    return model_0_results,


@app.cell
def __(model_0_results):
    model_0_results
    return


@app.cell
def __(naive_results):
    naive_results
    return


@app.cell
def __(
    X_test_c,
    model_0_preds,
    plot_time_series,
    plt,
    test_labels,
    test_windows,
):
    # Let's plot model results
    offset = 450
    plt.figure(figsize=(10,7))
    # Account for the test_window offset and index into test_labels to ensure correct plotting
    plot_time_series(timesteps=X_test_c[-len(test_windows):], values=test_labels[:,0], start=offset, label="Test Data")
    plot_time_series(timesteps=X_test_c[-len(test_windows):], values=model_0_preds, start=offset, format="-", label="model_0_preds")
    plt.show()
    return offset,


@app.cell
def __(mo):
    mo.md("#Model 1: Dense (window = 30, horizon = 1)")
    return


@app.cell
def __():
    HORIZON_new = 1
    WINDOW_SIZE_new = 30
    return HORIZON_new, WINDOW_SIZE_new


@app.cell
def __(HORIZON_new, WINDOW_SIZE_new, make_windows, prices):
    # Make window data with appropriate horizon and window sizes
    full_windows_new, full_labels_new = make_windows(prices, window_size=WINDOW_SIZE_new, horizon=HORIZON_new)
    len(full_windows_new), len(full_labels_new)
    return full_labels_new, full_windows_new


@app.cell
def __(full_labels_new, full_windows_new, make_train_test_splits):
    train_windows_new, test_windows_new, train_labels_new, test_labels_new = make_train_test_splits(windows=full_windows_new, labels=full_labels_new, test_split=0.2)
    len(train_windows_new), len(test_windows_new), len(train_labels_new), len(test_labels_new)
    return (
        test_labels_new,
        test_windows_new,
        train_labels_new,
        train_windows_new,
    )


@app.cell
def __(
    HORIZON_new,
    create_model_checkpoint,
    layers,
    test_labels_new,
    test_windows_new,
    tf,
    train_labels_new,
    train_windows_new,
):
    model_1 = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(HORIZON_new, activation="linear")
    ], name="model_1_dense")

    model_1.compile(loss="mae",
                   optimizer=tf.keras.optimizers.Adam())

    model_1.fit(train_windows_new,
               train_labels_new,
               epochs=100,
               batch_size=128,
               verbose=0,
               validation_data=(test_windows_new, test_labels_new),
               callbacks=[create_model_checkpoint(model_name=model_1.name)])
    return model_1,


@app.cell
def __(model_1, test_labels_new, test_windows_new):
    model_1.evaluate(test_windows_new, test_labels_new)
    return


@app.cell
def __(test_labels_new, test_windows_new, tf):
    model_1_best = tf.keras.models.load_model("model_experiments/model_1_dense.keras")
    model_1_best.evaluate(test_windows_new, test_labels_new)
    return model_1_best,


@app.cell
def __(make_preds, model_1, test_windows_new):
    model_1_preds = make_preds(model_1, test_windows_new)
    return model_1_preds,


@app.cell
def __(evaluate_preds, model_1_preds, test_labels_new, tf):
    model_1_results = evaluate_preds(tf.squeeze(test_labels_new), model_1_preds)
    model_1_results
    return model_1_results,


@app.cell
def __(model_0_results):
    model_0_results
    return


@app.cell
def __(
    X_test_c,
    model_1_preds,
    plot_time_series,
    plt,
    test_labels_new,
    test_windows_new,
):
    offset1 = 450
    plt.figure(figsize=(10,7))
    plot_time_series(X_test_c[-len(test_windows_new):], values=test_labels_new[:,0], start=offset1, label="test data")
    plot_time_series(X_test_c[-len(test_windows_new):], values=model_1_preds, start=offset1, format="-", label="model_1_preds")
    plt.show()
    return offset1,


@app.cell
def __(mo):
    mo.md("# Model 3: Dense (window = 30, horizon = 7)")
    return


@app.cell
def __(make_windows, prices):
    HORIZON3 = 7
    WINDOW_SIZE3 = 30

    full_windows3, full_labels3 = make_windows(prices, window_size=WINDOW_SIZE3, horizon=HORIZON3)
    return HORIZON3, WINDOW_SIZE3, full_labels3, full_windows3


@app.cell
def __(full_labels3, full_windows3, make_train_test_splits):
    train_windows3, test_windows3, train_labels3, testlabels3 = make_train_test_splits(full_windows3, full_labels3)
    return test_windows3, testlabels3, train_labels3, train_windows3


@app.cell
def __(
    HORIZON3,
    create_model_checkpoint,
    layers,
    test_windows3,
    testlabels3,
    tf,
    train_labels3,
    train_windows3,
):
    tf.random.set_seed(42)

    model_3 = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(HORIZON3, activation="linear")
    ], name="model_3_dense")

    model_3.compile(loss="mae",
                   optimizer=tf.keras.optimizers.Adam())

    model_3.fit(train_windows3,
               train_labels3,
               batch_size=128,
               epochs=100,
               verbose=0,
               validation_data=(test_windows3, testlabels3),
               callbacks=[create_model_checkpoint(model_name=model_3.name)])
    return model_3,


@app.cell
def __(model_3, test_windows3, testlabels3):
    # Evaluate model_3
    model_3.evaluate(test_windows3, testlabels3)
    return


@app.cell
def __(test_windows3, testlabels3, tf):
    model_3_best = tf.keras.models.load_model("model_experiments/model_3_dense.keras")
    model_3_best.evaluate(test_windows3, testlabels3)
    return model_3_best,


@app.cell
def __(make_preds, model_3_best, test_windows3):
    model_3_preds = make_preds(model_3_best, test_windows3)

    return model_3_preds,


@app.cell
def __(evaluate_preds, model_3_preds, testlabels3, tf):
    model_3_results = evaluate_preds(tf.squeeze(testlabels3), model_3_preds) 
    model_3_results
    return model_3_results,


@app.cell
def __(model_1_results):
    model_1_results
    return


@app.cell
def __(model_0_results):
    model_0_results
    return


@app.cell
def __(
    X_test_c,
    model_3_preds,
    plot_time_series,
    plt,
    test_windows3,
    testlabels3,
    tf,
):
    offset3 = 100
    plt.figure(figsize=(12,7))
    plot_time_series(timesteps=X_test_c[-len(test_windows3):], values=testlabels3[:,0], start=offset3, format="-", label="Test data")
    plot_time_series(timesteps=X_test_c[-len(test_windows3):], values=tf.reduce_mean(model_3_preds, axis=1), start=offset3, label="model_3_preds") # condeses 7 values into 1
    plt.show()

    return offset3,


@app.cell
def __(
    model_0_results,
    model_1_results,
    model_3_results,
    naive_results,
    pd,
):
    pd.DataFrame({"naive" : naive_results["mae"],
                 "horizon_1_window_7": model_0_results["mae"],
                 "horizon_1_window_30": model_1_results["mae"],
                 "horizon_7_window_30" : model_3_results["mae"]}, index=["mae"]).plot(figsize=(10,7), kind="bar")
    return


@app.cell
def __(mo):
    mo.md("#Model 4: Conv1D")
    return


@app.cell
def __():
    HORIZON_conv = 1
    WINDOW_SIZE_conv = 7
    return HORIZON_conv, WINDOW_SIZE_conv


@app.cell
def __(HORIZON_conv, WINDOW_SIZE_conv, make_windows, prices):
    # Create window dataset
    full_windows_conv, full_labels_conv = make_windows(prices, window_size=WINDOW_SIZE_conv, horizon=HORIZON_conv)
    len(full_labels_conv)
    return full_labels_conv, full_windows_conv


@app.cell
def __(full_labels_conv, full_windows_conv, make_train_test_splits):
    # Create train/test set
    train_windows_conv, test_windows_conv, train_labels_conv, test_labels_conv = make_train_test_splits(full_windows_conv, full_labels_conv)
    len(train_windows_conv)
    return (
        test_labels_conv,
        test_windows_conv,
        train_labels_conv,
        train_windows_conv,
    )


@app.cell
def __(mo):
    mo.md("For Conv1D need input shape of: `(batch_size, timesteps, input_dim)`")
    return


@app.cell
def __(train_windows_conv):
    print(train_windows_conv[0].shape)
    return


@app.cell
def __(tf, train_windows_conv):
    # Need to reshape data
    x = tf.constant(train_windows_conv[0])
    x
    return x,


@app.cell
def __(layers, tf):
    expand_dims_layer = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))
    return expand_dims_layer,


@app.cell
def __(expand_dims_layer, x):
    # Test out our lambda layer
    print(f"Original Shape: {x.shape}")
    print(f"Expanded shape: {expand_dims_layer(x).shape}")
    print(f"Original values with expanded shape:\n {expand_dims_layer(x)}")
    return


@app.cell
def __(layers):
    help(layers.Conv1D)
    return


@app.cell
def __(
    HORIZON_conv,
    create_model_checkpoint,
    expand_dims_layer,
    layers,
    test_labels_conv,
    test_windows_conv,
    tf,
    train_labels_conv,
    train_windows_conv,
):
    tf.random.set_seed(42)
    # Create conv1d
    model_4 = tf.keras.Sequential([
        expand_dims_layer,
        layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="causal", activation="relu"),
        layers.Dense(HORIZON_conv)
    ], name="model_4_conv1D")
    # lambda layer
    # Conv1D model, filter=128, kernel_size=, padding...
    # output layer = Dense

    model_4.compile(loss="mae",
            optimizer=tf.keras.optimizers.Adam())

    model_4.fit(train_windows_conv,
               train_labels_conv,
               batch_size=128,
               epochs=100,
               verbose=1,
               validation_data=(test_windows_conv, test_labels_conv),
               callbacks=[create_model_checkpoint(model_name=model_4.name)])
    return model_4,


@app.cell
def __(model_4):
    model_4.summary()
    return


@app.cell
def __(model_4, test_labels_conv, test_windows_conv):
    model_4.evaluate(test_windows_conv, test_labels_conv)
    return


@app.cell
def __(tf):
    help(tf.keras.models.load_model)
    return


@app.cell
def __(make_preds, model_4, test_windows_conv):
    # Make predictions
    model_4_preds = make_preds(model_4, test_windows_conv)
    return model_4_preds,


@app.cell
def __(evaluate_preds, model_4_preds, test_labels_conv, tf):
    model_4_results = evaluate_preds(tf.squeeze(test_labels_conv), model_4_preds)
    model_4_results
    return model_4_results,


@app.cell
def __(model_0_results):
    model_0_results
    return


@app.cell
def __(mo):
    mo.md("#Model 5: RNN (LSTM)")
    return


@app.cell
def __(
    HORIZON_conv,
    WINDOW_SIZE_conv,
    create_model_checkpoint,
    layers,
    test_labels_conv,
    test_windows_conv,
    tf,
    train_labels_conv,
    train_windows_conv,
):
    tf.random.set_seed(42)

    # functional api
    inputs = layers.Input(shape=(WINDOW_SIZE_conv,))
    x1 = tf.keras.layers.Reshape((1, WINDOW_SIZE_conv))(inputs)
    x1 = layers.LSTM(128, activation="relu", return_sequences=True)(x1)
    x1 = layers.LSTM(128, activation="relu")(x1)
    x1 = layers.Dense(32, activation="relu")(x1)
    output = layers.Dense(HORIZON_conv)(x1)

    model_5 = tf.keras.Model(inputs=inputs, outputs=output, name="model_5_LSTM")

    model_5.compile(loss="mae",
                   optimizer=tf.keras.optimizers.Adam())

    model_5.fit(train_windows_conv,
               train_labels_conv,
               epochs=100,
               verbose=1,
               validation_data=(test_windows_conv, test_labels_conv),
               callbacks=[create_model_checkpoint(model_name=model_5.name)])
    return inputs, model_5, output, x1


@app.cell
def __(tf):
    # Load in best version of model_5
    model_5_best = tf.keras.models.load_model("model_experiments/model_5_LSTM.keras")
    return model_5_best,


@app.cell
def __(model_5_best, test_labels_conv, test_windows_conv):
    model_5_best.evaluate(test_windows_conv, test_labels_conv)
    return


@app.cell
def __(make_preds, model_5, test_windows_conv):
    model_5_preds = make_preds(model_5, test_windows_conv)
    return model_5_preds,


@app.cell
def __(evaluate_preds, model_5_preds, test_labels_conv, tf):
    # Evalutate predictions
    model_5_results = evaluate_preds(tf.squeeze(test_labels_conv), model_5_preds)
    model_5_results
    return model_5_results,


@app.cell
def __(mo):
    mo.md("## Make a Multivariate Time Series")
    return


@app.cell
def __(bitcoin_prices):
    # Let's make a multivariate time series
    bitcoin_prices.head()
    return


@app.cell
def __(np):
    # Let's add the halving data to the time series
    block_reward_1 = 50 # 3 Jan 2009 - not in our dataset
    block_reward_2 = 25 # 8 Nov 2012
    block_reward_3 = 12.5 # 9 July 2016
    block_reward_4 = 6.25 # 18 May 2020

    # Block reward dates
    block_reward_2_datetime = np.datetime64("2012-11-28")
    block_reward_3_datetime = np.datetime64("2016-07-09")
    block_reward_4_datetime = np.datetime64("2020-05-18")

    print(block_reward_2_datetime)

    return (
        block_reward_1,
        block_reward_2,
        block_reward_2_datetime,
        block_reward_3,
        block_reward_3_datetime,
        block_reward_4,
        block_reward_4_datetime,
    )


@app.cell
def __(bitcoin_prices, pd):
    # Add in block_reward values as feature in dataframe
    bitcoin_prices.index = pd.to_datetime(bitcoin_prices.index)
    return


@app.cell
def __(bitcoin_prices):
    bitcoin_prices.index
    return


@app.cell
def __(bitcoin_prices, block_reward_3_datetime, block_reward_4_datetime):
    # Create data ranges of where specific block_reward values should be
    block_reward_2_days = (block_reward_3_datetime - bitcoin_prices.index[0]).days
    block_reward_3_days = (block_reward_4_datetime - bitcoin_prices.index[0]).days
    block_reward_2_days, block_reward_3_days
    return block_reward_2_days, block_reward_3_days


@app.cell
def __(bitcoin_prices):
    bitcoin_prices_block = bitcoin_prices.copy()
    bitcoin_prices_block["block_reward"] = None
    bitcoin_prices_block
    return bitcoin_prices_block,


@app.cell
def __(
    bitcoin_prices_block,
    block_reward_2,
    block_reward_2_days,
    block_reward_3,
    block_reward_3_days,
    block_reward_4,
):
    bitcoin_prices_block.iloc[:block_reward_2_days, -1] = block_reward_2
    bitcoin_prices_block.iloc[block_reward_2_days:block_reward_3_days, -1] = block_reward_3
    bitcoin_prices_block.iloc[block_reward_3_days:, -1] = block_reward_4

    bitcoin_prices_block.head()
    return


@app.cell
def __(bitcoin_prices_block, pd):
    # Plot the block reward vs price over time
    from sklearn.preprocessing import minmax_scale
    scaled_price_block_df = pd.DataFrame(minmax_scale(bitcoin_prices_block[["Price", "block_reward"]]), columns=bitcoin_prices_block.columns, index=bitcoin_prices_block.index)
    scaled_price_block_df.plot(figsize=(10,7))
    return minmax_scale, scaled_price_block_df


@app.cell
def __(mo, scaled_price_block_df):
    table = mo.ui.table(scaled_price_block_df)
    table
    return table,


@app.cell
def __(mo):
    mo.md("## Making a windowed dataset with pandas")
    return


@app.cell
def __():
    # pandas.DataFrame.shift()
    # Set up dataset hyperparameters

    return


@app.cell
def __(HORIZON):
    HORIZON
    return


@app.cell
def __(WINDOW_SIZE):
    WINDOW_SIZE
    return


@app.cell
def __(bitcoin_prices_block):
    bitcoin_prices_block.head()
    return


@app.cell
def __(WINDOW_SIZE, bitcoin_prices_block):
    # Make a copy of the Bitcoin historical data with block reward feature
    bitcoin_prices_windowed = bitcoin_prices_block.copy()

    # Add windowed columns
    for j in range(WINDOW_SIZE): # shift values for each step in window size
        bitcoin_prices_windowed[f"Price+{j+1}"] = bitcoin_prices_windowed["Price"].shift(periods=j+1)

    bitcoin_prices_windowed.head()
    return bitcoin_prices_windowed, j


@app.cell
def __(bitcoin_prices_windowed, np):
    # Create X (windows) and y (horizons) features
    X = bitcoin_prices_windowed.dropna().drop("Price", axis=1).astype(np.float32)
    y = bitcoin_prices_windowed.dropna()["Price"].astype(np.float32)
    X.head()
    return X, y


@app.cell
def __(y):
    y.head()
    return


@app.cell
def __(mo):
    mo.md("##Create Train and Test Sets")
    return


@app.cell
def __(X, y):
    split_size1 = int(len(X) * 0.8)
    X_train1, y_train1 = X[:split_size1], y[:split_size1]
    X_test1, y_test1 = X[split_size1:], y[split_size1:]
    print(len(X_train1), len(y_train1), len(X_test1), len(y_test1))
    return X_test1, X_train1, split_size1, y_test1, y_train1


@app.cell
def __(
    HORIZON,
    X_test1,
    X_train1,
    create_model_checkpoint,
    layers,
    tf,
    y_test1,
    y_train1,
):
    tf.random.set_seed(42)

    model_6 = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(HORIZON)
    ], name="model_6_dense_multivariate")

    model_6.compile(loss="mae",
                   optimizer=tf.keras.optimizers.Adam())

    model_6.fit(X_train1, y_train1, epochs=100, verbose=1, validation_data=(X_test1, y_test1), callbacks=[create_model_checkpoint(model_name=model_6.name)])
    return model_6,


@app.cell
def __(X_test1, model_6, y_test1):
    # Evaluate
    model_6.evaluate(X_test1, y_test1)
    return


@app.cell
def __(tf):
    model_6_best = tf.keras.models.load_model("model_experiments/model_6_dense_multivariate.keras")
    return model_6_best,


@app.cell
def __(X_test1, model_6_best, y_test1):
    model_6_best.evaluate(X_test1, y_test1)
    return


@app.cell
def __(X_test1, model_6_best, tf):
    model_6_preds = tf.squeeze(model_6_best.predict(X_test1))
    model_6_preds[:10]
    return model_6_preds,


@app.cell
def __(evaluate_preds, model_6_preds, y_test1):
    model_6_results = evaluate_preds(y_test1, model_6_preds)
    model_6_results
    return model_6_results,


@app.cell
def __(model_1_results):
    model_1_results
    return


@app.cell
def __(naive_results):
    naive_results
    return


@app.cell
def __(mo):
    mo.md("## [N-BEATS Algorithm: Neural basis expansion analysis for interpretable time series forcasting (univariate)](https://arxiv.org/abs/1905.10437)")
    return


@app.cell
def __(mo):
    mo.md("Use TensorFlow sub-classing to create custom layers")
    return


@app.cell
def __():
    ## Building and testing the N-BEAST block layer
    return


@app.cell
def __(tf):
    # Create NBEATBlock custom layer
    class NBeatsBlock(tf.keras.layers.Layer):
        def __init__(self, input_size: int, theta_size: int, horizon: int, n_neurons:int, n_layers: int, **kwargs):
            super().__init__(**kwargs)
            self.input_size = input_size
            self.theta_size = theta_size
            self.horizon = horizon
            self.n_neurons = n_neurons
            self.n_layers = n_layers

            # Block contains stack of 4 FC layers with ReLU activation
            self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
            # Output of block is a theta layer with Linear activation
            self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

        def call(self, inputs):
            x= inputs
            for layer in self.hidden:
                x = layer(x)
            theta = self.theta_layer(x)
            # Output the backcast and the forcast 
            backcast, forecast = theta[:,:self.input_size], theta[:, -self.horizon:]

            return backcast, forecast
        
    return NBeatsBlock,


@app.cell
def __():
    # Let's test NBeatsBlock class
    return


@app.cell
def __(HORIZON, NBeatsBlock, WINDOW_SIZE):
    # Set up dummy layer to represent inputs and outputs
    dummy_nbeats_block_layer = NBeatsBlock(input_size=WINDOW_SIZE,
                                          theta_size=WINDOW_SIZE+HORIZON,
                                          horizon=HORIZON,
                                          n_neurons=128, 
                                           n_layers=4)
    return dummy_nbeats_block_layer,


@app.cell
def __(WINDOW_SIZE, tf):
    dummy_inputs = tf.expand_dims(tf.range(WINDOW_SIZE)+1, axis=0) # has to reflect input layer dims
    dummy_inputs
    return dummy_inputs,


@app.cell
def __(dummy_inputs, dummy_nbeats_block_layer, tf):
    # Pass dummy inputs to dummy NBeatsBlock layer
    backcast, forecast = dummy_nbeats_block_layer(dummy_inputs)
    # These are the activation output of the theta layer
    print(f"Backcast: {tf.squeeze(backcast.numpy())}")
    print(f"Forcast: {tf.squeeze(forecast.numpy())}")
    return backcast, forecast


@app.cell
def __(mo):
    mo.md("### Preparing data for the N-BEATs algorithm using `tf.data`")
    return


@app.cell
def __():
    HORIZON1 = 1
    WINDOW_SIZE1 = 7
    return HORIZON1, WINDOW_SIZE1


@app.cell
def __(bitcoin_prices):
    # Create NBEATs data inputs (N-BEATs works with univariate time series)
    bitcoin_prices.head()
    return


@app.cell
def __(WINDOW_SIZE, bitcoin_prices):
    # Add windowed columns
    bitcoin_prices_nbeats = bitcoin_prices.copy()
    for k in range(WINDOW_SIZE):
        bitcoin_prices_nbeats[f"Price+{k+1}"] = bitcoin_prices_nbeats["Price"].shift(periods=k+1)
    bitcoin_prices_nbeats.head()
    return bitcoin_prices_nbeats, k


@app.cell
def __(bitcoin_prices_nbeats, np):
    # Make features and labels
    X_nbeats = bitcoin_prices_nbeats.dropna().drop("Price", axis=1).astype(np.float64)
    y_nbeats = bitcoin_prices_nbeats.dropna()["Price"].astype(np.float64)

    # Split data into train and test sets
    split_size_nbeats = int(len(X_nbeats)*0.8)
    X_train_nbeats, y_train_nbeats = X_nbeats[:split_size_nbeats], y_nbeats[:split_size_nbeats]
    X_test_nbeats, y_test_nbeats = X_nbeats[split_size_nbeats:], y_nbeats[split_size_nbeats:]
    print(len(X_train_nbeats), len(y_train_nbeats), len(X_test_nbeats), len(y_test_nbeats))
    return (
        X_nbeats,
        X_test_nbeats,
        X_train_nbeats,
        split_size_nbeats,
        y_nbeats,
        y_test_nbeats,
        y_train_nbeats,
    )


@app.cell
def __(X_test_nbeats, X_train_nbeats, tf, y_test_nbeats, y_train_nbeats):
    # Make dataset performant using tf.data API
    train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train_nbeats, name="train_features")
    train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train_nbeats, name="train_labels")

    test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test_nbeats, name="test_features")
    test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test_nbeats, name="test_labels")

    # combine labels and features by zipping together
    train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset), name="train_dataset")

    test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset), name="test_dataset")

    BATCH_SIZE = 1024
    train_dataset = train_dataset.batch(BATCH_SIZE, name="train_batch").prefetch(tf.data.AUTOTUNE, name="train_prefetch")
    test_dataset = test_dataset.batch(BATCH_SIZE, name="test_batch").prefetch(tf.data.AUTOTUNE, name="test_prefetch")

    train_dataset, test_dataset
    return (
        BATCH_SIZE,
        test_dataset,
        test_features_dataset,
        test_labels_dataset,
        train_dataset,
        train_features_dataset,
        train_labels_dataset,
    )


@app.cell
def __(mo):
    mo.md("## Setting up hyperparameters")
    return


@app.cell
def __(HORIZON, WINDOW_SIZE):
    # Values from N-Beats paper
    N_EPOCHS = 5000
    N_NEURONS = 512
    N_LAYERS = 4
    N_STACKS = 30


    INPUT_SIZE = WINDOW_SIZE * HORIZON # paper uses ensemble of models, we are only using 1
    THETA_SIZE = INPUT_SIZE + HORIZON

    INPUT_SIZE, THETA_SIZE
    return INPUT_SIZE, N_EPOCHS, N_LAYERS, N_NEURONS, N_STACKS, THETA_SIZE


@app.cell
def __(mo):
    mo.md("### Getting ready for residual connections")
    return


@app.cell
def __(tf):
    # Make tensors
    tensor_1 = tf.range(10) + 10
    tensor_2 = tf.range(10)
    tensor_1, tensor_2
    return tensor_1, tensor_2


@app.cell
def __(layers, tensor_1, tensor_2):
    # Subtract
    subtracted = layers.subtract([tensor_1, tensor_2]) # subtracts tensors

    # Add
    added = layers.add([tensor_1, tensor_2]) # adds tensors

    subtracted, added
    return added, subtracted


@app.cell
def __(mo):
    mo.md("## Building, compiling, and fitting the N-BEATS algorithm")
    return


@app.cell
def __(mo):
    mo.md("""
    1. Setup and instance of the N-BEATs block layer using `NBeatsBlock`
    2. Create an input layer for the N-BEATs stack (residual connections & functional API)
    3. Make the initial backcast and forecasts for the model
    4. Use for loop to create stacks of block layers
    5. use the `NBeatsBlock` class within for loop in 4 to create blocks
    6. Create the double residual stacking using subtract and add layers
    7. Put the model inputs and output together
    8. Compile the model with MAE loss and Adam optimizer and lr = 0.001
    9. Fit the N-BEATs model for 5000 epochs with a couple of callbacks: early stopping and reduce lr on plateau 
    """)
    return


@app.cell
def __(
    HORIZON,
    HORIZON1,
    INPUT_SIZE,
    NBeatsBlock,
    N_EPOCHS,
    N_LAYERS,
    N_NEURONS,
    N_STACKS,
    THETA_SIZE,
    layers,
    test_dataset,
    tf,
    train_dataset,
):
    tf.random.set_seed(42)

    # 1. Setup and instance of NBeatsBlock layer
    nbeats_block_layer = NBeatsBlock(input_size=INPUT_SIZE,
                                    theta_size=THETA_SIZE,
                                    horizon=HORIZON1,
                                    n_neurons=N_NEURONS,
                                    n_layers=N_LAYERS,
                                    name="InitialBlock")
    # 2. Create and input layer to stack
    stack_input = layers.Input(shape=(INPUT_SIZE,), name="stack_input")

    # 3. Create initial backcast and forecast input
    residuals, forecast_model = nbeats_block_layer(stack_input)

    # 4. Create stacks of block layers
    for l, _ in enumerate(range(N_STACKS-1)): # first stack already created
        # 5. use the nbeatsblock to calculate backcast and forecast
        backcast_model, block_forecast = NBeatsBlock(input_size=INPUT_SIZE,
                                              theta_size=THETA_SIZE,
                                              horizon=HORIZON,
                                              n_neurons=N_NEURONS,
                                              n_layers=N_LAYERS,
                                              name=f"BeatsBlock_{l}")(residuals)
        # 6. Create the double residual stacking
        residuals = layers.subtract([residuals, backcast_model], name=f"substract_{l}")
        forecast_model = layers.add([forecast_model, block_forecast], name=f"add_{l}")

    # 7. Put together inputs and outputs
    model_7 = tf.keras.Model(inputs=stack_input, outputs=forecast_model, name="model_7_nbeats")

    # 8. Compile
    model_7.compile(loss="mae", optimizer=tf.keras.optimizers.Adam())

    # 9. Fit model with callbacks
    model_7.fit(train_dataset,
               epochs=N_EPOCHS,
               validation_data=test_dataset,
               verbose=0,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                           patience=200,
                                                    restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                             patience=100,
                                                             verbose=1)])
    return (
        backcast_model,
        block_forecast,
        forecast_model,
        l,
        model_7,
        nbeats_block_layer,
        residuals,
        stack_input,
    )


@app.cell
def __(model_7, test_dataset):
    # Evaluate N-Beats model on the test dataset
    model_7.evaluate(test_dataset)
    return


@app.cell
def __(evaluate_preds, make_preds, model_7, test_dataset, y_test_nbeats):
    model_7_preds = make_preds(model_7, test_dataset)
    model_7_results = evaluate_preds(y_test_nbeats, model_7_preds)
    return model_7_preds, model_7_results


@app.cell
def __(model_7_results):
    model_7_results
    return


@app.cell
def __(mo):
    mo.md("### Plotting the N-Beats architecture we've created")
    return


@app.cell
def __(model_7):
    # Plot the N-BEATs model
    from tensorflow.keras.utils import plot_model
    plot_model(model_7, to_file="model_7_nbeats.png")
    return plot_model,


@app.cell
def __(mo):
    mo.image(src="model_7_nbeats.png")
    return


@app.cell
def __(mo):
    mo.md("# Model 8: Creating an ensemble (stacking different models together)")
    return


@app.cell
def __(HORIZON, WINDOW_SIZE):
    HORIZON, WINDOW_SIZE
    return


@app.cell
def __(mo):
    mo.md("### Constructing and fitting and ensemble of models (using different loss functions")
    return


@app.cell
def __(HORIZON, layers, test_dataset, tf, train_dataset):
    def get_ensemble_models(horizon=HORIZON, train_data=train_dataset, test_data=test_dataset, num_iter=10, num_epochs=1000, loss_fun=["mae", "mse", "mape"]): # num_iter = # of models
        """
        Returns a list of num_iter models each trained on MAE, MSE, and MAPE loss
        """
        # Make empty list for ensemble models
        ensemble_models = []

        # Create num_iter number of models per loss function
        for o in range(num_iter):
            for loss_function in loss_fun:
                print(f"Optimizing model by reducing: {loss_function} for {num_epochs}, model number: {o}")

                # construct simple model 
                model = tf.keras.Sequential([
                    layers.Dense(128, kernel_initializer="he_normal",activation="relu"),
                    layers.Dense(128,kernel_initializer="he_normal", activation="relu"),
                    layers.Dense(HORIZON, activation="linear")
                ])
                model.compile(loss=loss_function,
                             optimizer=tf.keras.optimizers.Adam(),
                             metrics=["mae", "mse"])
                model.fit(train_data,
                         epochs=num_epochs,
                         verbose=0,
                         validation_data=test_data,
                         callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                                    patience=200,
                                                                    restore_best_weights=True),
                                   tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                                        patience=100,
                                                                        verbose=1)])
                ensemble_models.append(model)
        return ensemble_models
    return get_ensemble_models,


@app.cell
def __(get_ensemble_models):
    ensemble_models = get_ensemble_models(num_iter=5)
    return ensemble_models,


@app.cell
def __(ensemble_models):
    len(ensemble_models)
    return


@app.cell
def __(mo):
    mo.md("## Make predictions with ensemble model")
    return


@app.cell
def __(tf):
    # Create a function which uses list of trained models to make and return a list of predictions
    def make_ensemble_preds(ensemble_models, data):
        ensemble_preds=[]
        for model in ensemble_models:
            preds = model.predict(data)
            ensemble_preds.append(preds)
        return tf.constant(tf.squeeze(ensemble_preds))
    return make_ensemble_preds,


@app.cell
def __(ensemble_models, make_ensemble_preds, test_dataset):
    ensemble_preds = make_ensemble_preds(ensemble_models, data=test_dataset)
    return ensemble_preds,


@app.cell
def __(ensemble_preds):
    ensemble_preds
    return


@app.cell
def __(ensemble_preds):
    ensemble_preds.shape
    return


@app.cell
def __(ensemble_preds, tf):
    # Evaluate preds
    ensembed_combined_preds = tf.keras.ops.median(ensemble_preds, axis=0)
    return ensembed_combined_preds,


@app.cell
def __(ensembed_combined_preds):
    ensembed_combined_preds
    return


@app.cell
def __(ensembed_combined_preds, evaluate_preds, y_test_nbeats):
    ensemble_results = evaluate_preds(y_test_nbeats, ensembed_combined_preds)
    ensemble_results
    return ensemble_results,


@app.cell
def __(naive_results):
    naive_results
    return


@app.cell
def __(model_0_results):
    model_0_results
    return


@app.cell
def __(mo):
    mo.md("### Plotting the prediction intervals (uncertainty estimates) of our ensemble")
    return


@app.cell
def __(ensemble_preds, tf):
    # Measure standard deviation of predictions 
    ensemble_std = tf.math.reduce_std(ensemble_preds, axis=0)*1.96 # 1.96 because want 95% of data
    ensemble_mean = tf.math.reduce_mean(ensemble_preds, axis=0)
    ensemble_mean.shape, ensemble_std.shape
    return ensemble_mean, ensemble_std


@app.cell
def __(ensemble_mean, ensemble_std):
    upperbound = ensemble_mean+ensemble_std
    lowerbound = ensemble_mean-ensemble_std
    upperbound.shape, lowerbound.shape
    return lowerbound, upperbound


@app.cell
def __(
    X_test_nbeats,
    ensemble_mean,
    lowerbound,
    plt,
    upperbound,
    y_test_nbeats,
):
    # Plot mean with prediction intervals
    offset2=500
    plt.figure(figsize=(10,7))
    plt.plot(X_test_nbeats.index[offset2:], y_test_nbeats[offset2:], "g", label="Test data")
    plt.plot(X_test_nbeats.index[offset2:], ensemble_mean[offset2:], "k-", label="Ensemble Mean")
    plt.xlabel("Date")
    plt.ylabel("BTC Price")
    # Upper and lower bounds
    plt.fill_between(X_test_nbeats.index[offset2:], lowerbound[offset2:], upperbound[offset2:], label="Prediction Intervals")
    plt.legend(loc="upper left", fontsize=14)
    return offset2,


@app.cell
def __(mo):
    mo.md("Models predictions have been lagging behind the test data...")
    return


@app.cell
def __(mo):
    mo.md("# Model 9: Train a model on the full historical data to make predictions into the future")
    return


@app.cell
def __(bitcoin_prices_windowed):
    bitcoin_prices_windowed.tail()
    return


@app.cell
def __(bitcoin_prices_windowed, np):
    # Train model on entire data to make predictions for the next day
    X_all = bitcoin_prices_windowed.dropna().drop(["Price", "block_reward"], axis=1).to_numpy().astype(np.float64)

    y_all = bitcoin_prices_windowed.dropna()["Price"].to_numpy().astype(np.float64)

    len(X_all), len(y_all)
    return X_all, y_all


@app.cell
def __(X_all, y_all):
    X_all[:5], y_all[:5]
    return


@app.cell
def __(X_all, tf, y_all):
    # Turn into tensors
    features_dataset_all = tf.data.Dataset.from_tensor_slices(X_all)
    labels_dataset_all = tf.data.Dataset.from_tensor_slices(y_all)

    dataset_all = tf.data.Dataset.zip((features_dataset_all, labels_dataset_all))

    BATCH_SIZE_all = 1024

    dataset_all = dataset_all.batch(BATCH_SIZE_all).prefetch(tf.data.AUTOTUNE)

    dataset_all
    return (
        BATCH_SIZE_all,
        dataset_all,
        features_dataset_all,
        labels_dataset_all,
    )


@app.cell
def __(HORIZON, dataset_all, layers, tf):
    model_9 = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(HORIZON)
    ])

    model_9.compile(loss="mae",
                   optimizer=tf.keras.optimizers.Adam())

    model_9.fit(dataset_all,
               epochs=100,
               verbose=0)
    return model_9,


@app.cell
def __(mo):
    mo.md("### Make predictions into the future")
    return


@app.cell
def __():
    # How many timesteps to predict into the future
    INTO_FUTURE = 14
    return INTO_FUTURE,


@app.cell
def __(INTO_FUTURE, WINDOW_SIZE, np, tf):
    # 1. Create function to make predictions into future
    def make_future_forcasts(values, model, into_future, window_size=WINDOW_SIZE) -> list:
        future_forecast = []
        last_window = values[-window_size:]

        for _ in range(INTO_FUTURE):
            # Predict on last window then append it again, again, ..., again
            # Make forecast on it's own forecasts
            future_pred = model.predict(tf.expand_dims(last_window, axis=0))
            print(f"Predicting on:\n {last_window} -> Prediction: {tf.squeeze(future_pred).numpy()}\n")

            future_forecast.append(tf.squeeze(future_pred).numpy())

            # Update last window with new pred and get window_size most recent preds
            last_window = np.append(last_window, future_pred)[-window_size:]
        return future_forecast
    return make_future_forcasts,


@app.cell
def __(INTO_FUTURE, WINDOW_SIZE, make_future_forcasts, model_9, y_all):
    # Make forecasts into future
    future_forecast = make_future_forcasts(y_all, model_9, INTO_FUTURE, WINDOW_SIZE)
    return future_forecast,


@app.cell
def __(bitcoin_prices_windowed):
    bitcoin_prices_windowed.tail(5)
    return


@app.cell
def __(mo):
    mo.md("###Plot future forecasts")
    return


@app.cell
def __(future_forecast):
    future_forecast
    return


@app.cell
def __(np):
    def get_future_dates(start_date, into_future, offset=1):
        start_date = start_date + np.timedelta64(offset, "D") # specify start date, "D" stands for date
        end_date = start_date + np.timedelta64(into_future, "D") 
        return np.arange(start_date, end_date, dtype="datetime64[D]")
    return get_future_dates,


@app.cell
def __(bitcoin_prices):
    # last timestep of timesteps
    last_timestep = bitcoin_prices.index[-1]
    type(last_timestep)
    return last_timestep,


@app.cell
def __(INTO_FUTURE, get_future_dates, last_timestep):
    # Get next two weeks of timesteps
    next_time_steps = get_future_dates(start_date=last_timestep,
                                      into_future=INTO_FUTURE)
    next_time_steps
    return next_time_steps,


@app.cell
def __(btc_price, future_forecast, last_timestep, next_time_steps, np):
    # Insert last timesetp/final price into next time steps and future forcasts
    next_time_steps_insert = np.insert(next_time_steps, 0, last_timestep)
    future_forecast_insert = np.insert(future_forecast, 0, btc_price[-1])
    return future_forecast_insert, next_time_steps_insert


@app.cell
def __(
    bitcoin_prices,
    btc_price,
    future_forecast_insert,
    next_time_steps_insert,
    plot_time_series,
    plt,
):
    # Plot future price predictions of Bitcoin
    plt.figure(figsize=(10,7))
    plot_time_series(bitcoin_prices.index, btc_price, start=2500, format="-", label="Actual BTC Price")
    plot_time_series(next_time_steps_insert, future_forecast_insert, format="-", label="Predicted BTC Price")
    plt.show()
    return


@app.cell
def __(mo):
    mo.md("## Compare models")
    return


@app.cell
def __(
    ensemble_results,
    model_0_results,
    model_1_results,
    model_3_results,
    model_4_results,
    model_5_results,
    model_6_results,
    model_7_results,
    naive_results,
    pd,
):
    model_results = pd.DataFrame({"naive_model": naive_results,
                                 "model_1_dense_w7_h1": model_0_results,
                                 "model_2_dense_w30_h1": model_1_results,
                                "model_3_dense_w30_h7": model_3_results,
                                 "model_4_CONV1D": model_4_results,
                                 "model_5_LSTM": model_5_results,
                                 "model_6_multivariate": model_6_results,
                                 "model_7_NBEATS": model_7_results,
                                 "model_8_ensemble": ensemble_results}).T
    return model_results,


@app.cell
def __(model_results):
    model_results
    return


@app.cell
def __(model_results):
    # Sort model results by mae
    model_results[["mae"]].sort_values(by="mae").plot(figsize=(10,20), kind="bar")
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
