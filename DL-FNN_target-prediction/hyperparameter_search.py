from joblib import dump, Parallel, delayed
from functools import partial
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import os
import random
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load the data and target codes
df = pd.read_csv('input_data.csv', delimiter=",")
print(df)
df['target'] = df['target'].str.extract('(\d+)').astype(int)
print(df)


# Get the list of targets
targets = df['target'].unique()
print(targets)

best_params = {}

seed = 42
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
if not os.path.exists("all_models"):
    os.mkdir("all_models")
#build the function
def create_model(activation, n_neurons, n_hidden_layers, learning_rate, input_dropout, dropout):
    model = Sequential()
    model.add(Dense(n_neurons, activation=activation, input_shape=(X.shape[1],)))
    model.add(tf.keras.layers.Dropout(input_dropout))
   
    for _ in range(n_hidden_layers):
        model.add(Dense(n_neurons, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout))
       
    model.add(Dense(1, activation='linear'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_absolute_error', metrics=['mean_absolute_error'])
    return model

def calculate_metrics(model, X, y, kfold):
    r2_scores = []
    mse_scores = []
    mae_scores = []
   
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train, epochs=20, batch_size=batch_size)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2_scores.append(r2)
        mse_scores.append(mse)
        mae_scores.append(mae)
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)
    mse_mean = np.mean(mse_scores)
    mse_std = np.std(mse_scores)
    mae_mean = np.mean(mae_scores)
    mae_std = np.std(mae_scores)
    
    return r2_mean, r2_std, mse_mean, mse_std, mae_mean, mae_std

# Get the list of targets
targets = df['target'].unique()
print(targets)

#define the 10-fold cross validation
# Loop over the targets
for target in targets:
    print(f'the target is {target}')
   
    # Select data for current target
    data = df[df['target'] == target]
    print(data.head())
    X = data.drop(["activity", 'target', 'molecule'], axis=1)
    print(X)
    print(X.shape)
    y = data["activity"]
    y = y.values.reshape(-1, 1)
    print(y)
    print(y.shape)
    #define hyperparameter grid
    param_grid = {
    'activation': ['relu', 'selu'],
    'n_neurons': [16, 32, 64],
    'n_hidden_layers': [1, 2, 3],
    'learning_rate': [0.01, 0.1],
    'input_dropout': [0, 0.2, 0.5],
    'dropout': [0.2, 0.5]}
    batch_size=int((len(X))*0.2)

    #define model
    model = KerasRegressor(model=create_model, epochs=20, batch_size=batch_size, verbose=0, activation=None, n_neurons=None, n_hidden_layers=None, learning_rate=None, input_dropout=None, dropout=None, random_state=seed)
    print(model.get_params().keys())
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, cv=kfold, n_jobs=-1, random_state=seed, verbose=2, scoring='neg_mean_absolute_error')
    random_search.fit(X, y)
    best_params[target] = random_search.best_params_
    print("Best estimator:", random_search.best_estimator_)  # Add this line to check the best_estimator_ object
    best_model=random_search.best_estimator_
    dump(model, f'all_models/model_{target}.joblib')
    r2_mean, r2_std, mse_mean, mse_std, mae_mean, mae_std = calculate_metrics(best_model, X, y, kfold)
    with open('metrics.txt', 'a') as f:
        f.write('target\tr2\tstd r2\tmse\tstd mse\tmae\tstd mae\n')
    with open('metrics.txt', 'a') as f:
        f.write(f'{target}\t{r2_mean}\t{r2_std}\t{mse_mean}\t{mse_std}\t{mae_mean}\t{mae_std}\n')
# Save the best_params and models to a file
dump(best_params, 'best_params.joblib')
