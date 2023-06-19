import pandas as pd
import os 
import csv
import numpy as np
from joblib import dump, Parallel, delayed
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, WhiteKernel
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import preprocessing



# Load the data and target codes
df = pd.read_csv('input_data.csv', delimiter=",")
df['target'] = df['target'].str.extract('(\d+)').astype(int)
print(df)
print(df['5'])


# Get the list of targets
targets = df['target'].unique()
X = df.drop(["activity", 'target', 'molecule'], axis=1)
print(X)
df=df[["molecule", 'target', 'activity']]
print(df)
X = preprocessing.normalize(X, norm='l2')
X = pd.DataFrame(X)
print(X)
df=pd.concat([df, X], axis=1, verify_integrity=True)
print(df)
print(df[5])

best_params = {}
seed = 42
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


# Number of cores to use for parallel processing
n_jobs = -1
def calculate_metrics(model, X_target, y, train_index, test_index):
    X_train, X_test = X_target.iloc[train_index], X_target.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return r2, mse, mae
if not os.path.exists("all_models"):
    os.mkdir("all_models")

    
# Loop over the targets
for target in targets:
    print(df)
    # Select data for current target
    data = df[df['target'] == target]
    print(data.head())
    print(df.info())
    X_target = data.drop(["activity", 'target', 'molecule'], axis=1)
    print(X_target)
    print(len(X_target))
    print(X_target.shape)
    y = data["activity"]


    # Define hyperparameter grid
    param_grid = {'kernel': [RBF(length_scale_bounds='fixed'), None, Matern(length_scale_bounds='fixed', nu=1.5), WhiteKernel(noise_level_bounds='fixed')],
                  'alpha': [1e-12, 10e-10, 10e-8, 1e-4],
                  'normalize_y': [True, False]}

    # Instantiate and fit the model
    rnd_search = RandomizedSearchCV(GaussianProcessRegressor(random_state=seed), param_distributions=param_grid, n_iter=20, cv=kfold, n_jobs=n_jobs, random_state=seed, scoring='neg_mean_absolute_error')
    rnd_search.fit(X_target, y)
    best_params[target] = rnd_search.best_params_
    model = rnd_search.best_estimator_
    dump(model, f'all_models/model_{target}.joblib')

    #get the model metrics and perform 10-fold cross-validaiton and get the predicted values
    results = Parallel(n_jobs=1)(delayed(calculate_metrics)(model, X_target, y, train_index, test_index) for train_index, test_index in kfold.split(X_target))
    r2_values, mse_values, mae_values = zip(*results)
    r2_mean = np.mean(r2_values)
    r2_std = np.std(r2_values)
    mse_mean = np.mean(mse_values)
    mse_std = np.std(mse_values)
    mae_mean = np.mean(mae_values)
    mae_std = np.std(mae_values)
        
    with open('metrics.txt', 'a') as f:
        f.write('target\tr2\tstd r2\tmse\tstd mse\tmae\tstd mae\n')
    with open('metrics.txt', 'a') as f:
        f.write(f'{target}\t{r2_mean}\t{r2_std}\t{mse_mean}\t{mse_std}\t{mae_mean}\t{mae_std}\n')
# Save the best_params and models to a file
dump(best_params, 'best_params.joblib')
