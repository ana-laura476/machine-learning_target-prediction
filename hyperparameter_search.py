import pandas as pd
from joblib import dump
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump


# Load the data
df = pd.read_csv('input_data.csv')
print(df)
df['target'] = df['target'].str.extract('(\d+)').astype(int)

print(df)


# Get the list of targets
targets = df['target'].unique()
print(targets)

best_params = {}

# Loop over the targets
for target in targets:
    # Select data for current target
    data = df[df['target'] == target]
    print(data.head())
    print(df.info())
    print(df['activity'].unique())
    X = data.drop("activity", axis = 1)
    y = data["activity"]

    # Define hyperparameter grid
    param_grid = {'n_estimators': [50, 100, 250],
                  'max_depth': [5, 10, 15, None],
                  'max_features': ['sqrt', 'log2', 'auto']}

    # Instantiate and fit the model
    rnd_search = RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_grid, n_iter=10, cv=10, return_train_score=True)
    rnd_search.fit(X, y)
    best_params[target] = rnd_search.best_params_
    #get the model metrics
    r2 = rnd_search.cv_results_['mean_test_score']
    std = rnd_search.cv_results_['std_test_score']

    y_pred = rnd_search.predict(X)

    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    with open('metrics.txt', 'a') as f:
        f.write(f'Target: {target}, R2: {r2}, MSE: {mse}, MAE: {mae}\n')

# Save the best_params to a file
dump(best_params, 'best_params.joblib')
joblib.dump(best_params, "best_params.pkl", compress=True)

