import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from joblib import dump
import pickle

# Load the data
df = pd.read_csv('input_data.csv')

# Get the list of targets
targets = df['target'].unique()
best_params = {}

# Loop over the targets
for target in targets:
    data = df.query(f'target=={target}')
    X = data.drop("activity", axis=1)
    y = data["activity"]
    seed = 42
    # Tune the model's hyperparameters
    param_dist = {'n_estimators':[50, 100, 250],
                  'max_depth': [3, 5, 10, None],
                  'max_features': ['sqrt', 'log2', 'auto']}
    rnd_search = RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_dist, n_iter=10, cv=5, iid=False)
    rrnd_search = RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_dist, n_iter=10, cv=5, iid=False)
    rnd_search.fit(X, y)
    best_params[target] = rnd_search.best_params_

# Save the best_params to a file
with open("best_params.pkl", "wb") as f:
    pickle.dump(best_params, f)
