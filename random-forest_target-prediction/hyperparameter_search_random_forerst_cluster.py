import pandas as pd
import os 
import csv
import numpy as np
from joblib import dump, Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans


def cluster_split(X, n_splits=10, seed=42):
    kmeans = KMeans(n_clusters=n_splits, random_state=seed)
    clusters = kmeans.fit_predict(X)

    all_indices = np.arange(len(X))
    splits = []

    for cluster in range(n_splits):
        test_index = np.where(clusters == cluster)[0]
        train_index = np.setdiff1d(all_indices, test_index)
        splits.append((train_index, test_index))

    return splits

def run_model(seed):
    # Function to calculate metrics
    def calculate_metrics(model, X, y, train_index, test_index):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return r2, mse, mae
        
    # Load the data and target codes
    df = pd.read_csv('input_data_with_smiles.csv', delimiter=",") 
    df['target'] = df['target'].str.extract('(\d+)').astype(int)
    print(df)

    # Get the list of targets
    targets = df['target'].unique()
    print(targets)

    best_params = {}
    # Number of cores to use for parallel processing
    n_jobs = -1

    if not os.path.exists(f"all_models_seed_cluster{seed}"):
        os.mkdir(f"all_models_seed_cluster{seed}")

    # Loop over the targets
    for target in targets:
        # Select data for current target
        data = df[df['target'] == target]
        # Remove duplicates
        data = data.drop_duplicates()
        # Check if total number of entries is over 50
        if data.shape[0] < 50:
            continue
        print(data.head())


        X = data.drop(["activity", 'target', 'molecule', 'smiles', 'chembl ID'], axis=1)
        print(X.columns)
        print(X.shape)
 

        y = data["activity"]
        kfold = cluster_split(X, n_splits=10, seed=seed)
        cluster_sizes = [len(test_index) for train_index, test_index in kfold]
        print("Cluster sizes:", cluster_sizes)
        if min(cluster_sizes) < 5:
        	print(f"Target {target} has a cluster with fewer than 3 molecules")
        	continue

        # Define hyperparameter grid
        param_grid = {'n_estimators': list(range(10, 160, 20)),
                      'max_depth': [4,6,8,10],
                      'max_features': ['log2', 'sqrt', 0.25, 0.5]}

        # Instantiate and fit the model
        rnd_search = RandomizedSearchCV(RandomForestRegressor(random_state=seed), param_distributions=param_grid, n_iter=20, cv=kfold, random_state=seed, n_jobs=n_jobs, scoring='neg_mean_absolute_error')
        rnd_search.fit(X, y)
        best_params[target] = rnd_search.best_params_
        model = rnd_search.best_estimator_
        dump(model, f'all_models_seed_cluster{seed}/model_cluster{target}_seed{seed}.joblib')

        #get the model metrics and perform 10-fold cross-validaiton and get the predicted values
        results = Parallel(n_jobs=n_jobs)(delayed(calculate_metrics)(model, X, y, train_index, test_index) for train_index, test_index in kfold)
        r2_values, mse_values, mae_values = zip(*results)
        r2_mean = np.mean(r2_values)
        r2_std = np.std(r2_values)
        mse_mean = np.mean(mse_values)
        mse_std = np.std(mse_values)
        mae_mean = np.mean(mae_values)
        mae_std = np.std(mae_values)
        
        with open(f'metrics_seed_cluster{seed}.txt', 'a') as f:
            f.write('target\tr2\tstd r2\tmse\tstd mse\tmae\tstd mae\n')
        with open(f'metrics_seed_cluster{seed}.txt', 'a') as f:
            f.write(f'{target}\t{r2_mean}\t{r2_std}\t{mse_mean}\t{mse_std}\t{mae_mean}\t{mae_std}\n')
    # Save the best_params and models to a file
    dump(best_params, f'best_params_seed_cluster{seed}.joblib')

# Call the function with the different seed values
for seed in [126, 168]: #42, 84, 126, 168
    run_model(seed)




