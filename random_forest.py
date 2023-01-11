import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV


# Load the data
dataset = pd.read_csv('input_data.csv')
df = dataset.rename(columns={"log (Standard value (10^-9) (Mean)": "activity_normalized", "Target ChEMBL ID": "protein_target"})
targets = df['protein_target'].unique()

# Initialize an empty dictionary to store the results
scores = {}

# Loop over the targets
for target in targets:
    data = df.query(f'protein_target=={target}')
    X = data.drop("activity_normalized", axis=1)
    y = data["activity_normalized"]
    seed = 42
    model = RandomForestRegressor(max_features='sqrt', random_state=seed)
    # Perform 10-fold cross validation and store the mean accuracy
    score = cross_val_score(model, X, y, cv=10, random_state=seed,scoring='r2')
    scores[target] = {
                        'Mean R2': score.mean(),
                        'Std':score.std()
                        }
param_grid = {'n_estimators':[50, 100, 250],
              'max_depth': [3, 5, 10, None],
              'max_features': ['sqrt', 'log2']}
grid_search = GridSearchCV(model, param_grid, cv=10,scoring='r2',random_state=seed)
grid_search.fit(X, y)
print('Best parameters: ', grid_search.best_params_)

eval_metrics = {'Target': target,
                'Mean R2': score.mean(),
                'Std': score.std(),
                'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
                'Mean Squared Error': mean_squared_error(y_test, y_pred)}
eval_df = pd.DataFrame(eval_metrics, index=[0])
eval_df.to_csv(f'{output_dir}/target_{target}_eval.csv', index=False)


# Fit the classifier on the full dataset for the current target
model.fit(X, y)
from joblib import dump
...
dump(model, f'{output_dir}/target_{target}_model.joblib')

molecule1_descriptors = pd.read_csv("molecule1_descriptors.csv")
X_molecule1 = molecule1_descriptors.iloc[:, :209]
for target in targets:
    #Loading the best model
    best_model = RandomForestRegressor(**grid_search.best_params_, random_state=seed)
    best_model.fit(X, y)
    y_pred_molecule1 = best_model.predict(X_molecule1)
pd.DataFrame(y_pred_molecule1).to_csv('prediction_for_molecule1.csv')


