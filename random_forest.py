from sklearn.model_selection import KFold
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Load the best_params dictionary from the previous script
best_params = joblib.load("best_params.joblib")
# Load the data
df = pd.read_csv('input_data.csv')
df['target'] = df['target'].str.extract('(\d+)').astype(int)


scores = {}

# Loop over the targets
for target in best_params.keys():
    data = df.query(f'target=={target}')
    X = data.drop("activity", axis=1)
    y = data["activity"]
    seed = 42
    # Initialize the model with the best hyperparameters for the current target
    model = RandomForestRegressor(**best_params[target], random_state=seed)
    # Initialize KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    scores_for_target = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # fit the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # evaluate the model and store the score
        score = r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred)
        scores_for_target.append(score)
    mean_score = sum(scores_for_target) / len(scores_for_target)
    scores[target] = {'Mean R2': mean_score}

    # Make predictions on the molecule1_descriptors
    molecule1_descriptors = pd.read_csv("molecule1_descriptors.txt", sep="\t", header = 0)
    molecule1_descriptors = df[df.columns[1:]]
    y_pred_molecule1 = model.predict(molecule1_descriptors)
    pd.DataFrame(y_pred_molecule1).to_csv(f'prediction_for_molecule1_{target}.csv')
