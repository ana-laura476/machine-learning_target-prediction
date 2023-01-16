from sklearn.model_selection import KFold
import pickle

# Load the best_params dictionary from the previous script
with open("best_params.pkl", "rb") as f:
    best_params = pickle.load(f)


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
        score = r2_score(y_test, y_pred)
        scores_for_target.append(score)
    mean_score = sum(scores_for_target) / len(scores_for_target)
    scores[target] = {'Mean R2': mean_score}
    # Make predictions on the molecule1_descriptors
    molecule1_descriptors = pd.read_csv("molecule1_descriptors.csv")
    X_molecule1 = molecule1_descriptors.iloc[:, :209]
    y_pred_molecule1 = model.predict(X_molecule1)
    pd.DataFrame(y_pred_molecule1).to_csv(f'prediction_for_molecule1_{target}.csv')
