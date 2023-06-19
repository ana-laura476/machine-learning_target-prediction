import pandas as pd
import os 
import csv
import numpy as np
from joblib import dump, Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import subprocess
from rdkit import Chem
import pandas as pd
import os 
import csv
import numpy as np
from joblib import dump, Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import subprocess
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, MACCSkeys, AllChem, RDKFingerprint
from rdkit.Avalon import pyAvalonTools
from rdkit import Chem
import pandas, numpy
from multiprocessing import Pool, Manager
import multiprocessing
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from tqdm import tqdm

class FingerprintGenerator:

    def __init__(self):
        print("FingerprintGenerator geladen")

    def generate_fingerprints(self, data, lengthPerFP):
        length = lengthPerFP
        fp_dict = {}
        radii = [2, 4, 6, 8]
        morgan_radii = [0, 2, 4, 6]
        fp_dict['RDKit_2'] = self.__generate_boolArray(self.__generateFingerprints_RDKit(data, 2, length))
        fp_dict['RDKit_4'] = self.__generate_boolArray(self.__generateFingerprints_RDKit(data, 4, length))
        fp_dict['RDKit_6'] = self.__generate_boolArray(self.__generateFingerprints_RDKit(data, 6, length))
        fp_dict['RDKit_8'] = self.__generate_boolArray(self.__generateFingerprints_RDKit(data, 8, length))

        fp_dict['RDKit_linear_2'] = self.__generate_boolArray(self.__generateFingerprints_RDKitlinear(data, 2, length))
        fp_dict['RDKit_linear_4'] = self.__generate_boolArray(self.__generateFingerprints_RDKitlinear(data, 4, length))
        fp_dict['RDKit_linear_6'] = self.__generate_boolArray(self.__generateFingerprints_RDKitlinear(data, 6, length))
        fp_dict['RDKit_linear_8'] = self.__generate_boolArray(self.__generateFingerprints_RDKitlinear(data, 8, length))

        fp_dict['MorganCircle_0'] = self.__generate_boolArray(self.__generateFingerprints_Morgan_Circular(data, 0, length))
        fp_dict['MorganCircle_2'] = self.__generate_boolArray(self.__generateFingerprints_Morgan_Circular(data, 2, length))
        fp_dict['MorganCircle_4'] = self.__generate_boolArray(self.__generateFingerprints_Morgan_Circular(data, 4, length))
        fp_dict['MorganCircle_6'] = self.__generate_boolArray(self.__generateFingerprints_Morgan_Circular(data, 6, length))

        fp_dict['MorganCircle_feature_0'] = self.__generate_boolArray(self.__generateFingerprints_Morgan_Circular_Feature(data, 0, length))
        fp_dict['MorganCircle_feature_2'] = self.__generate_boolArray(self.__generateFingerprints_Morgan_Circular_Feature(data, 2, length))
        fp_dict['MorganCircle_feature_4'] = self.__generate_boolArray(self.__generateFingerprints_Morgan_Circular_Feature(data, 4, length))
        fp_dict['MorganCircle_feature_6'] = self.__generate_boolArray(self.__generateFingerprints_Morgan_Circular_Feature(data, 6, length))

        fp_dict['Layerd_2'] = self.__generate_boolArray(self.__generateFingerprints_LayerdFingerprint(data, 2, length))
        fp_dict['Layerd_4'] = self.__generate_boolArray(self.__generateFingerprints_LayerdFingerprint(data, 4, length))
        fp_dict['Layerd_6'] = self.__generate_boolArray(self.__generateFingerprints_LayerdFingerprint(data, 6, length))
        fp_dict['Layerd_8'] = self.__generate_boolArray(self.__generateFingerprints_LayerdFingerprint(data, 8, length))

        fp_dict['Avalon'] = self.__generate_boolArray(self.__generateFingerprints_Avalon(data, length))

        fp_dict['MACCS'] = self.__generate_boolArray(self.__generateFingerprints_MACCS_keys(data))
        fp_dict['AtomPairs'] = self.__generate_boolArray(self.__generateFingerprints_Atom_Pairs(data, length))
        fp_dict['TopologicalTorsions'] = self.__generate_boolArray(self.__generateFingerprints_Topological_Torsions(data, length))

        return fp_dict

    def __generate_boolArray(self, fp):
        string = fp.ToBitString()
        boolarr = []
        for c in string:
            if c == '1':
                boolarr.append(1) #True
            else:
                boolarr.append(-1) #False

        return boolarr

    def __getEmptyBitVector(self, length):
        bitvector = ExplicitBitVect(length)
        return bitvector

    def __generateFingerprints_RDKit(self, data, maxPath, length):
        if data == 'NA':
            return self.__getEmptyBitVector(length)
        fp = Chem.RDKFingerprint(mol=data, maxPath=maxPath, fpSize=length) 
        return fp

    def __generateFingerprints_RDKitlinear(self, data, maxPath, length):
        if data == 'NA':
            return self.__getEmptyBitVector(length)
        fp = Chem.RDKFingerprint(mol=data, maxPath=maxPath, branchedPaths=False, fpSize=length)
        return fp

    def __generateFingerprints_Atom_Pairs(self, data, length):
        if data == 'NA':
            return self.__getEmptyBitVector(length)
        return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(data, nBits=length)

    def __generateFingerprints_Topological_Torsions(self, data, length):
        if data == 'NA':
            return self.__getEmptyBitVector(length)
        return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(data, nBits=length)

    def __generateFingerprints_MACCS_keys(self, data):
        if data == 'NA':
            return self.__getEmptyBitVector(167)
        return MACCSkeys.GenMACCSKeys(data)

    def __generateFingerprints_Morgan_Circular(self, data, r, length):
        if data == 'NA':
            return self.__getEmptyBitVector(length)
        return AllChem.GetMorganFingerprintAsBitVect(data, r, nBits=length)

    def __generateFingerprints_Morgan_Circular_Feature(self, data, r, length):
        if data == 'NA':
            return self.__getEmptyBitVector(length)
        return AllChem.GetMorganFingerprintAsBitVect(data, r, useFeatures=True, nBits=length)

    def __generateFingerprints_Avalon(self, data, bitlength):
        if data == 'NA':
            return self.__getEmptyBitVector(bitlength)
        return pyAvalonTools.GetAvalonFP(data, nBits=bitlength)

    def __generateFingerprints_LayerdFingerprint(self, data, r, bitlength):
        if data == 'NA':
            return self.__getEmptyBitVector(bitlength)
        return Chem.LayeredFingerprint(data, maxPath=r, fpSize=bitlength)

def reformat_feature_MFF(feature_list):
    fingerprints = list(feature_list[0].keys())
    MulitF_arr = []
    for feature in feature_list:
        temp_row = []
        for finger in fingerprints:
            temp_row.extend(feature[finger])
        MulitF_arr.append(temp_row)
    return MulitF_arr


# Load the data and target codes
FP_length = 3096
df = pd.read_csv('input_data_MFF.csv', delimiter=",")
df['target'] = df['target'].str.extract('(\d+)').astype(int)

print(df)
print(df.shape)
# Get the list of targets
targets = df['target'].unique()
print(targets)

best_params = {}
seed = 42
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


# Number of cores to use for parallel processing
n_jobs = -1
def calculate_metrics(model, X, y, train_index, test_index):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return r2, mse, mae
if not os.path.exists("all_models_MFF"):
    os.mkdir("all_models_MFF")
exclude_targets = [251, 203, 244, 204, 218, 261, 205, 233, 237, 217, 279, 4822, 240, 2971, 253, 4296, 220, 234, 226, 325, 260, 4005, 228, 214, 224, 235, 5145, 4409]
# Loop over the targets
for target in [t for t in targets if t not in exclude_targets]:
    # Select data for current target
    print('this is the target', target)
    data = df[df['target'] == target]
    y = data["activity"]
    print(data.head())
    # Get the data for the current target
    smiles = data['smiles']
    print(len(smiles))
    data = [Chem.MolFromSmiles(smile) for smile in smiles]
    FG = FingerprintGenerator()
    mff_array = []
    for d in tqdm(data):
        fingerprints = FG.generate_fingerprints(d, FP_length)
        mff_array.append(fingerprints)

    mff_fingerprints = np.array(mff_array)
    mff = reformat_feature_MFF(mff_fingerprints)
    mff = pd.DataFrame(mff)
    print(mff)

    X_encoded = mff
    # Identify columns where all values are the same
    cols_to_drop = X_encoded.columns[X_encoded.std() == 0]
    print(cols_to_drop)
    # Drop these columns
    X = X_encoded.drop(cols_to_drop, axis=1)    
    # Count the number of features after preprocessing
    num_features = X.shape[1]
    print(X)
    print(X.shape)
    

    # Define hyperparameter grid
    param_grid = {'n_estimators': list(range(10, 160, 20)),
                  'max_depth': [4,6,8,10],
                  'max_features': ['log2', 'sqrt', 0.25, 0.5]}

    # Instantiate and fit the model
    rnd_search = RandomizedSearchCV(RandomForestRegressor(random_state=seed), param_distributions=param_grid, n_iter=20, cv=kfold, random_state=seed, n_jobs=n_jobs, scoring='neg_mean_absolute_error')
    rnd_search.fit(X, y)
    best_params[target] = rnd_search.best_params_
    model = rnd_search.best_estimator_
    dump(model, f'all_models_MFF/model_MFF_{target}.joblib')

    #get the model metrics and perform 10-fold cross-validaiton and get the predicted values
    results = Parallel(n_jobs=n_jobs)(delayed(calculate_metrics)(model, X, y, train_index, test_index) for train_index, test_index in kfold.split(X))
    r2_values, mse_values, mae_values = zip(*results)
    r2_mean = np.mean(r2_values)
    r2_std = np.std(r2_values)
    mse_mean = np.mean(mse_values)
    mse_std = np.std(mse_values)
    mae_mean = np.mean(mae_values)
    mae_std = np.std(mae_values)
        
    with open('metrics_MFF.txt', 'a') as f:
        f.write('target\tr2\tstd r2\tmse\tstd mse\tmae\tstd maet\tnum_features\n')
    with open('metrics_MFF.txt', 'a') as f:
        f.write(f'{target}\t{r2_mean}\t{r2_std}\t{mse_mean}\t{mse_std}\t{mae_mean}\t{mae_std}\t{num_features}\n')
# Save the best_params and models to a file
dump(best_params, 'best_params_MFF.joblib')



