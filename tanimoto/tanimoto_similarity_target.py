#in this example, query molecule YM086 and target oatpb1b3 was used

import pandas as pd
import numpy as np 
import csv
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

#import csv with the smiles of the query molecule
YM086 = pd.read_csv('YM086.csv', delimiter=',', header=None)
YM086 = YM086.rename(columns={0: 'Smiles'})
print(YM086)
df = pd.read_csv('tanimoto_input.csv', delimiter=',')
df['target'] = df['target'].str.extract('(\d+)').astype(int)
print(df)
#filter the targets to get just the target of interest
oatpb1b3 = df[df['target'] == 1743121]
print(oatpb1b3)
print(len(oatpb1b3))
oatpb1b3 = oatpb1b3.copy()

#smiles of the target of interest molecules
smiles = oatpb1b3['Smiles']
print(smiles)

def calculate_morgan_fingerprint(smiles, radius=2, nBits=2024):
    mol = Chem.MolFromSmiles(smiles)
    morgan_fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return morgan_fingerprint
#append to the target of interest the morgan fingerprints of its molecules
oatpb1b3['MorganFingerprint'] = oatpb1b3['Smiles'].apply(calculate_morgan_fingerprint)

print(oatpb1b3)
print(len(oatpb1b3))

YM086_smiles = YM086.loc[0, 'Smiles']
print(YM086_smiles)
YM086_morgan_fingerprint = calculate_morgan_fingerprint(YM086_smiles)

#append to the query molecule its morgan fingerprints 
YM086['MorganFingerprint'] = [YM086_morgan_fingerprint]

# Print the updated query molecule DataFrame
print(YM086)


# Function to calculate the Tanimoto similarity between two fingerprints
def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# Calculate the Tanimoto similarity between the query molecule and every molecule in the target of interest database
oatpb1b3['TanimotoSimilarity'] = oatpb1b3['MorganFingerprint'].apply(lambda x: tanimoto_similarity(x, YM086_morgan_fingerprint))

# Print the updated target of interest DataFrame with the calculated Tanimoto similarities
print(oatpb1b3)

oatpb1b3.to_csv('oatpb1b3_with_tanimoto_similarities.csv', index=False)
