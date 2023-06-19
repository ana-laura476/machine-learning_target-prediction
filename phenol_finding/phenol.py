##finding phenol moiety present from a group of molecules
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

# Define your phenol SMILES string
phenol_smiles = 'c1ccccc1O'

# Convert the phenol SMILES string to a molecule object
phenol = Chem.MolFromSmiles(phenol_smiles)

# Read the CSV file
df = pd.read_csv('drug_bank.csv')  

# DataFrame to store rows that contain a phenol moiety
phenol_molecules = pd.DataFrame()

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

for index, row in df.iterrows():
    mol = Chem.MolFromSmiles(row['SMILES'])

    # Check if the molecule was correctly created
    if mol is None:
        print(f"Couldn't parse SMILES string: {row['SMILES']}")
        continue

    # Get atom indices of phenol moieties
    atom_indices = mol.GetSubstructMatches(phenol)

    # Check if oxygen in each phenol moiety is bonded to hydrogen
    for indices in atom_indices:
        oxygen_index = [i for i in indices if mol.GetAtomWithIdx(i).GetSymbol() == 'O'][0]
        oxygen_atom = mol.GetAtomWithIdx(oxygen_index)
        
        if oxygen_atom.GetTotalNumHs() == 1:
            phenol_molecules = phenol_molecules.append(row)
            break

print("Number of molecules containing a phenol moiety:", len(phenol_molecules))

# Save the results to a new CSV file
phenol_molecules.to_csv('phenol_molecules.csv', index=False)
