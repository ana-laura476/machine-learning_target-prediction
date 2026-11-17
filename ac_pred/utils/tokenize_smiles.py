import numpy as np
import torch
from rdkit import Chem
from utils.tokenizer_class import SmilesTokenzier


class tokenizer_class:

    def __init__(self, smiles_df):
        self.smiles_tok = SmilesTokenzier(max_length=100)
        self.dataset = smiles_df

    def tokenize_smi(self):
        cannonical_smiles = []
        array = np.zeros((self.dataset.shape[0], 100))
        for i in range(self.dataset.shape[0]):
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.dataset['smiles'][i]), canonical=True)
            cannonical_smiles.append(smiles)
            array[i, :] = self.smiles_tok.encode(smiles)[1]

        self.dataset['can_SMILES'] = cannonical_smiles
        return torch.LongTensor(array)

    def mask_input(self, smi_tok_tensor, mask=0.001):
        labels = smi_tok_tensor
        input_ids = labels.detach().clone()

        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < mask) * (input_ids != 0) * (input_ids != 3) * (input_ids != 4)

        # loop through each row in input_ids tensor (cannot do in parallel)
        for i in range(input_ids.shape[0]):
            # get indices of mask positions from mask array
            selection = torch.flatten(mask_arr[i].nonzero()).tolist()
            # mask input_ids
            input_ids[i, selection] = 5  # our custom [MASK] token == 5

        return input_ids, labels


