import numpy as np
from sklearn.model_selection import KFold


def generate_splits(X_data, n_splits, curr_split, seeds):

    if n_splits == 5:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seeds)
        fold_indices = list(kfold.split(X_data))

        split_1 = fold_indices[0][1]
        split_2 = fold_indices[1][1]
        split_3 = fold_indices[2][1]
        split_4 = fold_indices[3][1]
        split_5 = fold_indices[4][1]

        if curr_split == 1:
            train_index = np.concatenate((split_1, split_2, split_3))
            val_index = split_4
            test_index = split_5

        if curr_split == 2:
            train_index = np.concatenate((split_2, split_3, split_4))
            val_index = split_5
            test_index = split_1

        if curr_split == 3:
            train_index = np.concatenate((split_3, split_4, split_5))
            val_index = split_1
            test_index = split_2

        if curr_split == 4:
            train_index = np.concatenate((split_4, split_5, split_1))
            val_index = split_2
            test_index = split_3

        if curr_split == 5:
            train_index = np.concatenate((split_5, split_1, split_2))
            val_index = split_3
            test_index = split_4

    if n_splits == 10:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seeds)
        fold_indices = list(kfold.split(X_data))

        split_1 = fold_indices[0][1]
        split_2 = fold_indices[1][1]
        split_3 = fold_indices[2][1]
        split_4 = fold_indices[3][1]
        split_5 = fold_indices[4][1]
        split_6 = fold_indices[5][1]
        split_7 = fold_indices[6][1]
        split_8 = fold_indices[7][1]
        split_9 = fold_indices[8][1]
        split_10 = fold_indices[9][1]

        if curr_split == 1:
            train_index = np.concatenate((split_1, split_2, split_3, split_4, split_5, split_6, split_7, split_8))
            val_index = split_9
            test_index = split_10

        if curr_split == 2:
            train_index = np.concatenate((split_2, split_3, split_4, split_5, split_6, split_7, split_8, split_9))
            val_index = split_10
            test_index = split_1

        if curr_split == 3:
            train_index = np.concatenate((split_3, split_4, split_5, split_6, split_7, split_8, split_9, split_10))
            val_index = split_1
            test_index = split_2

        if curr_split == 4:
            train_index = np.concatenate((split_4, split_5, split_6, split_7, split_8, split_9, split_10, split_1))
            val_index = split_2
            test_index = split_3

        if curr_split == 5:
            train_index = np.concatenate((split_5, split_6, split_7, split_8, split_9, split_10, split_1, split_2))
            val_index = split_3
            test_index = split_4

        if curr_split == 6:
            train_index = np.concatenate((split_6, split_7, split_8, split_9, split_10, split_1, split_2, split_3))
            val_index = split_4
            test_index = split_5

        if curr_split == 7:
            train_index = np.concatenate((split_7, split_8, split_9, split_10, split_1, split_2, split_3, split_4))
            val_index = split_5
            test_index = split_6

        if curr_split == 8:
            train_index = np.concatenate((split_8, split_9, split_10, split_1, split_2, split_3, split_4, split_5))
            val_index = split_6
            test_index = split_7

        if curr_split == 9:
            train_index = np.concatenate((split_9, split_10, split_1, split_2, split_3, split_4, split_5, split_6))
            val_index = split_7
            test_index = split_8

        if curr_split == 10:
            train_index = np.concatenate((split_10, split_1, split_2, split_3, split_4, split_5, split_6, split_7))
            val_index = split_8
            test_index = split_9

    return train_index, val_index, test_index



# seeds = 42
# raw_data = pd.read_csv('/mnt/share/user/employee/rdebanja/AC_Pred/1_try/seed_42/target5293/data.csv')
# X_data = generate_mqns(raw_data)
# Y_data = raw_data['activity']
# n_splits = 10
# curr_split = 8
#
# tr, val, test = generate_splits(X_data, n_splits, curr_split, seeds)
#
# print('asd')