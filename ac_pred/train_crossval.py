import pandas as pd
import torch
import time
from torch import nn
from pathlib import Path
from tqdm.auto import tqdm
from transformers import BertConfig
from utils.const_inputs import embedding_BERT
from utils.shallow_mlp_nn import MLP_NN, Dataset
from utils.compute_epoc_bs import select_bs
from utils.data_split import generate_splits
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

seeds = 42
metric_evaluation_interval = 1
target = '203'
n_folds = 10
device_sp = "cpu"

# load the pretrained BERT model to generate the embeddings
model_config = BertConfig(
        vocab_size=105,  # we align this to the tokenizer vocab_size
        max_position_embeddings=102,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=8,
        num_hidden_layers=12,
        type_vocab_size=1)
model_config.output_hidden_states = True
bert_model_path = Path.cwd().joinpath("pretrained_model/50_256_12_8.pt")
raw_data = pd.read_csv(f'data/target{target}_data.csv')


embedder = embedding_BERT(raw_data, pretrained_model_path=bert_model_path, config=model_config, device="cpu")
embedder.extract()

X_data = pd.DataFrame(embedder.cls_store_arr)
Y_data = raw_data['activity']

loss_mse_func = nn.MSELoss()

# initializing lists to store all the train/val/test metric data
split_list = []
best_epoch_list = []
best_val_r2_list = []
best_val_mae_list = []
best_val_mse_list = []
best_test_r2_list = []
best_test_mae_list = []
best_test_mse_list = []

val_r2_list = []
val_mae_list = []
val_mse_list = []
test_r2_list = []
test_mae_list = []
test_mse_list = []
train_loss_list = []
epoch_list = []

# selecting batch size and epochs depending on the size of the dataset.
train_batch_size, epochs = select_bs(raw_data)

start = time.time()
for split in range(n_folds):

    curr_split = split + 1
    # initialize the neural network model to be trained
    model_obj = MLP_NN()
    for p in model_obj.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model_obj.to(torch.device(device_sp))
    optim = torch.optim.Adam(model_obj.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # splitting the data and preprocessing the tensors
    train_index, val_index, test_index = generate_splits(X_data, n_folds, curr_split, seeds)
    X_train, X_val, X_test = X_data.iloc[train_index], X_data.iloc[val_index], X_data.iloc[test_index]
    Y_train, Y_val, Y_test = Y_data.iloc[train_index], Y_data.iloc[val_index], Y_data.iloc[test_index]

    X_train_ten = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    X_val_ten = torch.tensor(X_val.to_numpy(), dtype=torch.float32).to(torch.device(device_sp))
    X_test_ten = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(torch.device(device_sp))

    Y_train_arr = Y_train.to_numpy().reshape(Y_train.shape[0], 1)
    Y_val_arr = Y_val.to_numpy().reshape(Y_val.shape[0], 1)
    Y_test_arr = Y_test.to_numpy().reshape(Y_test.shape[0], 1)

    Y_train_ten = torch.tensor(Y_train_arr, dtype=torch.float32)
    Y_val_ten = torch.tensor(Y_val_arr, dtype=torch.float32).to(torch.device(device_sp))
    Y_test_ten = torch.tensor(Y_test_arr, dtype=torch.float32).to(torch.device(device_sp))

    encodings = {'X_train': X_train_ten, 'Y_train': Y_train_ten}
    dataset = Dataset(encodings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):

        total_train_loss = 0
        iter_counter = 0

        model_obj.train()
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=False)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            X_train_batch = batch['X_train'].to(torch.device(device_sp))
            Y_train_batch = batch['Y_train'].to(torch.device(device_sp))

            outputs = model_obj(X_train_batch)

            optim.zero_grad()
            train_loss = loss_mse_func(Y_train_batch, outputs)
            total_train_loss += train_loss
            iter_counter += 1
            train_loss.backward()
            optim.step()

        # validation after each epoch on the val split
        if (epoch + 1) % metric_evaluation_interval == 0:

            model_obj.eval()
            valid_preds = model_obj(X_val_ten)
            valid_preds = valid_preds.detach().numpy()
            val_mae = mean_absolute_error(Y_val_ten, valid_preds)
            val_r2 = r2_score(Y_val_ten, valid_preds)
            val_mse = mean_squared_error(Y_val_ten, valid_preds)

            avg_train_loss = (total_train_loss.detach().numpy()) / iter_counter
            train_loss_list.append(avg_train_loss)
            val_r2_list.append(val_r2)
            val_mae_list.append(val_mae)
            val_mse_list.append(val_mse)
            epoch_list.append(epoch + 1)

            if epoch == 0:
                best_val_r2 = val_r2
                best_val_mae = val_mae
                best_val_mse = val_mse

                model_obj.eval()
                test_preds = model_obj(X_test_ten)
                test_preds = test_preds.detach().numpy()
                test_mae = mean_absolute_error(Y_test_ten, test_preds)
                test_r2 = r2_score(Y_test_ten, test_preds)
                test_mse = mean_squared_error(Y_test_ten, test_preds)
                best_epoch = epoch + 1

            # evaluating test r2 as and when validation r2 improves
            else:
                if val_r2 > best_val_r2:

                    best_val_r2 = val_r2
                    best_val_mae = val_mae
                    best_val_mse = val_mse

                    model_obj.eval()
                    test_preds = model_obj(X_test_ten)
                    test_preds = test_preds.detach().numpy()
                    test_mae = mean_absolute_error(Y_test_ten, test_preds)
                    test_r2 = r2_score(Y_test_ten, test_preds)
                    test_mse = mean_squared_error(Y_test_ten, test_preds)
                    best_epoch = epoch+1

            summary_df = pd.DataFrame({'epoch': epoch_list, 'train_loss': train_loss_list, 'val_r2': val_r2_list,
                                       'val_mae': val_mae_list, 'val_mse': val_mse_list})

            if (epoch + 1) % 40 == 0:
                print(f', epoch={epoch + 1}, train_loss=: {total_train_loss / iter_counter}, val_mse=: {val_mse:.3f}'
                      f', val_mae=: {val_mae:.3f}, val_r2=: {val_r2:.3f}, time={time.time() - start}')


    split_list.append(curr_split)
    best_epoch_list.append(best_epoch)
    best_val_r2_list.append(best_val_r2)
    best_val_mae_list.append(best_val_mae)
    best_val_mse_list.append(best_val_mse)
    best_test_r2_list.append(test_r2)
    best_test_mae_list.append(test_mae)
    best_test_mse_list.append(test_mse)

    best_summary_df = pd.DataFrame({'split': split_list, 'epoch': best_epoch_list, 'val_r2': best_val_r2_list,
                                    'val_mae': best_val_mae_list, 'val_mse': best_val_mse_list,
                                    'test_r2': best_test_r2_list, 'test_mae': best_test_mae_list,
                                    'test_mse': best_test_mse_list})

    best_summary_df.to_csv(f'short_summary_{target}.csv', index=False)
    summary_df.to_csv(f'all_summary_{target}.csv', index=False)
print('ads')
