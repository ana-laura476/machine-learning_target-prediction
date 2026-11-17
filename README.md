# random-forests_target-prediction

* input_data_with_smiles.csv contains the full dataset with all targets, annotated pactivity values, molecules in CATS2 descriptors, and smiles (for supervised learning).
* input_data.csv contains the full dataset with all targets, and molecules in CATS2 descriptors (for unsupervised learning).
* data_base_caracterization.csv has all information regarding the targets and pactivity ranges (supervised learning)
* each folder contains the scripts used to assess predictive performance, for each algorithm
* ac_pred contains the pretrained BERT model and specific environment used
* for chemprop usage:
*   Without rdkit features: chemprop_train --data_path /dir/data.csv --dataset_type regression --split_type cv --num_folds 10 --save_dir dir/target1075094_checkpoints --extra_metrics r2 mae mse
*   With rdkit features: chemprop_train --data_path /dir/data.csv --dataset_type regression --features_generator rdkit_2d_normalized --no_features_scaling --split_type cv --num_folds 10 --save_dir dir/target1075094_checkpoints --extra_metrics r2 mae mse
