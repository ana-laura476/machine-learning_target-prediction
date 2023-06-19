import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load spreadsheet
xl = pd.ExcelFile('all_models_metrics.xlsx')

# Load a sheet into a DataFrame by its name
df = xl.parse('Folha1')

# List of your models
models = ['svr', 'rf', 'knn', 'gp', 'cat', 'mlp', 'fnn', 'lstm']

# Corresponding model labels
model_labels = ['SVR', 'RF', 'k-NN', 'GP', 'CatBoost', 'MLP', 'FNN', 'LSTM']

# List of metrics
metrics = ['r2', 'mae', 'mse']
title = ['r2', 'MAE', 'MSE']
# Fontdict for title 'r2'
font_r2 = {'fontname':'DejaVu Sans', 'size':'16', 'style':'italic'}

# Create a figure with 3 subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# For each metric, we create a box plot
for i, metric in enumerate(metrics):
    # We'll gather data for each model
    data_for_metric = [df[f'{model}_{metric}'] for model in models]

    # Draw the boxplot on the corresponding subplot
    box_plot = sns.boxplot(data=data_for_metric, ax=axs[i], notch=True, linewidth=1.5)
    box_plot.set_xticklabels(model_labels, fontsize=14)  # Set custom model labels and increase font size

    # If the metric is 'r2', use the fontdict to make the title italic
    if metric == 'r2':
        axs[i].set_title('r$^2$', fontdict=font_r2)
    else:
        axs[i].set_title(metric.upper(), fontsize=16)  # Set title to just the metric and increase font size

plt.tight_layout()  # This ensures that labels are not cut off

# Save the plot with subplots to a file with a higher DPI for better quality
plt.savefig('all_metrics_boxplot.png', dpi=300)

# Close the plot to free memory
plt.close()

# Then we create individual box plots for each metric
for metric in metrics:
    plt.figure(figsize=(12, 8))  # Set the figure size

    # We'll gather data for each model
    data_for_metric = [df[f'{model}_{metric}'] for model in models]

    # Draw the boxplot
    box_plot = sns.boxplot(data=data_for_metric, notch=True, linewidth=1.5)
    box_plot.set_xticklabels(model_labels, fontsize=14)  # Set custom model labels and increase font size

    plt.title(metric.upper(), fontsize=16)  # Set title to just the metric and increase font size
    plt.tight_layout()  # This ensures that labels are not cut off

    # Save the plot to a file with a higher DPI for better quality
    plt.savefig(f'{metric}_boxplot.png', dpi=300)

    # Close the plot to free memory
    plt.close()
