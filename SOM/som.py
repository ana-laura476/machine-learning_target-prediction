from som import SOM
import pandas as pd
import numpy as np
import os 
import csv
import numpy as np
import joblib
import subprocess
import pickle


seed=42
#load the molecules and input data
subprocess.run(['python', 'cats.py', 'mols.csv', 'mols2.csv', 'output.txt', 'output2.txt'])
#load input data
df = pd.read_csv('input_data.csv', delimiter=",")
X = df.drop(['target', 'molecule'], axis=1)
print(X)
print(X.shape)
som = SOM(70, 70)  # initialize a 70 by 70 SOM
som.fit(X, 20000, save_e=True, interval=100)  # fit the SOM for 20000 epochs, save the error every 100 steps
som.plot_error_history(filename='images/som_error.png')  # plot the training error history
som.plot_distance_map(filename='images/distance_map.png')
som.plot_density_map(data, filename='images/density_map.png')

with open('model_pkl', 'wb') as files:
    pickle.dump(som, files)
result=som.winner_neurons(X)
data_neurons= result[1].ravel()

# Create neuron map dataframe
nueron_map = pd.DataFrame({'identity': molecule, 'neuron': data_neurons, 'target': target})
neuron_map = pd.concat([cluster_map, X], axis=1)

# Save neuron map to file
neuron_map.to_csv("neuron_map.csv", index=False)

#load the new molecules to predict


#load the new molecules to predict
with open('output.txt') as f:
    for i, line in enumerate(f):
        molecule_descriptor = [float(x) for x in line.split()]
        molecule_descriptor = np.array(molecule_descriptor).reshape(1, -1)
        molecule_neuron = som.winner(molecule_descriptor)
        #get distance to new molecule and backgroud noise with dark chemical matter (dcm)

        nueron_rows = neuron_map[nueron_map['neuron'] == molecule_neuron[1]]
        for index, row in nueron_rows.iterrows():
            neuron_descriptor = row.drop(['identity', 'cluster', 'target'], axis=1)
            distance = dist.euclidean(molecule_descriptor[0], neuron_descriptor)

            #get backgroud noise with dark chemical matter (dcm)
            with open('output2.txt') as f:
                for i, line in enumerate(f):
                    dcm_descriptor = [float(x) for x in line.split()]
                    dcm_descriptor = np.array(dcm_descriptor).reshape(1, -1)
                    distance_to_cluster_dcm = []
                    distance_dcm = dist.euclidean(dcm_descriptor[0], neuron_descriptor)

                    distance_to_cluster_dcm.append(distance_dcm)
                    mean_distance_dcm=np.mean(distance_to_cluster_dcm)          
                    std_distance_dcm=np.std(distance_to_cluster_dcm)
                    #get p-vale for each molecule of the cluster
                    z_score = (float(distance) - float(mean_distance_dcm)/float(std_distance_dcm))
                    print(z_score)
                    p_value = scipy.stats.norm.sf(z_score)
                    print(p_value)

                    molecules = f'molecules{i}'
                    if molecules not in molecules_dfs:
                        molecules_dfs[molecules] = pd.DataFrame(columns=['neuron', 'molecule_within_cluster', 'target', 'distance', 'dcm_mean_distance', 'dcm_std_distance', 'z-score', 'p-value'])
                    molecules_dfs[molecules] = molecules_dfs.append({'neuron': molecule_neuron,'molecules_within_neuron': nueron_rows['molecule'], 'target': nueron_rows['target'], 'distance to cluster': distance, 'dcm mean distance': mean_distance_dcm, 'dcm std distance': std_distance_dcm, 'z-score': z_score, 'p-value': p_value}, ignore_index=True)
         
          
for molecules in molecule_dfs:
    molecules_dfs[molecules].to_csv(f'{molecules}_clusters.csv', index=False)
