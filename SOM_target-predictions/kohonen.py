from som import SOM
import pandas as pd
import numpy as np
import os 
import csv
import numpy as np
import joblib
import subprocess
import scipy
import math
from scipy.spatial import distance

if __name__ == '__main__':
    seed=42
    #load the molecules and input data
    subprocess.run(['python', 'cats.py', 'mols.csv', 'mols2.csv', 'output.txt', 'output2.txt'])
    #load input data
    df = pd.read_csv('input_data.csv', delimiter=",")

    molecule=df['molecule']
    print(molecule.shape)
    target=df['target']
    print(target.shape)
    X = df.drop(['target', 'molecule', 'activity'], axis=1).values
    print(X)
    print(X.shape)
    l = int(np.sqrt(len(molecule)))
    print(f'the SOM has a {l} by {l}')
    som = SOM(l, l)  # initialize a 70 by 70 SOM
    som.fit(X, 10000, save_e=True, interval=100)  # fit the SOM for 20000 epochs, save the error every 100 steps
    som.plot_error_history(filename='images/som_error.png')  # plot the training error history
    result=som.winner_neurons(X)
    print(result)
    print(result.shape)
    result_str = [f"{x},{y}" for x, y in result]
    result_str = np.array(result_str).reshape(-1, 1)
    result_str = result_str.reshape(-1,)
    print(result_str)
    print(result_str.shape)
    # Create neuron map dataframe
    neuron_map = pd.DataFrame({'identity': molecule, 'neuron': result_str, 'target': target})
    X=pd.DataFrame(X)
    neuron_map = pd.concat([neuron_map, X], axis=1)
    # Save neuron map to file
    neuron_map.to_csv("neuron_map.csv", index=False)
    #load the new molecules to predict
    molecules_dfs = {}
    #load the new molecules to predict
    
    if __name__ == '__main__':
        neuron_rows_true = pd.DataFrame()
        with open('output.txt') as f:
            for i, line in enumerate(f):
                neuron_map_molecule=neuron_map.copy()
                molecule_descriptor = [float(x) for x in line.split()]
                molecule_descriptor = np.array(molecule_descriptor).reshape(1, -1)
                molecule_neuron = np.array([som.winner(molecule_descriptor)])
                molecule_neuron_str = [f"{x},{y}" for x, y in molecule_neuron]
                molecule_neuron_str = np.array(molecule_neuron_str).reshape(-1, 1)
                molecule_neuron_str = molecule_neuron_str.reshape(-1,)
                print(f'the winner neuron is: {molecule_neuron_str} for molecule {i}')
                print(molecule_neuron_str.shape)
                #get distance to new molecule and backgroud noise with dark chemical matter (dcm)
                
                neuron_rows_true[f'neuron{i}']=neuron_map_molecule.neuron.apply(lambda x: x == molecule_neuron_str)
                print(neuron_rows_true)
                print(neuron_rows_true.shape)
                neuron_map_molecule = pd.concat([neuron_map_molecule, neuron_rows_true[f'neuron{i}']], axis=1)
                print(neuron_map_molecule)
                neuron_rows = pd.DataFrame(neuron_map_molecule[neuron_map_molecule[f'neuron{i}'] == True])
                print(neuron_rows)
                print(neuron_rows.shape)
                for index, row in neuron_rows.iterrows():
                    print(row)
                    iden=row['identity']
                    print(iden)
                    neu=row['neuron']
                    print(neu)
                    tar=row['target']
                    print(tar)
                    neuron_descriptor = row.drop(['identity', 'neuron', 'target', f'neuron{i}'], axis=0)
                    print(neuron_descriptor)
                    print(neuron_descriptor.shape)
                    distances = distance.euclidean(molecule_descriptor[0], neuron_descriptor)
                    #get backgroud noise with dark chemical matter (dcm)
                    distance_to_cluster_dcm = []
                    with open('output2.txt') as g:
                        for k, line in enumerate(g):
                            dcm_descriptor = [float(x) for x in line.split()]
                            dcm_descriptor = np.array(dcm_descriptor).reshape(1, -1)
                            distance_dcm = distance.euclidean(dcm_descriptor[0], neuron_descriptor)
                            distance_to_cluster_dcm.append(distance_dcm)
                    mean_distance_dcm=np.mean(distance_to_cluster_dcm)          
                    std_distance_dcm=np.std(distance_to_cluster_dcm)
                    #get p-vale for each molecule of the cluster
                    z_score = abs(float(distances) - float(mean_distance_dcm)/float(std_distance_dcm))
                    print(z_score)
                    p_value = scipy.stats.norm.sf(z_score)
                    print(p_value)
                    molecules = f'molecules{i}'
                    if molecules not in molecules_dfs:
                        molecules_dfs[molecules] = pd.DataFrame(columns=['neuron', 'molecule_within_neuron', 'target', 'distance', 'dcm_mean_distance', 'dcm_std_distance', 'z-score', 'p-value'])
                    molecules_dfs[molecules] = molecules_dfs[molecules].append({'neuron': neu,'molecule_within_neuron': iden, 'target': tar, 'distance': distances, 'dcm_mean_distance': mean_distance_dcm, 'dcm_std_distance': std_distance_dcm, 'z-score': z_score, 'p-value': p_value}, ignore_index=True)
                         
                          
                for molecules in molecules_dfs:
                    molecules_dfs[molecules].to_csv(f'{molecules}_clusters.csv', index=False)
