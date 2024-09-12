import os
import numpy as np
import pandas as pd
import pickle


current_dir = os.path.dirname(os.path.abspath(__file__))

# Read the CSV files into pandas DataFrames
GO_terms_all = pd.read_csv(os.path.join(current_dir,'..', 'dataset/GO_terms.csv'))
protein_names = pd.read_csv(os.path.join(current_dir,'..', 'dataset/protein_names.csv'))

GO_terms_list = pd.merge(GO_terms_all, protein_names, on='ID', how='inner')

GO_data_mat = GO_terms_list.pivot_table(index='ID', columns='GO', aggfunc=lambda x: 1, fill_value=0)
GO_data_mat.reset_index(inplace=True)
GO_data_mat.columns.name = None
GO_num = GO_data_mat.shape[1]-1


def dataset_creator(dataset_file, ave_ESM_file,
                    feature_file, label_file):     
    
    dataset = pd.read_csv(os.path.join(current_dir,'..', dataset_file)).to_numpy()
    
    with open(os.path.join(current_dir,'..', ave_ESM_file ), 'rb') as f:
        ESM2_embed = pickle.load(f) 
          
    
    data_feature = []
    data_label = []
    for pr_id, label in dataset:
        if pr_id in GO_data_mat['ID'].values:
            GO_array = np.array(GO_data_mat.loc[GO_data_mat['ID'] == pr_id])[:, 1:].reshape((GO_num,))
        else:
            GO_array = np.zeros((GO_num,))       
         
        # Concatenate pr_id, ESM2 embeddings, and GO array
        feature = [pr_id] + list(ESM2_embed[pr_id]) + list(GO_array)
        data_feature.append(feature)
        data_label.append( [pr_id] + [label])
    
    # Convert the list to a DataFrame
    train_feature = pd.DataFrame(data_feature)
    train_label = pd.DataFrame(data_label)
    
    # Now assign the columns
    train_feature.columns = ['ID'] + ['ESM2-' + str(i+1) for i in range(1280)] + list(GO_data_mat.columns[1:])
    train_label.columns = ['ID'] + ['label']
    
    # Set 'ID' as the index and drop the default index (1 to 4067 numbering)
    train_feature.set_index('ID', inplace=True)
    train_label.set_index('ID', inplace=True)
    
    # Optionally, remove index names
    train_feature.index.name = None
    train_label.index.name = None
    
    # Print shape for confirmation
    print("Shape of train_feature DataFrame:", train_feature.shape)
    print("Shape of train_label DataFrame:", train_label.shape)
    
    # Save X_train as a Pickle file
    train_feature.to_pickle(os.path.join(current_dir,'..', feature_file))
    train_label.to_pickle(os.path.join(current_dir,'..', label_file))

# ProTar-II
dataset_creator(dataset_file='dataset/ProTar-II.csv', ave_ESM_file='dataset/ave_ESM2.pkl',
                feature_file='feature/train_feature.pkl', label_file='feature/train_label.pkl')

# ProTar-II-Ind
dataset_creator(dataset_file='dataset/ProTar-II-Ind.csv', ave_ESM_file='dataset/ave_ESM2_Ind.pkl',
                feature_file='feature/test_feature.pkl', label_file='feature/test_label.pkl')
