import random
import numpy as np
import pandas as pd
from scipy.spatial import distance
import torch
from sklearn.preprocessing import LabelEncoder


def load_data(data: pd.DataFrame, phenotypic_features=['SITE_ID', 'SEX'], dataset='ABIDE1', method='partial', atlas='CC200'):
    """
    ## Args:
        - data : (pd.DataFrame) DataFrame containing subject IDs and labels
        - phenotypic_features : (list) List of phenotypic feature names to extract from DataFrame
        - dataset : (str) Name of the dataset ('ABIDE1', 'ABIDE2', 'ADHD', 'ADNI', 'MDD')
        - method : (str) Correlation method used ('pearson' or 'partial')
        - atlas : (str) Atlas used for ROI definition ('AAL', 'CC200', etc.)
        - self_loop : (bool) Whether to include self-loops in the graph
    ## Returns:
        - corr_mat_flat : (list) Numpy array of flattend correlation matrices for each subject (shape: (num_subjects, flattened_corr_mat_dim))
        - phenotypic_dict : (dict) Dictionary of phenotypic features
        - labels : (list) List of labels for each subject
    """    
    df = data.copy()
    corr_mat_list = []
    labels = []
    if dataset == 'ADNI':
        for ID, PHASE, DX in df[['ID', 'PHASE', 'DX']].values:
            corr_mat = np.load(f'/nasdata4/jaemin/data/ADNI_new/{PHASE}/{ID}/{ID}_{atlas}_{method}_corr_mat.npy')
            corr_mat_list.append(corr_mat)
            labels.append(DX)

    else:
        for ID, SITE, DX in df[['ID', 'SITE_ID', 'DX']].values:
            if dataset == 'ABIDE1':
                # corr_mat = np.load(f'/nasdata4/jaemin/data/ABIDE/ABIDE_prep/{ID}/func_preproc/DPARSF/{ID}_HO_pearson_corr_mat.npy')
                corr_mat = np.load(f'/nasdata4/jaemin/data/ABIDE_new/1/{SITE}/{ID}/{ID}_{atlas}_{method}_corr_mat.npy')   # ABIDE_new with DPARSF

            elif dataset == 'ABIDE2':
                corr_mat = np.load(f'/nasdata4/jaemin/data/ABIDE_new/2/{SITE}/{ID}/{ID}_{atlas}_{method}_corr_mat.npy')
                
            elif dataset == 'ADHD':
                if len(str(ID)) == 5:
                    ID = '00' + str(ID)
                corr_mat = np.load(f'/nasdata3/jaemin/ADHD/{SITE}/{ID}/{ID}_{atlas}_{method}_corr_mat.npy')
            
            elif dataset == 'MDD':
                corr_mat = np.load(f'/nasdata3/jaemin/MDD/{ID}/{ID}_{atlas}_{method}_corr_mat.npy')
                
            corr_mat_list.append(corr_mat)
            labels.append(DX)
        
    R = corr_mat_list[0].shape[0]   # number of ROIs
    triu_idx = np.triu_indices(R, k=1)
    corr_mat_flat = np.array([cmat[triu_idx] for cmat in corr_mat_list])  # shape: (num_subjects, flattened corr_mat dim)
    
    # Check if each phenoypic feature exists in dataframe columns
    for feature in phenotypic_features:
        if feature not in df.columns:
            raise ValueError(f"Phenotypic feature '{feature}' not found in DataFrame columns.")
    
    if 'SITE_ID' in phenotypic_features:
        enc = LabelEncoder()
        df['SITE_ID'] = enc.fit_transform(df['SITE_ID'])
    
    phenotypic_dict = {}
    for feature in phenotypic_features:
        phenotypic_dict[feature] = df[feature].values

    return corr_mat_flat, phenotypic_dict, labels

def get_global_edge_inputs(embeddings: torch.Tensor, phenotypic_dict: dict, edge_thr: float):
    """
    Construct population graph based on imaging and non-imaging similarities.
    ## Args:
        - embeddings : (torch.Tensor) Node embeddings of shape (num_subjects, embedding_dim)
        - phenotypic_dict : (dict) Dictionary of non-image features
        - edge_thr : (float) Threshold for edge creation based on combined similarity
    ## Returns:
        - edge_index : (np.ndarray) Edge index array of shape (2, num_edges)
        - edgenet_input : (np.ndarray) Edge features array of shape (num_edges, 2 * num_phenotypic_features)
    """
    embeddings_np = np.array(embeddings.detach().cpu().numpy())
    n = embeddings_np.shape[0]    # number of subjects
    num_edge = n*(1+n)//2 - n  
    
    nonimg = np.array([v for v in phenotypic_dict.values()]).T
    num_phenotypic_features = len(phenotypic_dict.keys())
    
    edge_index = np.zeros([2, num_edge], dtype=np.int64) 
    edgenet_input = np.zeros([num_edge, 2*num_phenotypic_features], dtype=np.float32)  
    C = np.zeros(num_edge, dtype=np.float32)
        
    imaging_sim = get_imaging_similarity_matrix(embeddings_np)  # S_img
    phenotypic_sim = get_phenotypic_similarity_matrix(phenotypic_dict)  # S_ph
    
    # C_prime = phenotypic_sim
    C_prime = imaging_sim * phenotypic_sim
    
    flatten_ind = 0 
    for i in range(n):
        for j in range(i+1, n):
            edge_index[:, flatten_ind] = [i, j]
            edgenet_input[flatten_ind] = np.concatenate((nonimg[i], nonimg[j]))
            C[flatten_ind] = C_prime[i][j]  
            flatten_ind +=1
    
    keep_ind = np.where(C > edge_thr)[0]  
    edge_index = edge_index[:, keep_ind]
    edgenet_input = edgenet_input[keep_ind]
    
    return edge_index, edgenet_input

def get_imaging_similarity_matrix(embeddings: np.ndarray):
    """
    Caculate imaging similarity matrix
    ## Args.
        - embeddings : (numpy.ndarray) Node embeddings of shape (num_subjects, embedding_dim)
    ## Returns.
        - sim_mat : (numpy.ndarray) Imaging similarity matrix of shape (num_subjects, num_subjects)
    """
    dist = distance.cdist(embeddings, embeddings, metric='correlation')
    sigma = np.mean(dist)
    sim_mat = np.exp(- dist ** 2 / (2 * sigma ** 2))
    return sim_mat

def get_phenotypic_similarity_matrix(phenotypic_dict: dict):
    """
    Calculate phenotypic similarity matrix
    ## Args.
        - phenotypic_dict : (dict) Dictionary containing non-image features for each subject (ex. {'AGE': [value_1, value_2, ...]} )
    ## Returns.
        - sim_mat : (numpy.ndarray) Phenotypic similarity matrix of shape (num_subjects, num_subjects)
    """
    num_nodes = len(next(iter(phenotypic_dict.values())))   # length of first value
    sim_mat = np.zeros((num_nodes, num_nodes))

    for feat_name in phenotypic_dict.keys():
        feature_list = phenotypic_dict[feat_name]

        if feat_name in ['AGE', 'FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    val = abs(float(feature_list[k]) - float(feature_list[j]))
                    if val < 2:
                        sim_mat[k, j] += 1
                        sim_mat[j, k] += 1

        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if feature_list[k] == feature_list[j]:
                        sim_mat[k, j] += 1
                        sim_mat[j, k] += 1

    return sim_mat

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def balanced_binary_sampling(df: pd.DataFrame, seed: int):
    """
    ## Args:
    - df : (pandas.DataFrame) Must contain columns 'DX' (values in {0,1}) and 'SEX' (values in {0,1}).
    - seed : (int) Random seed for reproducibility.

    ## Returns:
    - df_balanced : (pandas.DataFrame)  Subset with balanced sampling applied.
    """
    # Check validity of input DataFrame
    required_cols = ['DX', 'SEX']
    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")
    
    df['DX'] = df['DX'].astype(int)
    df['SEX'] = df['SEX'].astype(int)

    allowed_values = {0, 1}
    unique_dx = set(df['DX'].unique())
    unique_sex = set(df['SEX'].unique())

    if not unique_dx.issubset(allowed_values):
        raise ValueError(f"Column 'DX' must only contain values {{0, 1}}. Found: {unique_dx}")
    
    if not unique_sex.issubset(allowed_values):
        raise ValueError(f"Column 'SEX' must only contain values {{0, 1}}. Found: {unique_sex}")


    # Create a copy to work on
    df_bal = df.copy()

    count_matrix = {
        0: {0: 0, 1: 0},
        1: {0: 0, 1: 0}
    }

    for dx_val in [0, 1]:
        for sex_val in [0, 1]:
            c = ((df_bal['DX'] == dx_val) & (df_bal['SEX'] == sex_val)).sum()
            count_matrix[dx_val][sex_val] = int(c)

    target_counts_by_sex = {
        0: min(count_matrix[0][0], count_matrix[1][0]),
        1: min(count_matrix[0][1], count_matrix[1][1]) 
    }

    rng = np.random.default_rng(seed)
    sampled_parts = []

    for dx_val in [0, 1]:
        for sex_val in [0, 1]:
            target_n = target_counts_by_sex[sex_val]
            
            sel = df_bal[(df_bal['DX'] == dx_val) & (df_bal['SEX'] == sex_val)]
            
            if len(sel) <= target_n:
                sampled_parts.append(sel)
            else:
                chosen_idx = rng.choice(sel.index.values, size=target_n, replace=False)
                sampled_parts.append(df_bal.loc[chosen_idx])

    final_df = pd.concat(sampled_parts, ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return final_df
