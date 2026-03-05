import argparse
import time
import datetime
import numpy as np
import pandas as pd

import torch
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold

from models import IPGNN
from utils import load_data, set_seed, balanced_binary_sampling


def arg_parse():
    parser = argparse.ArgumentParser()
    
    # Training
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--phenotypic_features', type=str, nargs='+', default=['SITE_ID', 'SEX'], choices=['SITE_ID', 'SEX', 'AGE'], help='Phenotypic features to use')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--weight_decay', default=5e-5, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=400, type=int, help='Epochs for training')
    parser.add_argument('--result_path', type=str, help='Path to save trained model checkpoints and results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='ABIDE1', choices=['ABIDE1', 'ABIDE2', 'ADHD'])
    
    # Individual FC graph construction
    parser.add_argument('--method', type=str, default='pearson', choices=['pearson', 'partial'], help='Method for individual graph construction')
    parser.add_argument('--atlas', type=str, default='AAL', choices=['AAL', 'CC200'], help='Brain atlas for rs-fMRI parcellation')
    
    # Model
    parser.add_argument('--hidden_dim_gnn', type=int, default=16, help='Hidden dimension of GNN layers')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout ratio for Global-GNN')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--global_edge_thr', type=float, default=1.1, help='Threshold for global edge construction')
    
    # Edge MLP
    parser.add_argument('--hidden_dim_edge', type=int, default=128, help='Hidden dimension of edge MLP')
    parser.add_argument('--edge_dropout', type=float, default=0.2, help='Dropout ratio for edge MLP')
    
    # For ablation study
    parser.add_argument('--test_batch', type=int, default=1, help='Batch size for inductive inference')
    return parser.parse_args()

def train(model, pheno_dict: dict, x: torch.Tensor, y: torch.Tensor, train_ind, val_ind, opt, model_path):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    
    best_acc = 0
    PATIENCE_LIMIT = 100
    patience_count = 0
    for epoch in range(opt.epochs):
        model.train()
        optimizer.zero_grad()
        
        node_logits, num_edges = model(x, pheno_dict)
        if epoch == 0:
            print(f'# Number of edges in the population graph: {num_edges}')
        
        loss = loss_fn(node_logits[train_ind], y[train_ind])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            node_logits, _ = model(x, pheno_dict)
        loss_val = loss_fn(node_logits[val_ind], y[val_ind])
        
        acc_val = (node_logits[val_ind].argmax(dim=-1) == y[val_ind]).float().mean().item()
        auc_val = roc_auc_score(y[val_ind].detach().cpu(), node_logits[val_ind].softmax(-1)[:, 1].detach().cpu())
        
        print(f"[ Epoch {epoch+1:3d} ]  Train Loss: {loss.item():.5f},  Val Loss: {loss_val.item():.5f},  Val Acc: {acc_val:.5f},  Val AUC: {auc_val:.5f}")

        if acc_val > best_acc:
            best_acc = acc_val
            patience_count = 0
            
            torch.save(model.state_dict(), model_path)
            print("# Saved best model!")
        else:
            patience_count += 1
            if patience_count >= PATIENCE_LIMIT:
                print(f"# Early stopping at epoch {epoch+1}")
                break
    
def inductive_inference(model, pheno_dict: dict, x: torch.Tensor, y: torch.Tensor, test_ind):    
    model.eval()
    with torch.no_grad():
        node_logits, num_edges = model(x, pheno_dict)
    print(f"# Number of edges in the poulation graph: {num_edges}")
    
    return node_logits[test_ind].softmax(-1)[:, 1].detach().cpu(), y[test_ind].detach().cpu()

def main():
    args = arg_parse()
    print("==========       CONFIG      =============")
    for arg, content in args.__dict__.items():
        print(f"{arg}: {content}")
    print("===========================================\n")
    device = torch.device(f'cuda:{args.gpu}')
    set_seed(args.seed)
    torch.set_num_threads(4)
    
    NUM_FOLDS = 5
    NUM_ROI_DICT = {'AAL': 116, 'CC200': 200}
    IN_DIM = NUM_ROI_DICT[args.atlas]
    
    # Load csv for demographic information
    if args.dataset == 'ABIDE1':
        df = pd.read_csv('/nasdata4/jaemin/data/ABIDE_new/csv/df_ABIDE1_876.csv')
    elif args.dataset == 'ABIDE2':
        # df = pd.read_csv('/nasdata4/jaemin/data/ABIDE_new/csv/df_ABIDE2_1025.csv')
        df = pd.read_csv('/nasdata4/jaemin/temp/df_temp_3.csv')
    elif args.dataset == 'ADHD':
        df = pd.read_csv('/nasdata3/jaemin/ADHD/csv/df_ADHD_768.csv', dtype={'ID': str})
        
        df['DX'].replace({2: 1, 3: 1}, inplace=True)
        df = df.sort_values(by=['SITE_ID', 'DX'], ascending=False)
        
        df = balanced_binary_sampling(df, seed=args.seed)   # subsample to balance classes

    # To split data considering both SITE_ID and DX for stratification
    df['DX_SITE'] = df['DX'].astype(str) + '_' + df['SITE_ID'].astype(str)

    df = df.reset_index(drop=True).reset_index().rename(columns={'index': 'orig_index'})
    
    # Cross-validation
    test_aucs = []; test_accs = []; test_sens = []; test_spes = []; test_pres = []
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=args.seed)
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(df, df['DX_SITE'])):
        print(f"\n============================== Fold {fold+1} ==============================") 
        df_train_val_raw = df.iloc[train_val_idx]
        df_test = df.iloc[test_idx]
        
        df_train_raw, df_val_raw = train_test_split(df_train_val_raw, test_size=0.2, stratify=df_train_val_raw['DX_SITE'], random_state=args.seed+fold)
        
        df_train_val = df_train_val_raw.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        
        train_orig_set = set(df_train_raw['orig_index'])
        val_orig_set = set(df_val_raw['orig_index'])
        train_ind = df_train_val.index[df_train_val['orig_index'].isin(train_orig_set)].tolist()
        val_ind = df_train_val.index[df_train_val['orig_index'].isin(val_orig_set)].tolist()
        
        model = IPGNN(
            device=device,
            in_dim=IN_DIM,
            num_pheno_features=len(args.phenotypic_features),
            hid_dim_gnn=args.hidden_dim_gnn,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=args.num_classes,
            global_edge_thr=args.global_edge_thr,
            hid_dim_edge=args.hidden_dim_edge,
            dropout_edge=args.edge_dropout,
        ).to(device)

        print(f"======================== Phase 1 (Training) ========================") 
        train_val_cor_mats, train_val_phenotypic_dict, train_val_labels = load_data(df_train_val, args.phenotypic_features, args.dataset, args.method, args.atlas)
        train(
            model,
            pheno_dict=train_val_phenotypic_dict,
            x=torch.FloatTensor(train_val_cor_mats).to(device),
            y=torch.LongTensor(train_val_labels).to(device),
            train_ind=train_ind,
            val_ind=val_ind,
            opt=args,
            model_path=f'{args.result_path}/best_model_val_{fold+1}.pth'
        )
        
        print(f"\n======================== Phase 2 (Inductive inference) ========================")
        print(f'# Number of test samples: {len(df_test)}')
        model.load_state_dict(torch.load(f'{args.result_path}/best_model_val_{fold+1}.pth', map_location=device))
        
        y_probs = []; y_trues = []
        if args.test_batch == -1:    # if args.test_batch is -1, use all test samples at once for inference
            args.test_batch = len(df_test)
            
        for start in range(0, len(df_test), args.test_batch):
            end = min(start + args.test_batch, len(df_test))
            if args.test_batch > 1:
                print(f'=== [ Test Samples {start+1:3d} ~ {end:3d} ] ===')
            else:
                print(f'=== [ Test Sample {start+1:3d} ] ===')
                
            # Select "args.test_batch" test samples 
            df_test_subset = df_test.iloc[start:end]

            # train+val + test_batch
            df_combined = pd.concat([df_train_val, df_test_subset], axis=0)
            df_combined = df_combined.reset_index(drop=True)

            # orig → local mapping
            orig2local = (df_combined.reset_index().set_index('orig_index')['index'].to_dict())
            test_ind_combined  = [orig2local[orig] for orig in df_test_subset['orig_index']]
        
            all_cor_mats, all_phenotypic_dict, all_labels = load_data(df_combined, args.phenotypic_features, args.dataset, args.method, args.atlas)
            
            test_logit, test_label = inductive_inference(
                model,
                pheno_dict=all_phenotypic_dict, 
                x=torch.FloatTensor(all_cor_mats).to(device), 
                y=torch.LongTensor(all_labels).to(device), 
                test_ind=test_ind_combined, 
            )
            
            y_probs.extend(test_logit.tolist())
            y_trues.extend(test_label.tolist())
        
        # Compute test metrics
        y_preds = [np.round(y) for y in y_probs]
        auc_test = roc_auc_score(y_trues, y_probs)
        acc_test = np.mean(np.array(y_preds) == np.array(y_trues))
        tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
        sen_test = tp / (tp + fn)
        spe_test = tn / (tn + fp)
        pre_test = tp / (tp + fp)
        test_aucs.append(auc_test); test_accs.append(acc_test); test_sens.append(sen_test); test_spes.append(spe_test); test_pres.append(pre_test)
        
        print(f"\n => Fold {fold+1} |  Test accuacry : {acc_test:.5f}, Test AUC : {auc_test:.5f}")
    
    print("\n========================== Finish ==========================") 
    
    print(f'# Test Accuracies : {np.mean(test_accs)*100:.1f} ({np.std(test_accs)*100:.2f})')
    print(f'## {test_accs}\n')
    print(f'# Test AUCs : {np.mean(test_aucs)*100:.1f} ({np.std(test_aucs)*100:.2f})')
    print(f'## {test_aucs}\n')   
    print(f'# Test Sensitivities : {np.mean(test_sens)*100:.1f} ({np.std(test_sens)*100:.2f})')
    print(f'## {test_sens}\n')
    print(f'# Test Specificities : {np.mean(test_spes)*100:.1f} ({np.std(test_spes)*100:.2f})')
    print(f'## {test_spes}\n')
    print(f'# Test Precisions : {np.mean(test_pres)*100:.1f} ({np.std(test_pres)*100:.2f})')
    print(f'## {test_pres}\n')

if __name__ == '__main__':
    start = time.time()
    main()
    running_time = time.time() - start
    running_time = str(datetime.timedelta(seconds=running_time)).split('.')[0]
    print(f'Running Time : {running_time}')