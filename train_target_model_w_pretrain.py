
import argparse
from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model_w_pretrain import GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import random_split
import os
from utils import set_seed


criterion = nn.BCEWithLogitsLoss(reduction="none")


def train(model, device, loader, optimizer):
    model.train()

    pbar = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(pbar):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y**2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        # Loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix(loss=loss.item())

def eval(model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0)
    y_scores = torch.cat(y_scores, dim=0)
    y_scores = torch.sigmoid(y_scores)

    y_true = y_true.cpu().numpy()
    y_scores = y_scores.cpu().numpy()

    
    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list)

def compute_label_distribution(dataset, num_tasks, dataset_name):
    """Compute and print the positive/negative label distribution for a dataset."""
    labels = dataset.data.y.cpu().numpy()  # Shape: [num_samples, num_tasks] or [num_samples]
    print(f"\nLabel distribution for {dataset_name} dataset:")
    
    # Handle both 1D and 2D label arrays
    if labels.ndim == 1:
        # Single-task case: treat as [num_samples, 1]
        labels = labels[:, np.newaxis]  # Reshape to [num_samples, 1]
        num_tasks = 1  # Override num_tasks for single-task datasets

    
    for task_idx in range(num_tasks):
        task_labels = labels[:, task_idx]
        total_valid = np.sum(task_labels**2 > 0)  # Count non-null labels
        if total_valid == 0:
            print(f"Task {task_idx}: No valid labels.")
            continue
        
        pos_count = np.sum(task_labels == 1)
        neg_count = np.sum(task_labels == -1)
        pos_ratio = pos_count / total_valid if total_valid > 0 else 0
        neg_ratio = neg_count / total_valid if total_valid > 0 else 0
        
        print(f"Task {task_idx}:")
        print(f"  Total valid samples: {total_valid}")
        print(f"  Positive (1) samples: {pos_count} ({pos_ratio:.4f})")
        print(f"  Negative (-1) samples: {neg_count} ({neg_ratio:.4f})")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dataset', type=str, default='tox21',
                        help='root directory of dataset.')
    parser.add_argument('--input_model_file', type=str, default='pretrain_model/model_gin/supervised_contextpred.pth',
                        help='filename to read the model')
    parser.add_argument('--checkpoint_dir', type=str, default='model_weights_w_pretrain',
                        help='directory to save the best model checkpoint')
    parser.add_argument('--dataseed', type=int, default=41,
                        help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=2,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--eval_train', type=int, default=0,
                        help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for dataset loading')
    args = parser.parse_args()


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    set_seed(args.runseed)

    # Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 1
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "bace":
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    # Set checkpoint path with dataset name
    checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_{args.dataset}.pth')
    os.makedirs(args.checkpoint_dir, exist_ok=True)


    # Set up dataset
    dataset = MoleculeDataset("dataset_w_pretrain/dataset/" + args.dataset, dataset=args.dataset)

    # First split: 4:4:2 into target_model_dataset, shadow_dataset, test_dataset
    target_model_dataset, shadow_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.4, frac_valid=0.4, frac_test=0.2, seed=args.dataseed)
    print("First split: d1 (40%), d2 (40%), d3 (20%)")

    # Second split: d1 into train (80%) and val (20%)
    train_dataset, val_dataset, _ = random_split(target_model_dataset, null_value=0, frac_train=0.8, frac_valid=0.2, frac_test=0, seed=args.dataseed)
    print("Second split: train (80% of d1), val (20% of d1), test (d3)")


    # Compute and print label distribution for train, valid, and test sets
    compute_label_distribution(train_dataset, num_tasks, "Train")
    compute_label_distribution(val_dataset, num_tasks, "Validation")
    compute_label_distribution(test_dataset, num_tasks, "Test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks)
    

    if args.input_model_file != "":
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    best_val_roc_auc = float('-inf')  # Track the best validation ROC-AUC
    best_model_state = None  # Store the best model state in memory
    

    for epoch in range(1, args.epochs + 1):
        print("==================epoch " + str(epoch))
        
        train(model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_roc_auc = eval(model, device, train_loader)
        else:
            print("omit the training ROC-AUC computation")
            train_roc_auc = 0

        val_roc_auc = eval(model, device, val_loader)

        print("train ROC-AUC: %f val ROC-AUC: %f" % (train_roc_auc, val_roc_auc))

        # Update best model state if validation ROC-AUC improves
        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            best_model_state = model.state_dict()
            print(f"New best validation ROC-AUC: {best_val_roc_auc}")

        print("")

    # Save the best model to disk
    if best_model_state is not None:
        torch.save(best_model_state, checkpoint_path)
        print(f"Saved best model to {checkpoint_path} with val ROC-AUC: {best_val_roc_auc}")
    else:
        print("No valid model state was found during training.")

    # Load the best model and evaluate on test set
    print("====Final Test Evaluation with Best Model")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("Using final model state for test evaluation (no improvement found).")
    model.eval()
    final_test_roc_auc = eval(model, device, test_loader)
    print(f"Final test ROC-AUC with best model: {final_test_roc_auc}")

if __name__ == "__main__":
    main()
