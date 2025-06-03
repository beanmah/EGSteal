# target_model_inference_w_pretrain.py

import argparse
import os
from loader import MoleculeDataset
from torch_geometric.data import DataLoader, Data

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

from model_w_pretrain import GNN_graphpred, CAM_GNNGraphPred
from splitters import random_split
from utils import set_seed

def eval(model, cam, device, loader, results_file):
    """
    Evaluate the model on the given data loader and save inference results.

    Parameters:
    - model: GNN_graphpred model
    - cam: CAM_GNNGraphPred instance for computing explanations
    - device: Device to run the model
    - loader: DataLoader for the dataset
    - results_file: Path to save inference results

    Returns:
    - results: List of dictionaries containing inference results
    """
    model.eval()
    results = []
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Inference")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Calculate CAM explanations
        target_classes = (batch.y + 1) // 2  # Map {-1, 1} to {0, 1}
        explanations = cam.get_cam_scores(target_classes, batch.batch)
        node_mask = explanations  # [total_num_nodes]

        # Collect predictions and true labels for AUC
        y_true.append(batch.y.view(pred.shape))  # Ensure same shape as pred
        y_scores.append(pred)

        original_graphs = batch.to_data_list()
        pred_prob_np = torch.sigmoid(pred).cpu().numpy()
        num_nodes_per_graph = batch.ptr[1:] - batch.ptr[:-1]
        node_masks_list = torch.split(node_mask, num_nodes_per_graph.tolist())

        # Save each sample's prediction and explanation results
        for i, (graph, node_m) in enumerate(zip(original_graphs, node_masks_list)):
            data = Data(
                x=graph.x.cpu(),
                edge_index=graph.edge_index.cpu(),
                edge_attr=graph.edge_attr.cpu() if graph.edge_attr is not None else None,
                y=graph.y.cpu()
            )

            # Determine predicted class: -1 or 1 based on probability threshold (0.5)
            pred_class = 1 if pred_prob_np[i] >= 0.5 else -1

            # Save results
            results.append({
                'original_data': data,
                'pred': int(pred_class),
                'node_mask': node_m.to('cpu')
            })


    y_true = torch.cat(y_true, dim=0)
    y_scores = torch.cat(y_scores, dim=0)

    y_scores = torch.sigmoid(y_scores)
    y_true = y_true.cpu().numpy()
    y_scores = y_scores.cpu().numpy()

    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]  # Reshape to (n_samples, 1) for single task
        y_scores = y_scores[:, np.newaxis]


    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive and one negative sample
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i]**2 > 0  # Valid labels (non-zero)
            # Convert labels to 0/1 for roc_auc_score
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
        else:
            print(f"Task {i} skipped: insufficient positive/negative samples")

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print(f"Missing ratio: {1 - float(len(roc_list)) / y_true.shape[1]:.4f}")

    avg_roc_auc = sum(roc_list) / len(roc_list) if roc_list else 0
    print(f"Dataset ROC-AUC scores: {roc_list}")
    print(f"Average ROC-AUC: {avg_roc_auc:.4f}")

    # Save inference results to .pt file
    torch.save(results, results_file)
    print(f"Inference results saved to {results_file}")

    return results

def main():
    # Command-line arguments
    parser = argparse.ArgumentParser(description='PyTorch implementation of GNN inference with CAM explanations')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='input batch size for inference (default: 64)')
    parser.add_argument('--dataset', type=str, default='tox21',
                        help='root directory of dataset (default: tox21)')
    parser.add_argument('--checkpoint_dir', type=str, default='model_weights_w_pretrain',
                        help='directory where the best model checkpoint is saved')
    parser.add_argument('--dataseed', type=int, default=41,
                        help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=2,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for dataset loading')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.runseed)

    # Set number of tasks
    if args.dataset == "tox21":
        num_tasks = 1
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "bace":
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    # Set checkpoint path
    checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_{args.dataset}.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.")

    # Set results file paths
    results_dir = os.path.join("dataset_w_pretrain/dataset", args.dataset, "processed_splits")
    os.makedirs(results_dir, exist_ok=True)
    shadow_results_file = os.path.join(results_dir, 'queried_dataset_shadow.pt')
    test_results_file = os.path.join(results_dir, 'queried_dataset_test.pt')

    # Load dataset
    dataset = MoleculeDataset("dataset_w_pretrain/dataset/" + args.dataset, dataset=args.dataset)

    # Split: 4:4:2 into target_model_dataset, shadow_dataset, test_dataset
    target_model_dataset, shadow_dataset, test_dataset = random_split(
        dataset, null_value=0, frac_train=0.4, frac_valid=0.4, frac_test=0.2, seed=args.dataseed
    )

    print("First split: target_model (40%), shadow (40%), test (20%)")
    print(f"Shadow dataset size: {len(shadow_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create data loaders
    shadow_loader = DataLoader(shadow_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Set up model
    model = GNN_graphpred(num_layers=5, emb_dim=300, num_tasks=num_tasks)
    model.to(device)

    # Load best model weights
    print(f"Loading best model from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Initialize CAM
    cam = CAM_GNNGraphPred(model)

    # Perform inference on shadow data
    print("====Inference on Shadow Data")
    shadow_results = eval(model, cam, device, shadow_loader, shadow_results_file)
    print(f"Completed inference on {len(shadow_results)} samples from shadow dataset.")

    # Perform inference on test data
    print("====Inference on Test Data")
    test_results = eval(model, cam, device, test_loader, test_results_file)
    print(f"Completed inference on {len(test_results)} samples from test dataset.")

if __name__ == "__main__":
    main()
