# train_surrogate_model_for_target_w_pretrain.py

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
import os.path as osp
import os
import argparse
import random
import math
import numpy as np
from tqdm import tqdm
from scipy.stats import kendalltau
from collections import defaultdict
from model import GIN, GCN, GAT, GraphSAGE, SurrogateModel, Classifier, CAM
from utils import set_seed, RankNetLoss, DataAugmentor, safe_auc
from sklearn.metrics import roc_auc_score


def custom_collate(batch):
    """
    Custom collate function to batch PyTorch Geometric Data objects.

    Args:
        batch: List[Data]

    Returns:
        Batch object
    """
    return Batch.from_data_list(batch)


def process_query_dataset(query_dataset):
    """
    Process inference dataset, converting samples to Data objects and filtering invalid samples with y=0.

    Parameters:
    - query_dataset: List[dict], containing 'original_data', 'pred', and 'node_mask' fields

    Returns:
    - processed_data_list: List[Data], each Data object contains original features, y, target_pred, and node_mask
    """
    processed_data_list = []
    for sample in query_dataset:
        original_data = sample['original_data']
        pred = sample['pred']
        node_mask = sample['node_mask']

        # Filter invalid samples (y=0)
        if original_data.y.item() == 0:
            continue

        if isinstance(pred, torch.Tensor):
            pred = pred.item()
        elif isinstance(pred, (list, np.ndarray)):
            pred = pred[0]

        # Map pred from {-1, 1} to {0, 1}
        target_pred = 1 if pred == 1 else 0
        
        # Map y from {1, -1} to {1, 0}
        y_mapped = 1 if original_data.y.item() == 1 else 0
        y_tensor = torch.tensor([y_mapped], dtype=torch.long, device=original_data.x.device)
        
        # Create new Data object
        new_data = Data(
            x=original_data.x,
            edge_index=original_data.edge_index,
            edge_attr=getattr(original_data, 'edge_attr', None),
            y=y_tensor,
            target_pred=torch.tensor(target_pred, dtype=torch.long, device=original_data.x.device),
            node_mask=node_mask
        )

        processed_data_list.append(new_data)
    return processed_data_list


def augment(dataset, augmentor, augmentation_ratio, operation_ratio=0.1, augmentation_type='combined'):
    """
    Generate augmented samples based on the chosen augmentation strategy.

    Parameters:
    - dataset: Original training dataset (list of Data objects)
    - augmentor: DataAugmentor instance
    - augmentation_ratio: Ratio of augmented samples to generate
    - operation_ratio: Proportion for augmentation operations (e.g., node drop or edge perturbation)
    - augmentation_type: Type of augmentation ('drop_node', 'drop_edge', 'add_edge', 'combined')

    Returns:
    - augmented_data_list: List of augmented Data objects
    """
    augmented_data_list = []
    num_original_samples = len(dataset)
    num_augmented_samples = int(num_original_samples * augmentation_ratio)

    if num_augmented_samples == 0:
        return augmented_data_list

    # Group samples by label
    label_to_samples = defaultdict(list)
    for sample in dataset:
        label = sample.target_pred.item()
        label_to_samples[label].append(sample)

    # Calculate inverse frequency and normalize
    label_weights = {}
    total_inverse = 0
    for label, samples in label_to_samples.items():
        Ni = len(samples)
        if Ni == 0:
            continue
        inverse_freq = 1.0 / Ni
        label_weights[label] = inverse_freq
        total_inverse += inverse_freq

    for label in label_weights:
        label_weights[label] /= total_inverse

    # Calculate number of augmented samples per label
    num_augmented_samples_per_label = {}
    for label, weight in label_weights.items():
        num_aug = int(num_augmented_samples * weight)
        num_augmented_samples_per_label[label] = num_aug

    # Allocate remaining samples
    remaining = num_augmented_samples - sum(num_augmented_samples_per_label.values())
    labels = list(label_weights.keys())
    for i in range(remaining):
        label = labels[i % len(labels)]
        num_augmented_samples_per_label[label] += 1

    # Generate augmented samples
    for label, num_aug in num_augmented_samples_per_label.items():
        samples = label_to_samples[label]
        if len(samples) < 1:
            print(f"Label {label} has insufficient samples to generate augmented samples.")
            continue

        for _ in range(num_aug):
            try:
                sample = random.choice(samples)
                if augmentation_type == 'drop_node':
                    augmented_data = augmentor.drop_node(sample, drop_ratio=operation_ratio)
                elif augmentation_type == 'drop_edge':
                    augmented_data = augmentor.drop_edge(sample, drop_ratio=operation_ratio)
                elif augmentation_type == 'add_edge':
                    augmented_data = augmentor.add_edge(sample, add_ratio=operation_ratio)
                elif augmentation_type == 'combined':
                    augmented_data = augmentor.combined_augmentation(
                        sample,
                        drop_node_ratio=operation_ratio,
                        drop_edge_ratio=operation_ratio,
                        add_edge_ratio=operation_ratio
                    )
                else:
                    print(f"Unknown augmentation type: {augmentation_type}")
                    continue

                if augmented_data is None:
                    continue

                # Skip single-node graphs
                if augmented_data.x.size(0) <= 1: 
                    continue

                augmented_data_list.append(augmented_data)

            except Exception as e:
                print(f"Error generating augmented sample for label {label}: {e}")
                continue

    return augmented_data_list


def train(model, dataloader, optimizer, device, align_weight=1.0, criterion=None, ranknet_loss_fn=None):
    """
    Training function using target_pred as labels, incorporating RankNet loss.

    Parameters:
    - model: SurrogateModel instance
    - dataloader: DataLoader for training data
    - optimizer: Optimizer for model parameters
    - device: Device to run the model (CPU or CUDA)
    - align_weight: Weight for RankNet loss
    - criterion: Loss function for prediction
    - ranknet_loss_fn: RankNet loss function

    Returns:
    - avg_loss_pred: Average prediction loss
    - avg_ranknet_loss: Average RankNet loss
    """
    model.train()
    total_loss_pred = 0.0
    total_ranknet_loss = 0.0
    total_samples = 0

    ex = CAM(model)

    for batch_samples in dataloader:
        optimizer.zero_grad()
        all_data = batch_samples.to(device)

        node_emb, out_surr = model(all_data.x, all_data.edge_index, all_data.batch)
        loss_pred = criterion(out_surr, all_data.target_pred)

        # Calculate CAM scores
        cam_scores = ex.get_cam_scores(all_data.target_pred, all_data.batch)
        node_masks = all_data.node_mask
        batch_ids = all_data.batch

        ranknet_loss = torch.tensor(0.0, device=device)
        if align_weight != 0:
            ranknet_loss = ranknet_loss_fn(cam_scores, node_masks, batch_ids)

        total_batch_loss = loss_pred + align_weight * ranknet_loss
        total_batch_loss.backward()
        optimizer.step()

        batch_size = len(batch_samples)
        total_loss_pred += loss_pred.item() * batch_size
        total_ranknet_loss += ranknet_loss.item() * batch_size
        total_samples += batch_size

    avg_loss_pred = total_loss_pred / total_samples
    avg_ranknet_loss = total_ranknet_loss / total_samples
    return avg_loss_pred, avg_ranknet_loss


def eval(model, dataloader, device):
    """
    Evaluate the model on the validation set, computing accuracy and AUC.

    Parameters:
    - model: SurrogateModel instance
    - dataloader: DataLoader for validation data
    - device: Device to run the model (CPU or CUDA)

    Returns:
    - accuracy: Validation accuracy
    - auc: Validation AUC
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch_samples in dataloader:
            batch = batch_samples.to(device)
            target_preds_tensor = batch.target_pred

            node_emb, out_surr = model(batch.x, batch.edge_index, batch.batch)
            pred = out_surr.argmax(dim=1)

            total_correct += pred.eq(target_preds_tensor).sum().item()
            total_samples += len(batch_samples)

            all_targets.extend(target_preds_tensor.cpu().numpy())
            all_probs.extend(F.softmax(out_surr, dim=1)[:, 1].cpu().numpy())

    accuracy = total_correct / total_samples
    try:
        auc = safe_auc(all_targets, all_probs)
    except ValueError:
        auc = float('nan')

    return accuracy, auc


def calculate_rank_correlation(pred_scores, true_scores, batch_ids):
    """
    Calculate average Kendall's tau correlation between predicted and true node importance scores.

    Parameters:
    - pred_scores: Predicted node importance scores [total_num_nodes]
    - true_scores: True node importance scores [total_num_nodes]
    - batch_ids: Graph index for each node [total_num_nodes]

    Returns:
    - Mean correlation across graphs
    """
    correlations = []
    for b in torch.unique(batch_ids):
        mask = (batch_ids == b)
        p = pred_scores[mask].cpu().numpy()
        t = true_scores[mask].cpu().numpy()
        corr, _ = kendalltau(p, t)
        if not np.isnan(corr):
            correlations.append(corr)
    return np.mean(correlations) if correlations else float('nan')


def calculate_order_accuracy(pred_scores, true_scores, batch_ids):
    """
    Calculate order accuracy of CAM scores.

    Parameters:
    - pred_scores: Predicted node importance scores [total_num_nodes]
    - true_scores: True node importance scores [total_num_nodes]
    - batch_ids: Graph index for each node [total_num_nodes]

    Returns:
    - Mean order accuracy across graphs
    """
    unique_batch = torch.unique(batch_ids)
    per_graph_accuracies = []

    for b in unique_batch:
        mask = (batch_ids == b)
        p = pred_scores[mask]
        t = true_scores[mask]
        num_nodes = p.size(0)

        if num_nodes < 2:
            continue

        indices_i, indices_j = torch.triu_indices(num_nodes, num_nodes, offset=1, device=p.device)
        s_i = t[indices_i]
        s_j = t[indices_j]
        p_i = p[indices_i]
        p_j = p[indices_j]

        true_relation = torch.where(s_i > s_j, torch.ones_like(s_i),
                                   torch.where(s_i < s_j, torch.ones_like(s_i) * -1, torch.zeros_like(s_i)))
        pred_relation = torch.where(p_i > p_j, torch.ones_like(p_i),
                                   torch.where(p_i < p_j, torch.ones_like(p_i) * -1, torch.zeros_like(p_i)))

        correct = true_relation.eq(pred_relation).float()
        correct_total = correct.sum().item()
        total = correct.numel()

        if total > 0:
            graph_accuracy = correct_total / total
            per_graph_accuracies.append(graph_accuracy)

    return sum(per_graph_accuracies) / len(per_graph_accuracies) if per_graph_accuracies else float('nan')


def test(model, dataloader, device):
    """
    Evaluate the model on the test set, computing accuracy, AUC, fidelity, order accuracy, and rank correlation.

    Parameters:
    - model: SurrogateModel instance
    - dataloader: DataLoader for test data
    - device: Device to run the model (CPU or CUDA)

    Returns:
    - accuracy: Test accuracy
    - auc: Test AUC
    - fidelity_score: Fidelity score
    - order_accuracy: Order accuracy
    - rank_correlation: Rank correlation
    """
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_scores = []
    all_predictions = []
    all_target_preds = []
    all_pred_scores = []
    all_true_scores = []
    all_batch_ids = []
    graph_offset = 0

    ex = CAM(model)

    with torch.no_grad():
        for batch_samples in dataloader:
            batch = batch_samples.to(device)
            true_labels_tensor = batch.y  # Ground truth for AUC and Accuracy ({1, 0})
            target_preds_tensor = batch.target_pred  # Ground truth for Fidelity ({0, 1})

            if not torch.all((true_labels_tensor == 1) | (true_labels_tensor == 0)):
                print(f"Warning: Invalid y values found in batch: {true_labels_tensor.unique()}")
                continue

            # Model forward pass
            node_emb, out_surr = model(batch.x, batch.edge_index, batch.batch)
            pred = out_surr.argmax(dim=1)

            # Accuracy calculation: based on batch.y
            correct += pred.eq(true_labels_tensor).sum().item()
            total += len(true_labels_tensor)

            # Collect AUC data
            y_true.append(true_labels_tensor.cpu())
            y_scores.append(F.softmax(out_surr, dim=1)[:, 1].cpu())

            # Fidelity calculation: based on batch.target_pred
            all_predictions.append(pred)
            all_target_preds.append(target_preds_tensor)

            # CAM scores and node_mask for Order Accuracy and Rank Correlation
            cam_scores = ex.get_cam_scores(target_preds_tensor, batch.batch)
            all_pred_scores.append(cam_scores.cpu())
            all_true_scores.append(batch.node_mask.cpu())
            all_batch_ids.append(batch.batch.cpu() + graph_offset)

            graph_offset += batch.num_graphs

    # Concatenate y_true and y_scores
    y_true = torch.cat(y_true, dim=0).numpy()
    y_scores = torch.cat(y_scores, dim=0).numpy()

    # Calculate AUC
    if np.sum(y_true == 1) > 0 and np.sum(y_true == 0) > 0:
        auc = roc_auc_score(y_true, y_scores)
    else:
        print("Warning: Not enough positive or negative samples for AUC calculation.")
        auc = float('nan')

    # Calculate Accuracy
    accuracy = correct / total if total > 0 else float('nan')

    # Calculate Fidelity
    all_predictions = torch.cat(all_predictions)
    all_target_preds = torch.cat(all_target_preds)
    total_fidelity = all_target_preds.size(0)
    fidelity_score = all_predictions.eq(all_target_preds).sum().item() / total_fidelity if total_fidelity > 0 else float('nan')

    # Calculate Order Accuracy and Rank Correlation
    all_pred_scores = torch.cat(all_pred_scores)
    all_true_scores = torch.cat(all_true_scores)
    all_batch_ids = torch.cat(all_batch_ids)
    order_accuracy = calculate_order_accuracy(all_pred_scores, all_true_scores, all_batch_ids)
    rank_correlation = calculate_rank_correlation(all_pred_scores, all_true_scores, all_batch_ids)

    return accuracy, auc, fidelity_score, order_accuracy, rank_correlation


def random_split_list(dataset, frac_train=0.8, frac_valid=0.2, frac_test=0.0, seed=None):
    """
    Custom split function for List[Dict] dataset.

    Parameters:
    - dataset: List of dictionaries containing dataset samples
    - frac_train: Fraction of data for training
    - frac_valid: Fraction of data for validation
    - frac_test: Fraction of data for testing
    - seed: Random seed for reproducibility

    Returns:
    - train_dataset: Training subset
    - valid_dataset: Validation subset
    - test_dataset: Test subset
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    N = len(dataset)
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(N)
    train_size = int(frac_train * N)
    valid_size = int(frac_valid * N)
    train_idx = indices[:train_size]
    valid_idx = indices[train_size:train_size + valid_size]
    test_idx = indices[train_size + valid_size:]
    train_dataset = [dataset[i] for i in train_idx]
    valid_dataset = [dataset[i] for i in valid_idx]
    test_dataset = [dataset[i] for i in test_idx]
    return train_dataset, valid_dataset, test_dataset


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load inference data
    data_dir = os.path.join("dataset_w_pretrain/dataset", args.dataset, "processed_splits")
    queried_shadow = torch.load(osp.join(data_dir, 'queried_dataset_shadow.pt'))
    test_dataset = torch.load(osp.join(data_dir, 'queried_dataset_test.pt'))
    print(f"Loaded shadow dataset: {len(queried_shadow)} samples")
    print(f"Loaded test dataset: {len(test_dataset)} samples")

    # Split shadow_dataset
    queried_shadow_valid, _, _ = random_split_list(
        queried_shadow, frac_train=0.3, frac_valid=0.7, frac_test=0.0, seed=args.seed
    )

    train_dataset, val_dataset, _ = random_split_list(
        queried_shadow_valid, frac_train=0.8, frac_valid=0.2, frac_test=0.0, seed=args.seed
    )

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")

    input_dim = queried_shadow[0]['original_data'].x.size(1)
    num_classes = 2

    # Initialize model
    if args.gnn_backbone == 'GIN':
        encoder = GIN(input_dim=input_dim, hidden_dim=args.gnn_hidden_dim, num_layers=args.gnn_layer).to(device)
    elif args.gnn_backbone == 'GCN':
        encoder = GCN(input_dim=input_dim, hidden_dim=args.gnn_hidden_dim, num_layers=args.gnn_layer).to(device)
    elif args.gnn_backbone == 'GAT':
        encoder = GAT(input_dim=input_dim, hidden_dim=args.gnn_hidden_dim, num_layers=args.gnn_layer, heads=args.gat_heads).to(device)
    elif args.gnn_backbone == 'GraphSAGE':
        encoder = GraphSAGE(input_dim=input_dim, hidden_dim=args.gnn_hidden_dim, num_layers=args.gnn_layer).to(device)
    else:
        raise ValueError(f"Invalid GNN backbone: {args.gnn_backbone}")

    predictor = Classifier(input_dim=args.gnn_hidden_dim, output_dim=num_classes).to(device)
    model = SurrogateModel(encoder=encoder, predictor=predictor).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    ranknet_loss_fn = RankNetLoss().to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Process datasets
    processed_train_dataset = process_query_dataset(train_dataset)
    processed_val_dataset = process_query_dataset(val_dataset)
    processed_test_dataset = process_query_dataset(test_dataset)
    print(f"Processed Train Dataset Size (after filtering): {len(processed_train_dataset)}")
    print(f"Processed Val Dataset Size: {len(processed_val_dataset)}")
    print(f"Processed Test Dataset Size (after filtering): {len(processed_test_dataset)}")

    augmentor = DataAugmentor()

    val_loader = DataLoader(processed_val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(processed_test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)

    best_val_auc = -math.inf
    best_model_state = None

    print("Starting training...")
    with tqdm(total=args.epochs, desc='Training') as epoch_pbar:
        for epoch_num in range(1, args.epochs + 1):
            augmented_data = augment(
                dataset=processed_train_dataset,
                augmentor=augmentor,
                augmentation_ratio=args.augmentation_ratio,
                operation_ratio=args.operation_ratio,
                augmentation_type=args.augmentation_type
            )
            combined_train_dataset = processed_train_dataset + augmented_data
            combined_train_loader = DataLoader(
                combined_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate
            )

            train_loss_pred, train_ranknet_loss = train(
                model, combined_train_loader, optimizer, device,
                align_weight=args.align_weight, criterion=criterion, ranknet_loss_fn=ranknet_loss_fn
            )

            val_acc, val_auc = eval(model, val_loader, device)

            epoch_pbar.set_postfix({
                'Train Pred Loss': f'{train_loss_pred:.4f}',
                'Train RankNet Loss': f'{train_ranknet_loss:.4f}',
                'Val Acc': f'{val_acc:.4f}',
                'Val AUC': f'{val_auc:.4f}'
            })
            epoch_pbar.update(1)

            # Save best model
            if not math.isnan(val_auc) and val_auc >= best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict()

    # Test best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        test_acc, test_auc, fidelity_score, order_accuracy, rank_correlation = test(model, test_loader, device)

        print(f"\nBest Validation AUC: {best_val_auc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Fidelity Score: {fidelity_score:.4f}")
        print(f"Order Accuracy: {order_accuracy:.4f}")
        print(f"Rank Correlation: {rank_correlation:.4f}")

        results = {
            'test_acc': test_acc,
            'test_auc': test_auc,
            'fidelity_score': fidelity_score,
            'order_accuracy': order_accuracy,
            'rank_correlation': rank_correlation
        }

        save_dir_model = osp.join('model_weights_w_pretrain', args.dataset)
        os.makedirs(save_dir_model, exist_ok=True)
        results_save_path = osp.join(save_dir_model, 'surrogate_results.pt')
        torch.save(results, results_save_path)
        print(f"Results saved to {results_save_path}")
    else:
        print("No improvement in validation AUC during training.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Surrogate Model with Explainability')
    parser.add_argument('--dataset', type=str, default='tox21',
                       help='Dataset name (default: tox21)')
    parser.add_argument('--seed', type=int, default=41,
                       help='Seed for training and random initialization')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--gnn_layer', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--gnn_hidden_dim', type=int, default=128,
                       help='GNN hidden dimension')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--augmentation_ratio', type=float, default=0.1,
                       help='Ratio of augmented samples')
    parser.add_argument('--operation_ratio', type=float, default=0.05,
                       help='Proportion for augmentation operations')
    parser.add_argument('--align_weight', type=float, default=1,
                       help='Weight for RankNet loss')
    parser.add_argument('--augmentation_type', type=str, default='combined',
                       choices=['drop_node', 'drop_edge', 'add_edge', 'combined'],
                       help='Type of data augmentation')
    parser.add_argument('--gat_heads', type=int, default=4,
                       help='Number of attention heads for GAT')
    parser.add_argument('--gnn_backbone', type=str, default='GIN',
                       choices=['GIN', 'GCN', 'GAT', 'GraphSAGE'],
                       help='GNN backbone type')
    args = parser.parse_args()

    main(args)
