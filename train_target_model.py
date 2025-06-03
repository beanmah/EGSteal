# train_target_model.py

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
import os.path as osp
import os
from tqdm import tqdm
from model import GIN, GCN, GAT, GraphSAGE, TargetModel, Classifier
from utils import set_seed, safe_auc
import argparse
import numpy as np



def train(model, dataloader, optimizer, device, args):
    """Train the model for one epoch, returning average loss, accuracy, and AUC."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    criterion = torch.nn.CrossEntropyLoss()
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        if args.explanation_mode in ['GNNExplainer', 'PGExplainer']:
            out = model(data.x, data.edge_index, data.batch)
        else:
            logits, out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

        # Calculate accuracy
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.num_graphs

        all_labels.extend(data.y.cpu().numpy())
        probs = F.softmax(out, dim=1).detach().cpu().numpy()
        all_probs.extend(probs)

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total

    binary_labels = np.array(all_labels)
    binary_probs = np.array([prob[1] for prob in all_probs])
    auc = safe_auc(binary_labels, binary_probs)

    return avg_loss, accuracy, auc



def evaluate(model, dataloader, device, args):
    """Evaluate the model on the validation or test set, returning average loss, accuracy, and AUC."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            if args.explanation_mode in ['GNNExplainer','PGExplainer']:
                out = model(data.x, data.edge_index, data.batch)
            else:
                _, out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs

            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.num_graphs

            all_labels.extend(data.y.cpu().numpy())
            probs = F.softmax(out, dim=1).detach().cpu().numpy()
            all_probs.extend(probs)

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total

    binary_labels = np.array(all_labels)
    binary_probs = np.array([prob[1] for prob in all_probs])
    auc = safe_auc(binary_labels, binary_probs)

    return avg_loss, accuracy, auc



def main(args):
    # Set random seed
    set_seed(args.seed)

    # Load Target training set, validation set, and test set
    root = args.dataset_root
    dataset_name = args.dataset_name
    save_dir = osp.join(root, dataset_name, 'processed_splits')
    target_train_dataset = torch.load(osp.join(save_dir, 'target_train_dataset.pt'))
    target_val_dataset = torch.load(osp.join(save_dir, 'target_val_dataset.pt'))
    test_dataset = torch.load(osp.join(save_dir, 'test_dataset.pt'))

    # Create DataLoader
    batch_size = args.batch_size
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
    target_val_loader = DataLoader(target_val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get dataset information
    info = torch.load(osp.join(save_dir, 'dataset_info.pt'))
    input_dim = info['num_node_features']
    num_classes = info['num_classes']
    print(f"Number of node features: {input_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Target training dataset size: {len(target_train_dataset)}")
    print(f"Target validation dataset size: {len(target_val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    if args.gnn_backbone == 'GIN':
        encoder = GIN(
            input_dim=input_dim,
            hidden_dim=args.gnn_hidden_dim,
            num_layers=args.gnn_layer
        ).to(device)
    elif args.gnn_backbone == 'GCN':
        encoder = GCN(
            input_dim=input_dim,
            hidden_dim=args.gnn_hidden_dim,
            num_layers=args.gnn_layer
        ).to(device)
    elif args.gnn_backbone == 'GAT':
        encoder = GAT(
            input_dim=input_dim,
            hidden_dim=args.gnn_hidden_dim,
            num_layers=args.gnn_layer,
            heads=args.gat_heads
        ).to(device)
    elif args.gnn_backbone == 'GraphSAGE':
        encoder = GraphSAGE(
            input_dim=input_dim,
            hidden_dim=args.gnn_hidden_dim,
            num_layers=args.gnn_layer
        ).to(device)
    else:
        raise ValueError(f"Invalid GNN backbone specified: {args.gnn_backbone}. Expected 'GIN', 'GCN', or 'GAT', or 'GraphSAGE'.")

    predictor = Classifier(
        input_dim=args.gnn_hidden_dim,
        output_dim=num_classes
    ).to(device)

    model = TargetModel(encoder=encoder, predictor=predictor, explanation_mode=args.explanation_mode).to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    epochs = args.epochs
    best_val_auc = 0.0
    best_model_state = None

    with tqdm(total=epochs, desc='Epochs') as epoch_pbar:
        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_auc = train(model, target_train_loader, optimizer, device, args)
            val_loss, val_acc, val_auc = evaluate(model, target_val_loader, device, args)

            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Train Acc': f'{train_acc:.4f}',
                'Train AUC': f'{train_auc:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Val Acc': f'{val_acc:.4f}',
                'Val AUC': f'{val_auc:.4f}'
            })
            epoch_pbar.update(1)

            # Update best model if validation AUC is higher
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict()

    # Save the best model
    save_dir_model = osp.join('model_weights', dataset_name)
    if not osp.exists(save_dir_model):
        os.makedirs(save_dir_model)
    model_save_path = osp.join(save_dir_model, 'target_gnn_model.pth')
    torch.save(best_model_state, model_save_path)
    print(f"\nBest Target GNN model saved to {model_save_path} with Val AUC: {best_val_auc:.4f}")

    # Evaluate the best model on the test set
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_auc = evaluate(model, test_loader, device, args)

    print(f"Test Accuracy of the best model: {test_acc:.4f}")
    print(f"Test AUC of the best model: {test_auc:.4f}")

    # Save results
    acc_save_path = osp.join(save_dir_model, 'target_results.pt')
    torch.save({
        'test_acc': test_acc,
        'test_auc': test_auc
    }, acc_save_path)
    print(f"Best test accuracy and AUC saved to {acc_save_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Target GNN Model')
    parser.add_argument('--dataset_root', type=str, default='dataset',
                        help='Root directory for dataset storage')
    parser.add_argument('--dataset_name', type=str, default='NCI1',
                        help='Name of the TU dataset')
    parser.add_argument('--seed', type=int, default=43,
                        help='Random seed for data loading')
    parser.add_argument('--gnn_layer', type=int, default=3,
                        help='Layer number of GNN encoder')
    parser.add_argument('--gnn_hidden_dim', type=int, default=128,
                        help='GNN encoder hidden dim')  
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer (default: 0.001)')
    parser.add_argument('--gat_heads', type=int, default=4)
    parser.add_argument('--gnn_backbone', type=str, default='GIN',
                        choices=['GIN', 'GCN', 'GAT', 'GraphSAGE'])  
    parser.add_argument('--explanation_mode', type=str, default='CAM',
                        choices=['GNNExplainer', 'PGExplainer', 'GradCAM', 'CAM', 'Grad'])  
    
    args = parser.parse_args()
    main(args)
