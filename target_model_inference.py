# target_model_inference.py

import torch
import os
import os.path as osp
from torch_geometric.data import DataLoader
from model import GIN, GCN, GAT, GraphSAGE, TargetModel, Classifier, GradCAM, CAM, GradientExplainer
from torch_geometric.explain import Explainer, GNNExplainer
import argparse
from utils import set_seed, PGExplainer


def convert_edge_scores_to_node_scores(edge_mask, edge_index, num_nodes):
    """
    Convert edge importance scores to node importance scores.
    
    Parameters:
    - edge_mask (Tensor): Edge importance scores, shape [num_edges].
    - edge_index (Tensor): Edge connections, shape [2, num_edges], indicating the two nodes connected by each edge.
    - num_nodes (int): Number of nodes in the graph.
    
    Returns:
    - node_scores (Tensor): Node importance scores, shape [num_nodes].
    """
    # Initialize node importance scores tensor
    node_scores = torch.zeros(num_nodes, device=edge_mask.device) # shape: [num_nodes]

    # Initialize node degrees
    node_degrees = torch.zeros(num_nodes, device=edge_mask.device)  # shape: [num_nodes]

    # Iterate through each edge to calculate the contribution of edge importance to node importance
    for i in range(edge_index.shape[1]):
        node1, node2 = edge_index[:, i]  # Get the two nodes connected by the edge
        importance = edge_mask[i]  # Get the importance score of the edge

        node_scores[node1] += importance
        node_scores[node2] += importance

        node_degrees[node1] += 1
        node_degrees[node2] += 1
    
    node_degrees[node_degrees == 0] = 1  # Avoid division by zero

    # Calculate node importance, normalized by node degree
    node_scores = node_scores / node_degrees

    return node_scores


def main(args):
    set_seed(args.seed)

    dataset_root = args.dataset_root
    dataset_name = args.dataset_name
    save_dir = osp.join(dataset_root, dataset_name, 'processed_splits')
    output_dir = save_dir

    # Create output directory if it does not exist
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # Define the list of dataset files to process
    dataset_files = [
        ('shadow_dataset.pt', 'queried_dataset_shadow.pt'),
        ('test_dataset.pt', 'queried_dataset_test.pt')
    ]

    # Get dataset information
    info = torch.load(osp.join(save_dir, 'dataset_info.pt'))
    input_dim = info['num_node_features']
    num_classes = info['num_classes']

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load target model
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
        raise ValueError(f"Invalid GNN backbone specified: {args.gnn_backbone}. Expected 'GIN', 'GCN', 'GraphSAGE', or 'GAT'.")

    predictor = Classifier(
        input_dim=args.gnn_hidden_dim,
        output_dim=num_classes
    ).to(device)

    model = TargetModel(encoder=encoder, predictor=predictor, explanation_mode=args.explanation_mode).to(device)

    target_model_path = osp.join('model_weights', args.dataset_name, 'target_gnn_model.pth')
    if osp.exists(target_model_path):
        model.load_state_dict(torch.load(target_model_path, map_location=device))
        print("Loaded Target GNN model.")
    else:
        raise FileNotFoundError(f"Target model not found at {target_model_path}")
    model.eval()

    if args.explanation_mode == 'GNNExplainer':
        gnnexplainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=args.gnnexplainer_epochs),
            explanation_type='model',
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='raw'
            ),
            node_mask_type='object',
            edge_mask_type=None
        )

    if args.explanation_mode == 'PGExplainer':
        pgexplainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=args.pgexplainer_epochs, lr=0.003),
            explanation_type='phenomenon',
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='raw'
            ),
            node_mask_type=None,
            edge_mask_type='object',
        )

    if args.explanation_mode == 'GradCAM':
        gradcam = GradCAM(model=model)

    if args.explanation_mode == 'CAM':
        cam = CAM(model=model)    

    if args.explanation_mode == 'Grad':
        grad = GradientExplainer(model=model)    

    # Process each dataset
    for input_file, output_file in dataset_files:
        print(f"\nProcessing {input_file}...")

        # Load dataset
        dataset_path = osp.join(save_dir, input_file)
        if not osp.exists(dataset_path):
            print(f"Warning: {dataset_path} not found, skipping...")
            continue

        current_dataset = torch.load(dataset_path)
        print(f"Dataset Size: {len(current_dataset)}")

        # Create DataLoader
        dataloader = DataLoader(current_dataset, batch_size=args.batch_size, shuffle=False)

        results = []
        total_graph_idx = 0

        # Iterate through the dataset
        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = batch_data.to(device)
            if batch_data.x is None:
                batch_data.x = torch.ones((batch_data.num_nodes, 1)).to(device)

            batch = batch_data.batch

            # Get model predictions
            with torch.no_grad():
                if args.explanation_mode in ['GNNExplainer','PGExplainer']:
                    out = model(batch_data.x, batch_data.edge_index, batch)
                else:
                    _, out = model(batch_data.x, batch_data.edge_index, batch)
            
            preds = out.argmax(dim=1)

            # Generate explanations
            if args.explanation_mode == 'GNNExplainer':
                explanations = gnnexplainer(batch_data.x, batch_data.edge_index, batch=batch)
                node_mask = explanations.node_mask.view(-1)

            if args.explanation_mode == 'PGExplainer':
                for epoch in range(args.pgexplainer_epochs):
                    pgexplainer.algorithm = pgexplainer.algorithm.to(device)
                    loss = pgexplainer.algorithm.train(epoch, model, batch_data.x, batch_data.edge_index, target=preds, batch=batch)
                explanations = pgexplainer(batch_data.x, batch_data.edge_index, target=preds, batch=batch)

                edge_mask = explanations.edge_mask
                edge_index = explanations.edge_index
                num_nodes = batch_data.x.shape[0]

                # edge score -> node score
                node_mask = convert_edge_scores_to_node_scores(edge_mask, edge_index, num_nodes)

            if args.explanation_mode == 'GradCAM':
                explanations = gradcam.get_gradcam_scores(batch_data, preds)
                node_mask = explanations

            if args.explanation_mode == 'CAM':
                explanations = cam.get_cam_scores(preds, batch)
                node_mask = explanations

            if args.explanation_mode == 'Grad':
                explanations = grad.get_gradient_scores(batch_data, preds)
                node_mask = explanations

            # Split batch data
            original_graphs = batch_data.to_data_list()
            batch_preds = preds.tolist()

            # Get the number of nodes per graph
            num_nodes_per_graph = batch_data.ptr[1:] - batch_data.ptr[:-1]
            node_masks_list = torch.split(node_mask, num_nodes_per_graph.tolist())

            # Process each graph
            for idx_in_batch, (original_data, pred, node_m) in enumerate(zip(original_graphs, batch_preds, node_masks_list)):
                # Move data back to CPU
                original_data = original_data.to('cpu')
                node_m = node_m.to('cpu')

                results.append({
                    'original_data': original_data,
                    'pred': pred,
                    'node_mask': node_m
                })
                total_graph_idx += 1

            print(f"Processed {total_graph_idx}/{len(current_dataset)} graphs.")

        output_path = osp.join(output_dir, output_file)

        torch.save(results, output_path)
        print(f"Saved processed dataset to {output_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets with target model inference')

    parser.add_argument('--dataset_root', type=str, default='dataset',
                        help='Root directory for dataset storage')
    parser.add_argument('--dataset_name', type=str, default='NCI1',
                        help='Name of the TU dataset')
    parser.add_argument('--gnnexplainer_epochs', type=int, default=100,
                        help='Number of epochs for GNNExplainer')
    parser.add_argument('--pgexplainer_epochs', type=int, default=100,
                        help='Number of epochs for PGExplainer')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda if available)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for processing')
    parser.add_argument('--gnn_hidden_dim', type=int, default=128,
                        help='GNN encoder hidden dim')  
    parser.add_argument('--gnn_layer', type=int, default=3,
                        help='Layer number of GNN encoder')
    parser.add_argument('--gat_heads', type=int, default=4)
    parser.add_argument('--gnn_backbone', type=str, default='GIN',
                        choices=['GIN', 'GCN', 'GAT', 'GraphSAGE'])  
    parser.add_argument('--explanation_mode', type=str, default='CAM',
                        choices=['GNNExplainer', 'PGExplainer', 'GradCAM', 'CAM', 'Grad'])  
    parser.add_argument('--seed', type=int, default=43,
                        help='Random seed')
    
    args = parser.parse_args()

    main(args)
