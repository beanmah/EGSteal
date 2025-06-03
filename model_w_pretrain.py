# model_w_pretrain.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing

class GINConv(MessagePassing):
    def __init__(self, mlp, emb_dim, train_eps=False):
        super(GINConv, self).__init__(aggr='add')
        self.mlp = mlp
        self.eps = nn.Parameter(torch.Tensor([0])) if train_eps else 0
        self.edge_embedding1 = nn.Embedding(6, emb_dim)
        self.edge_embedding2 = nn.Embedding(3, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_index.dtype != torch.long:
            edge_index = edge_index.to(torch.long)

        edge_embedding1 = self.edge_embedding1(edge_attr[:, 0])
        edge_embedding2 = self.edge_embedding2(edge_attr[:, 1])
        edge_embeddings = edge_embedding1 + edge_embedding2

        out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
        out = self.mlp(out + (1 + self.eps) * x)
        return out

    def message(self, x_j, edge_attr):
        return x_j + edge_attr


class GIN(nn.Module):
    def __init__(self, num_layers=5, emb_dim=300):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            # MLP for each layer: Linear(emb_dim, 2*emb_dim) -> ReLU -> Linear(2*emb_dim, emb_dim)
            nn_seq = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim),
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim)
            )
            self.convs.append(GINConv(nn_seq, emb_dim, train_eps=False))
            self.bns.append(nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(h, edge_index, edge_attr)
            h = bn(h)
            h = F.relu(h)

        node_embeddings = h
        graph_embeddings = global_mean_pool(node_embeddings, batch)

        return node_embeddings, graph_embeddings


class GNN_graphpred(nn.Module):
    def __init__(self, num_layers=5, emb_dim=300, num_tasks=1):
        super(GNN_graphpred, self).__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        self.x_embedding1 = nn.Embedding(120, emb_dim)
        self.x_embedding2 = nn.Embedding(3, emb_dim)

        self.gin = GIN(
            num_layers=num_layers,
            emb_dim=emb_dim
        )

        self.graph_pred_linear = nn.Linear(emb_dim, num_tasks)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        node_embeddings, graph_embeddings = self.gin(h, edge_index, edge_attr, batch)

        return self.graph_pred_linear(graph_embeddings)

    def from_pretrained(self, model_file):
        print(f"Loading weights from: {model_file}")
        state_dict = torch.load(model_file, map_location='cpu', weights_only=True)
        print(f"Loaded {len(state_dict)} weights")

        # Adjust weight keys to match new structure
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('gnns.'):
                new_key = k.replace('gnns.', 'gin.convs.')
            elif k.startswith('batch_norms.'):
                new_key = k.replace('batch_norms.', 'gin.bns.')
            else:
                new_key = k
            new_state_dict[new_key] = v

        try:
            missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
            print("Weights loaded successfully")
        except Exception as e:
            print(f"Error loading weights: {e}")
        return self


class CAM_GNNGraphPred:
    def __init__(self, model):
        """
        CAM implementation for the GNN_graphpred model.

        Parameters:
        - model: GNN_graphpred model instance
        """
        self.model = model
        self.activations = None
        self.classifier_weights = None
        self._register_hooks()

    def _register_hooks(self):
        """
        Register forward hook to capture activations from the last convolutional layer
        and extract classifier weights.
        """
        def forward_hook(module, input, output):
            self.activations = output

        if not isinstance(self.model, nn.Module) or not hasattr(self.model, 'gin'):
            raise ValueError("Model must be an instance of GNN_graphpred with a 'gin' attribute.")

        # Extract classifier weights
        classifier = self.model.graph_pred_linear
        if not isinstance(classifier, nn.Linear):
            raise ValueError("Classifier (graph_pred_linear) must be an instance of nn.Linear.")
        self.classifier_weights = classifier.weight  # [num_tasks, emb_dim]

        # Register hook to the last convolutional layer
        last_conv = self.model.gin.convs[-1]
        last_conv.register_forward_hook(forward_hook)

    def get_cam_scores(self, target_classes, batch_ids):
        """
        Generate CAM scores based on captured activations and classifier weights.

        Parameters:
        - target_classes: [batch_size] tensor, each element is the class index (0 or 1)
        - batch_ids: [total_num_nodes] tensor, indicating the graph index for each node

        Returns:
        - cam_scores: Tensor, CAM scores for each node [total_num_nodes]
        """
        if self.activations is None:
            raise ValueError("No activations recorded. Ensure a forward pass has been done before generating CAM.")

        cam_scores = []
        num_graphs = batch_ids.max().item() + 1

        for graph_id in range(num_graphs):
            weight = self.classifier_weights[0]  # [1, emb_dim] -> [emb_dim]

            node_indices = (batch_ids == graph_id).nonzero(as_tuple=False).squeeze(-1)
            if node_indices.numel() == 0:
                cam = torch.tensor([], device=self.activations.device)
            else:
                activation = self.activations[node_indices]  # [num_nodes_in_graph, emb_dim]
                cam = torch.matmul(activation, weight)  # [num_nodes_in_graph]

            cam_scores.append(cam)

        cam_scores = torch.cat(cam_scores, dim=0)

        return cam_scores
