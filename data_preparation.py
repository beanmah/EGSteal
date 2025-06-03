# data_preparation.py

import torch
from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split
from torch_geometric.transforms import Constant
import os
import os.path as osp
import argparse
from utils import set_seed

def prepare_data(root='dataset',
                 dataset_name='Mutagenicity',
                 seed=42,
                 target_ratio=0.4,
                 target_val_ratio=0.2,
                 test_ratio=0.2,
                 shadow_ratio=0.4):
    """
    Prepare the dataset, splitting it into four subsets:
    1. target_train_dataset
    2. target_val_dataset
    3. test_dataset
    4. shadow_dataset
    If nodes lack features, use pre_transform to set default features for all nodes.
    """

    dataset_source_map = {
        'Mutagenicity': 'TU',
        'NCI1': 'TU',
        'NCI109': 'TU',
        'AIDS': 'TU'
    }

    # Check if the dataset name is supported
    if dataset_name not in dataset_source_map:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    # Check if the split ratios sum to 1
    total_ratio = target_ratio + test_ratio + shadow_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError(f"The sum of all dataset split ratios must equal 1, current sum is {total_ratio}")

    # Set random seed
    set_seed(seed)

    # Load TUDataset
    temp_dataset = TUDataset(root=root, name=dataset_name)

    # Check if node features need to be added
    data_transform = None
    if temp_dataset.num_node_features == 0:
        print("\nNo node features found. Adding constant node features (all ones).")
        data_transform = Constant(value=1, cat=False)

    dataset = TUDataset(root=root, name=dataset_name, transform=data_transform)

    num_graphs = len(dataset)
    print(f"\nTotal number of graphs in {dataset_name}: {num_graphs}")
    print(f"Node features dimension: {dataset.num_node_features}")
    print(f"Edge features dimension: {dataset.num_edge_features if hasattr(dataset, 'num_edge_features') else 'N/A'}")
    print(f"Number of classes: {dataset.num_classes if hasattr(dataset, 'num_classes') else 'N/A'}")

    # Dataset splitting
    target_num = int(num_graphs * target_ratio)
    test_num = int(num_graphs * test_ratio)
    shadow_num = num_graphs - target_num - test_num  # Ensure total consistency

    # Randomly split dataset
    target_dataset, test_dataset, shadow_dataset = random_split(
        dataset,
        [target_num, test_num, shadow_num]
    )

    # Further split target_dataset into train and val
    target_train_num = int(target_num * (1 - target_val_ratio))
    target_val_num = target_num - target_train_num

    target_train_dataset, target_val_dataset = random_split(
        target_dataset,
        [target_train_num, target_val_num]
    )

    # Print dataset split sizes
    print("\nDataset split sizes:")
    print(f"Target train set size: {len(target_train_dataset)} ({len(target_train_dataset) / num_graphs:.1%})")
    print(f"Target val set size: {len(target_val_dataset)} ({len(target_val_dataset) / num_graphs:.1%})")
    print(f"Test set size: {len(test_dataset)} ({len(test_dataset) / num_graphs:.1%})")
    print(f"Shadow dataset size: {len(shadow_dataset)} ({len(shadow_dataset) / num_graphs:.1%})")

    info = {
        'num_graphs': num_graphs,
        'num_node_features': dataset.num_node_features,
        'num_edge_features': dataset.num_edge_features if hasattr(dataset, 'num_edge_features') else None,
        'num_classes': dataset.num_classes if hasattr(dataset, 'num_classes') else None,
        'split_ratios': {
            'target_train': (1 - target_val_ratio) * target_ratio,
            'target_val': target_val_ratio * target_ratio,
            'test': test_ratio,
            'shadow': shadow_ratio
        },
        'seed': seed
    }

    # Create save directory
    save_dir = osp.join(root, dataset_name, 'processed_splits')
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    # Save dataset info
    torch.save(info, osp.join(save_dir, 'dataset_info.pt'))

    # Save all subsets
    splits_data = {
        'target_train_dataset.pt': target_train_dataset,
        'target_val_dataset.pt': target_val_dataset,
        'test_dataset.pt': test_dataset,
        'shadow_dataset.pt': shadow_dataset
    }

    for filename, data in splits_data.items():
        torch.save(data, osp.join(save_dir, filename))

    print(f"\nDatasets saved in {save_dir}")
    print("Saved files:")
    print("- dataset_info.pt")
    for filename in splits_data.keys():
        print(f"- {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset splits')
    parser.add_argument('--root', type=str, default='dataset',
                        help='Root directory for dataset storage')
    parser.add_argument('--dataset_name', type=str, default='Mutagenicity',
                        choices=['Mutagenicity', 'NCI1', 'NCI109', 'AIDS'],
                        help='Name of the dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting')
    parser.add_argument('--target_ratio', type=float, default=0.4,
                        help='Ratio of the dataset for the target task (train + val)')
    parser.add_argument('--target_val_ratio', type=float, default=0.2,
                        help='Ratio of the target dataset to be used for validation')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Ratio of test set')
    parser.add_argument('--shadow_ratio', type=float, default=0.4,
                        help='Ratio of shadow dataset')

    args = parser.parse_args()

    prepare_data(
        root=args.root,
        dataset_name=args.dataset_name,
        seed=args.seed,
        target_ratio=args.target_ratio,
        target_val_ratio=args.target_val_ratio,
        test_ratio=args.test_ratio,
        shadow_ratio=args.shadow_ratio
    )
