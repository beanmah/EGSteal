# sample_query_dataset.py

import torch
from torch.utils.data import random_split, Subset
import os.path as osp
import argparse
import random
from utils import set_seed



def random_sample_uniform(dataset, total_samples):
    """
    Perform uniform random sampling based on the total number of samples.
    Randomness is controlled by the global seed set via `set_seed`.

    Parameters:
    - dataset (Dataset): Original dataset
    - total_samples (int): Total number of samples to select
    - seed (int): Random seed

    Returns:
    - sampled_indices (list): List of sampled indices
    """

    if total_samples > len(dataset):
        raise ValueError("total_samples exceeds the size of the dataset.")

    sampled_indices = random.sample(range(len(dataset)), total_samples)

    return sampled_indices


def split_dataset_random(sampled_indices, train_ratio, seed=42):
    """
    Split the dataset into training and validation sets, maintaining the overall sample ratio.

    Parameters:
    - sampled_indices (list): List of sampled indices
    - train_ratio (float): Training set ratio
    - seed (int): Random seed

    Returns:
    - train_indices (list): List of training set indices
    - val_indices (list): List of validation set indices
    """

    train_size = int(len(sampled_indices) * train_ratio)
    train_indices, val_indices = random_split(sampled_indices, [train_size, len(sampled_indices) - train_size],
                                            generator=torch.Generator().manual_seed(seed))

    return list(train_indices), list(val_indices)


def sample_and_split_query_dataset(root, dataset_name, seed, sample_ratio, train_ratio):
    """
    Randomly sample from queried_dataset_shadow to generate query_dataset and split it into training and validation sets.

    Parameters:
    - root (str): Dataset root directory
    - dataset_name (str): Dataset name
    - seed (int): Random seed
    - sample_ratio (float): Overall sampling ratio
    - train_ratio (float): Training set ratio

    Returns:
    - None
    """

    # Load queried_dataset_shadow
    save_dir = osp.join(root, dataset_name, 'processed_splits')

    queried_dataset_shadow_path = osp.join(save_dir, 'queried_dataset_shadow.pt')
    queried_dataset_test_path = osp.join(save_dir, 'queried_dataset_test.pt')
    dataset_info_path = osp.join(save_dir, 'dataset_info.pt')

    queried_dataset_shadow = torch.load(queried_dataset_shadow_path)
    queried_dataset_test = torch.load(queried_dataset_test_path)
    dataset_info = torch.load(dataset_info_path)

    print(f"\nTotal number of graphs in queried_dataset_shadow: {len(queried_dataset_shadow)}")

    # Calculate total number of samples
    total_samples = int(len(queried_dataset_shadow) * sample_ratio)
    if total_samples == 0:
        raise ValueError("Sample ratio too low, resulting in zero samples.")

    print(f"Total samples: {total_samples}")

    # Random uniform sampling
    sampled_indices = random_sample_uniform(
        dataset=queried_dataset_shadow,
        total_samples=total_samples
    )

    print(f"Number of sampled samples: {len(sampled_indices)}")

    # Split training and validation sets, maintaining overall sample ratio
    train_indices, val_indices = split_dataset_random(
        sampled_indices=sampled_indices,
        train_ratio=train_ratio,
        seed=seed
    )

    print(f"\nTraining set:")
    print(f"Total training samples: {len(train_indices)}")

    print(f"\nValidation set:")
    print(f"Total validation samples: {len(val_indices)}")

    # Create Subset datasets
    sampled_queried_dataset_train = Subset(queried_dataset_shadow, train_indices)
    sampled_queried_dataset_val = Subset(queried_dataset_shadow, val_indices)

    # Save results
    query_train_path = osp.join(save_dir, 'queried_dataset_train.pt')
    query_val_path = osp.join(save_dir, 'queried_dataset_val.pt')
    query_test_path = osp.join(save_dir, 'queried_dataset_test.pt')
    data_info_path = osp.join(save_dir, 'dataset_info.pt')

    torch.save(sampled_queried_dataset_train, query_train_path)
    torch.save(sampled_queried_dataset_val, query_val_path)
    torch.save(queried_dataset_test, query_test_path)
    torch.save(dataset_info, data_info_path)

    print(f"\nQuery dataset split saved:")
    print(f"- Training set ({len(sampled_queried_dataset_train)} samples) saved to: {query_train_path}")
    print(f"- Validation set ({len(sampled_queried_dataset_val)} samples) saved to: {query_val_path}")
    print(f"Total query dataset size: {len(sampled_queried_dataset_train) + len(sampled_queried_dataset_val)}")
    print(f"Split ratio (train/val): {train_ratio}/{1 - train_ratio}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample and Split Query Dataset with Random Sampling')

    parser.add_argument('--dataset_root', type=str, default='dataset',
                        help='Root directory for dataset storage')
    parser.add_argument('--dataset_name', type=str, default='NCI1',
                        help='Name of the TU dataset')
    parser.add_argument('--seed', type=int, default=43,
                        help='Random seed for data splitting')
    parser.add_argument('--sample_ratio', type=float, default=0.3,
                        help='Ratio of data to sample from queried_dataset_shadow')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training set in query dataset split')

    args = parser.parse_args()

    # Validate train_ratio parameter
    if not 0 < args.train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    # Validate sample_ratio parameter
    if not 0 < args.sample_ratio <= 1:
        raise ValueError("sample_ratio must be between 0 (exclusive) and 1 (inclusive)")

    set_seed(args.seed)
    sample_and_split_query_dataset(
        root=args.dataset_root,
        dataset_name=args.dataset_name,
        seed=args.seed,
        sample_ratio=args.sample_ratio,
        train_ratio=args.train_ratio
    )
