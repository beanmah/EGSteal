# run_experiments.py

import os
import os.path as osp
import argparse
import subprocess
import numpy as np
import torch
import json
import yaml
from datetime import datetime
import time


def save_experiment_params(args, timestamp, dataset_name):

    params_dir = 'experiment_params'
    if not osp.exists(params_dir):
        os.makedirs(params_dir)

    params_filename = f'{dataset_name}_params_{timestamp}.json'
    params_file = osp.join(params_dir, params_filename)

    params_dict = vars(args)

    params_dict['experiment_timestamp'] = timestamp
    params_dict['experiment_datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(params_file, 'w') as f:
        json.dump(params_dict, f, indent=4)
    
    return params_file

def load_surrogate_settings(yaml_path):
    """Load surrogate model training settings"""
    if not osp.exists(yaml_path):
        print(f"Warning: Configuration file {yaml_path} not found, using default parameters")
        return [{"mode_name": "default"}]
        
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('settings', [{"mode_name": "default"}])


def main():
    parser = argparse.ArgumentParser(description='Run Experiments')
    parser.add_argument('--dataset_name', type=str, default='NCI1',   # AIDS  NCI1 NCI109  Mutagenicity
                        help='Name of the dataset')
    parser.add_argument('--dataset_root', type=str, default='dataset',
                        help='Root directory for dataset storage')
    parser.add_argument('--surrogate_epochs', type=int, default=200,
                        help='Number of epochs for surrogate training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for surrogate training')
    parser.add_argument('--target_ratio', type=float, default=0.4,
                        help='Ratio of the dataset for the target task (train + val)')
    parser.add_argument('--target_val_ratio', type=float, default=0.2,
                        help='Ratio of the target dataset to be used for validation')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Ratio of test set')
    parser.add_argument('--shadow_ratio', type=float, default=0.4,
                        help='Ratio of shadow dataset')
    parser.add_argument('--query_sample_ratio', type=float, nargs='+',
                        default=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help='List of query sample size')
    parser.add_argument('--target_epochs', type=int, default=200,
                        help='Number of epochs for target model training')
    parser.add_argument('--target_learning_rate', type=float, default=0.001,
                        help='Learning rate for target model training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--data_seed', type=int, default=43,
                        help='seed for each data split')
    parser.add_argument('--seeds', type=int, nargs='+', default=[41,42,43,44,45],
                        help='List of random seeds for each run')
    parser.add_argument('--gnnexplainer_epochs', type=int, default=100,
                        help='Number of epochs for GNNExplainer (default: 100)')
    parser.add_argument('--node_topk_percent', type=int, default=0.5,
                        help='Percentage of top nodes to select based on node mask')

    parser.add_argument('--augmentation_ratio', type=float, default=0.0,
                        help='Default ratio for data augmentation')
    parser.add_argument('--align_weight', type=float, default=0.0,
                        help='Default weight for alignment loss')

    parser.add_argument('--target_model_gnn_backbone', type=str, default='GCN',
                        choices=['GIN', 'GCN', 'GAT', 'GraphSAGE']) 
    parser.add_argument('--surrogate_model_gnn_backbone', type=str, default='GIN',
                        choices=['GIN', 'GCN', 'GAT', 'GraphSAGE']) 

    parser.add_argument('--target_model_gnn_layer', type=int, default=3,
                        help='Layer number of GNN encoder')
    parser.add_argument('--surrogate_model_gnn_layer', type=int, default=3,
                        help='Layer number of GNN encoder')

    parser.add_argument('--target_model_gnn_hidden_dim', type=int, default=128,
                        help='GNN encoder hidden dim')  
    parser.add_argument('--surrogate_model_gnn_hidden_dim', type=int, default=128,
                        help='GNN encoder hidden dim')  

    parser.add_argument('--explanation_mode', type=str, default='CAM',
                        choices=['GNNExplainer', 'PGExplainer', 'GradCAM', 'CAM', 'Grad'])  

    parser.add_argument('--gpu_id', type=str, default='0', help='GPU device number (eg. "0" or "1,2")')

    parser.add_argument('--surrogate_settings', type=str, default='experiment_settings.yaml',
                        help='YAML file containing surrogate model training settings')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    params_file = save_experiment_params(args, timestamp, args.dataset_name)
    print(f"Experiment parameters saved to: {params_file}")
    

    results_filename = f'{args.dataset_name}_results_{timestamp}.json'
    results_file = osp.join('results', results_filename)
    
    # Load surrogate training settings
    surrogate_settings = load_surrogate_settings(args.surrogate_settings)
    
    all_results = {}

    # Step 1: Data preparation (execute only once)
    print("=== Data Preparation ===")
    data_prep_cmd = [
        'python', 'data_preparation.py',
        '--dataset_name', args.dataset_name,
        '--root', args.dataset_root,
        '--seed', str(args.data_seed),
        '--target_ratio', str(args.target_ratio),
        '--target_val_ratio', str(args.target_val_ratio),
        '--test_ratio', str(args.test_ratio),
        '--shadow_ratio', str(args.shadow_ratio),
    ]
    subprocess.run(data_prep_cmd)

    # Step 2: Train target model (execute only once)
    print("=== Training Target Model ===")
    train_target_cmd = [
        'python', 'train_target_model.py',
        '--dataset_root', args.dataset_root,
        '--dataset_name', args.dataset_name,
        '--seed', str(args.data_seed),
        '--epochs', str(args.target_epochs),
        '--learning_rate', str(args.target_learning_rate),
        '--gnn_backbone', args.target_model_gnn_backbone,
        '--gnn_layer', str(args.target_model_gnn_layer),
        '--gnn_hidden_dim', str(args.target_model_gnn_hidden_dim),
        '--explanation_mode', args.explanation_mode
    ]
    subprocess.run(train_target_cmd)

    # Load target model results
    save_dir = 'model_weights'
    target_results_path = osp.join(save_dir, args.dataset_name, 'target_results.pt')
    if osp.exists(target_results_path):
        target_results = torch.load(target_results_path)
        target_test_acc = float(target_results.get('test_acc', 0.0))
        target_test_auc = float(target_results.get('test_auc', 0.0))
        print(f"Loaded target model test accuracy: {target_test_acc:.4f}")
        print(f"Loaded target model test AUC: {target_test_auc:.4f}")
    else:
        print(f"Target model results file not found: {target_results_path}")
        target_test_acc = None
        target_test_auc = None

    # Iterate through query_sample_ratio list
    for sample_ratio in args.query_sample_ratio:
        print(f"\n=== query_sample_ratio = {sample_ratio} ===")
        
        # Initialize results storage for current ratio
        results = {
            'train_modes': {},  # For storing results of different modes
            'target': {'test_acc': [], 'test_auc': []}
        }

        # Add target model results
        if target_test_acc is not None and target_test_auc is not None:
            results['target']['test_acc'].append(target_test_acc)
            results['target']['test_auc'].append(target_test_auc)

        for idx, seed in enumerate(args.seeds):
            print(f"\n=== Experiment {idx + 1}, Random Seed {seed} ===")

            # target_model inference
            print("=== Using target model to generate query dataset ===")
            inference_cmd = [
                'python', 'target_model_inference.py',
                '--dataset_root', args.dataset_root,
                '--dataset_name', args.dataset_name,
                '--seed', str(seed),
                '--gnnexplainer_epochs', str(args.gnnexplainer_epochs),
                '--node_topk_percent', str(args.node_topk_percent),
                '--gnn_backbone', str(args.target_model_gnn_backbone),
                '--gnn_layer', str(args.target_model_gnn_layer),
                '--gnn_hidden_dim', str(args.target_model_gnn_hidden_dim),
                '--explanation_mode', str(args.explanation_mode)
            ]
            subprocess.run(inference_cmd)

            # Sample surrogate_dataset.pt
            print("Sample query_dataset.pt")
            sample_cmd = [
                'python', 'sample_query_dataset.py',
                '--dataset_root', args.dataset_root,
                '--dataset_name', args.dataset_name,
                '--seed', str(seed),
                '--sample_ratio', str(sample_ratio)
            ]
            subprocess.run(sample_cmd)

            for setting in surrogate_settings:
                mode_name = setting.get('mode_name', 'default')
                print(f"\n=== Training surrogate model: {mode_name} ===")
                
                finetune_cmd = [
                    'python', 'train_surrogate_model.py',
                    '--dataset_root', args.dataset_root,
                    '--dataset_name', args.dataset_name,
                    '--seed', str(seed),
                    '--epochs', str(setting.get('epochs', args.surrogate_epochs)),
                    '--learning_rate', str(setting.get('learning_rate', args.learning_rate)),
                    '--augmentation_ratio', str(setting.get('augmentation_ratio', args.augmentation_ratio)),
                    '--gnn_backbone', str(setting.get('gnn_backbone', args.surrogate_model_gnn_backbone)),
                    '--gnn_layer', str(setting.get('gnn_layer', args.surrogate_model_gnn_layer)),
                    '--gnn_hidden_dim', str(setting.get('gnn_hidden_dim', args.surrogate_model_gnn_hidden_dim)),
                    '--align_weight', str(setting.get('align_weight', args.align_weight))
                ]

                print("finetune_cmd: ", finetune_cmd)
                subprocess.run(finetune_cmd)

                # Initialize result storage for current mode
                if mode_name not in results['train_modes']:
                    results['train_modes'][mode_name] = {
                        'test_acc': [], 'test_auc': [], 'fidelity_score': [],
                        'order_accuracy': [], 'rank_correlation': []
                    }
                
                # Load results
                results_path = osp.join('model_weights', args.dataset_name, 'surrogate_results.pt')
                if osp.exists(results_path):
                    model_results = torch.load(results_path)
                    results['train_modes'][mode_name]['test_acc'].append(model_results.get('test_acc', 0.0))
                    results['train_modes'][mode_name]['test_auc'].append(model_results.get('test_auc', 0.0))
                    results['train_modes'][mode_name]['fidelity_score'].append(model_results.get('fidelity_score', 0.0))
                    results['train_modes'][mode_name]['order_accuracy'].append(model_results.get('order_accuracy', 0.0))
                    results['train_modes'][mode_name]['rank_correlation'].append(model_results.get('rank_correlation', 0.0))
                else:
                    print(f"    Results file not found: {results_path}")

        # Calculate statistics for current ratio
        summary = {}
        print(f"\n=== Results Summary for query_sample_ratio = {sample_ratio:.4f} ===")
        
        # Process results for each training mode
        for mode_name, mode_results in results['train_modes'].items():
            summary[mode_name] = {
                'test_acc': {
                    'mean': np.mean(mode_results['test_acc']) if mode_results['test_acc'] else None,
                    'std': np.std(mode_results['test_acc']) if mode_results['test_acc'] else None,
                    'all_values': mode_results['test_acc']
                },
                'test_auc': {
                    'mean': np.mean(mode_results['test_auc']) if mode_results['test_auc'] else None,
                    'std': np.std(mode_results['test_auc']) if mode_results['test_auc'] else None,
                    'all_values': mode_results['test_auc']
                },
                'fidelity_score': {
                    'mean': np.mean(mode_results['fidelity_score']) if mode_results['fidelity_score'] else None,
                    'std': np.std(mode_results['fidelity_score']) if mode_results['fidelity_score'] else None,
                    'all_values': mode_results['fidelity_score']
                },
                'order_accuracy': {
                    'mean': np.mean(mode_results['order_accuracy']) if mode_results['order_accuracy'] else None,
                    'std': np.std(mode_results['order_accuracy']) if mode_results['order_accuracy'] else None,
                    'all_values': mode_results['order_accuracy']
                },
                'rank_correlation': {
                    'mean': np.mean(mode_results['rank_correlation']) if mode_results['rank_correlation'] else None,
                    'std': np.std(mode_results['rank_correlation']) if mode_results['rank_correlation'] else None,
                    'all_values': mode_results['rank_correlation']
                }
            }
            
            # Add training parameters used for this mode
            for setting in surrogate_settings:
                if setting.get('mode_name') == mode_name:
                    summary[mode_name]['training_params_from_yaml'] = setting
                    break

        # Process target results
        if results['target']['test_acc'] and results['target']['test_auc']:
            summary['target'] = {
                'test_acc': {
                    'mean': np.mean(results['target']['test_acc']),
                    'std': np.std(results['target']['test_acc']),
                    'all_values': results['target']['test_acc']
                },
                'test_auc': {
                    'mean': np.mean(results['target']['test_auc']),
                    'std': np.std(results['target']['test_auc']),
                    'all_values': results['target']['test_auc']
                }
            }
        else:
            print("Target model: No valid results")

        # Add data split ratios to summary
        summary['data_split_ratios'] = {
            'target_ratio': args.target_ratio,
            'target_val_ratio': args.target_val_ratio,
            'test_ratio': args.test_ratio,
            'shadow_ratio': args.shadow_ratio,
            'query_sample_ratio': sample_ratio,
        }
        
        # Add current ratio's summary to all_results
        all_results[f'query_sample_ratio_{sample_ratio}'] = summary

        # Save current all_results to JSON file
        if not osp.exists('results'):
            os.makedirs('results')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"Current results saved to {results_file}")

    print(f"\nAll experiment results saved to {results_file}")

if __name__ == '__main__':
    start_time = time.time()
    try:
        print(f"Starting experiments, time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        main()
        end_time = time.time()
        total_time = end_time - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        print(f"\nExperiment completed!")
        print(f"Total runtime: {hours} hours {minutes} minutes {seconds:.2f} seconds")
    except Exception as e:
        error_msg = f"\nProgram execution error!\nError time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nError message: {str(e)}\n"
        print(error_msg)
        raise
