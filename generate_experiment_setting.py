# generate_experiment_setting.py

import yaml

align_weights = [0, 0.01, 0.05, 0.07, 
                 0.1, 0.5, 0.7,
                 1, 3, 5, 7, 10]


augmentation_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

def generate_mode_name(align_weight, augmentation_ratio):
    return f"aug_{augmentation_ratio}_align_{align_weight}"

settings = []
for align_weight in align_weights:
    for augmentation_ratio in augmentation_ratios:
        settings.append({
            'mode_name': generate_mode_name(align_weight, augmentation_ratio),
            'align_weight': align_weight,
            'augmentation_ratio': augmentation_ratio
        })

with open('experiment_settings.yaml', 'w') as f:
    yaml.dump({'settings': settings}, f, default_flow_style=False)
