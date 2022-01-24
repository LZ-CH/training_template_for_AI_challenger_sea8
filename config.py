args_resnet = {
    'epochs': 200,
    # 'optimizer_name': 'SGD',
    # 'optimizer_hyperparameters': {
    #     'lr': 0.0001,
    #     'momentum': 0.9,
    #     'weight_decay': 1e-4
    # },
    # 'scheduler_name': 'CosineAnnealingLR',
    'optimizer_name': 'Adam',
    'optimizer_hyperparameters': {
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'weight_decay': 1e-4
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 200
    },
    'batch_size': 256,
}
args_densenet = {
    'epochs': 200,
    'optimizer_name': 'Adam',
    'optimizer_hyperparameters': {
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'weight_decay': 1e-4
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 200
    },
    'batch_size': 256,
}