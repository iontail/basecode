from easydict import EasyDict as edict
from arguments import parse_arguments, print_args

cfg = edict()

def get_config(name):
    cfg.dataset_configs = edict({
        'cifar10': edict(num_classes=10, img_size=32,
                        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        'cifar100': edict(num_classes=100, img_size=32,
                        mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        'imagenet': edict(num_classes=1000, img_size=224,
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    })
    
    return cfg.dataset_configs.get(name, None)

def get_dataset_config(name, args):
    return get_config(name)

def get_optimizer_config(name, args):
    cfg.optimizer_configs = edict({
        'AdamW': edict(lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)),
        'SGD': edict(lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay),
        'Adam': edict(lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)),
    })
    return cfg.optimizer_configs.get(name, None)

def get_scheduler_config(name, args):
    cfg.scheduler_configs = edict({
        'cosine': edict(T_max=100, eta_min=1e-6),
        'step': edict(step_size=30, gamma=0.1),
        'plateau': edict(patience=10, factor=0.1),
    })
    return cfg.scheduler_configs.get(name, None)

def get_augmentation_config(name, args):
    cfg.augmentation_configs = edict({
        'basic': edict(random_crop=args.random_crop, random_flip=args.random_flip, normalize=args.normalize),
        'advanced': edict(
            random_crop=args.random_crop,
            random_flip=args.random_flip,
            auto_augment=args.auto_augment,
            mixup=args.mixup,
            cutmix=args.cutmix,
            normalize=args.normalize,
            mixup_alpha=args.mixup_alpha,
        )
    })
    return cfg.augmentation_configs.get(name, None)

if __name__ == "__main__":
    print("Dataset Config for CIFAR-10:", get_config('cifar10'))
    print("Optimizer Config for AdamW:", get_optimizer_config('AdamW'))
    print("Scheduler Config for cosine:", get_scheduler_config('cosine'))
    print("Augmentation Config for basic:", get_augmentation_config('basic'))