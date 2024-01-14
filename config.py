from dataclasses import dataclass

@dataclass
class RetrieverConfig_small:
    gpu_num = 3
    batch_size = 32
    gradient_accumulation_steps = 3
    sequence_length = 1024
    learning_rate = 6e-4
    min_lr = 6e-5
    vocab_size = 32000
    num_layers = 6
    hidden_size = 512
    num_heads = 8
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 1e-1
    warmup_iters = 2000
    max_iters = 160000
    lr_decay_iters = 160000
    grad_clip = 1.0

@dataclass
class RetrieverConfig_medium:
    gpu_num = 3
    batch_size = 16
    gradient_accumulation_steps = 8
    sequence_length = 1024
    learning_rate = 6e-4
    min_lr = 6e-5
    vocab_size = 32000
    num_layers = 12
    hidden_size = 768
    num_heads = 12
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 1e-1
    warmup_iters = 2000
    max_iters = 200000
    lr_decay_iters = 200000
    grad_clip = 1.0

@dataclass
class RetrieverConfig_medium_finetune:
    batch_size = 16
    gradient_accumulation_steps = 8
    learning_rate = 1e-4
    min_lr = 1e-5
    max_iters = 20000
    lr_decay_iters = 20000

@dataclass
class RetrieverConfig_large:
    gpu_num = 3
    batch_size = 8
    gradient_accumulation_steps = 20
    sequence_length = 1024
    learning_rate = 6e-4
    min_lr = 6e-5
    vocab_size = 32000
    num_layers = 18
    hidden_size = 1280
    num_heads = 16
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 1e-1
    warmup_iters = 2000
    max_iters = 200000
    lr_decay_iters = 200000
    grad_clip = 1.0