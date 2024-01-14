import os
import time
import math
import torch
import inspect
from model import retriever
from config import RetrieverConfig_small, RetrieverConfig_medium, RetrieverConfig_large
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import PreTrainedTokenizerFast
from dataloader import GPTDataset, RandSampler, DistRandSampler, FixCollater
from multiprocessing import Process, Barrier

def get_lr(it, config):
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def tokenizer_processor(dataset, file_index, tokenizer_path, token_dump_path, barrier, need_prepare_first_dataset):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    barrier.wait()

    if need_prepare_first_dataset:
        fnames = list(map(lambda x: x.split('/')[5].split('_')[1]+x.split('/')[6].split('.')[1].split('-')[0], dataset.files[file_index+1]))
        print(f"preparing first dataset file: {', '.join(fnames)}")
        st = time.time()
        dataset.tokenize(file_index+1, tokenizer, token_dump_path, is_batch=True)
        print(f"finished, consume: {int(time.time() - st)}s")

    for i, file in enumerate(dataset.files[file_index+1:], file_index+1):
        barrier.wait()
        dataset.tokenize(i+1, tokenizer, token_dump_path)

def evaluate(valid_dataset, model, config, device, ptdtype, barrier):
    barrier.wait()

    model.eval()
    iter_num = 0
    accum_tokens = 0
    loss_knt = 0
    loss_num = 0
    st = time.time()
    with torch.no_grad():
        for i, file in enumerate(valid_dataset.files):
            barrier.wait()
            num_origin_samples = valid_dataset.load_samples(token_dump_path)
            accum_tokens += len(valid_dataset) * config.sequence_length / 1e6
            fnames = list(map(lambda x: x.split('/')[5].split('_')[1]+x.split('/')[6].split('.')[1].split('-')[0], file))
            print(f"processing {i}: {', '.join(fnames)}, "
                f"origin: {num_origin_samples}, "
                f"samples: {len(valid_dataset)}, "
                f"accum_tokens: {int(accum_tokens)}M, "
                f"iter_num: {iter_num}")
            
            valid_sampler = RandSampler(valid_dataset, batch_size=config.batch_size, drop_last=True, shuffle=False)
            valid_loader = DataLoader(valid_dataset, num_workers=0, pin_memory=True, collate_fn=FixCollater, batch_sampler=valid_sampler)

            for j, data in enumerate(valid_loader):
                input_ids, labels = data
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(dtype=ptdtype):
                    _, loss = model(input_ids, labels)
                loss_knt += loss.item()
                loss_num += 1
                iter_num += 1
                if j + 1 == len(valid_loader):
                    print(f"step {iter_num}, loss {loss_knt/loss_num:.2f}, consume {time.time()-st:.2f}s")
                    st = time.time()
                    loss_knt = 0
                    loss_num = 0

def main(gpu, gpu_num, distributed, load_model, save_model, load_dataset, eval, config, arch, dtype, model_root, model_path, tokenizer_path, token_dump_path, data_root, data_dirs, need_prepare_first_dataset, flash=True):
    model_path = os.path.join(model_root, model_path)
    tokenizer_path = os.path.join(model_root, tokenizer_path)
    token_dump_path = os.path.join(model_root, token_dump_path)
    is_master = distributed == False or gpu == 0
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

    if load_dataset:
        checkpoint = torch.load(model_path)
        train_dataset = checkpoint['train_dataset']
        trained_file_index = checkpoint['trained_file_index']
        iter_num = checkpoint['iter_num']
        accum_tokens = checkpoint['accum_tokens']
    else:
        train_dataset = GPTDataset(data_root, data_dirs, config.sequence_length, set_name="train", shuffle=True)
        trained_file_index = -1
        iter_num = 0
        accum_tokens = 0

    if eval:
        train_dataset = GPTDataset(data_root, data_dirs, config.sequence_length, set_name="validation", shuffle=False)
        trained_file_index = -1

    if is_master:
        barrier = Barrier(2)
        tp = Process(target=tokenizer_processor, args=(train_dataset, trained_file_index, tokenizer_path, token_dump_path, barrier, need_prepare_first_dataset))
        tp.start()

    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    model = arch(device, ptdtype, config, pretrained=load_model, model_path=model_path, flash=flash)
    if distributed:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=gpu_num, rank=gpu)
    model.cuda(gpu)
    model = torch.compile(model)

    if eval:
        evaluate(train_dataset, model, config, device, ptdtype, barrier)
        exit()

    if is_master:
        print(config)
        for k, v in list(filter(lambda x: x[0][:2] != '__', inspect.getmembers(config))):
            print(f"\t{k}: {v}")
        print()
        print(model)
        print()

    optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), is_master)
    scaler = GradScaler(enabled=(dtype == 'float16'))

    if is_master:
        print("number of total parameters: %.2fM" % (model.get_num_params()/1e6,))
        print(f"all train file nums: {len(train_dataset.files)}")

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if is_master:
        barrier.wait()
        if distributed:
            dist.barrier()
    else:
        dist.barrier()

    model.train()
    accum_knt = 0
    loss_knt = 0
    loss_num = 0
    st = time.time()
    for i, file in enumerate(train_dataset.files[trained_file_index+1:], trained_file_index+1):
        if is_master:
            barrier.wait()
            if distributed:
                dist.barrier()
        else:
            dist.barrier()
        num_origin_samples = train_dataset.load_samples(token_dump_path)
        accum_tokens += len(train_dataset) * config.sequence_length / 1e6
        if is_master:
            fnames = list(map(lambda x: x.split('/')[5].split('_')[1]+x.split('/')[6].split('.')[1].split('-')[0], file))
            print(f"processing {i}: {', '.join(fnames)}, "
                f"origin: {num_origin_samples}, "
                f"samples: {len(train_dataset)}, "
                f"accum_tokens: {int(accum_tokens)}M, "
                f"iter_num: {iter_num}")
        
        if distributed:
            train_sampler = DistRandSampler(train_dataset, batch_size=config.batch_size, drop_last=True)
        else:
            train_sampler = RandSampler(train_dataset, batch_size=config.batch_size, drop_last=True)
        train_loader = DataLoader(train_dataset, num_workers=0, pin_memory=True, collate_fn=FixCollater, batch_sampler=train_sampler)

        optimizer.zero_grad(set_to_none=True)
        for j, data in enumerate(train_loader):
            lr = get_lr(iter_num, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            input_ids, labels = data
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(dtype=ptdtype):
                _, loss = model(input_ids, labels)
                loss /= config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            accum_knt += 1
            loss_knt += loss.item()
            if accum_knt % config.gradient_accumulation_steps == 0:
                loss_num += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                iter_num += 1
                if iter_num % 100 == 0:
                    if is_master:
                        print(f"step {iter_num}, loss {loss_knt/loss_num:.2f}, lr {lr:.6f}, consume {time.time()-st:.2f}s")
                    st = time.time()
                    loss_knt = 0
                    loss_num = 0
                if iter_num >= config.max_iters:
                    break
        
        if save_model and is_master:
            train_dataset.reset_samples()
            torch.save({'state_dict': model.state_dict(),
                        'train_dataset': train_dataset,
                        "trained_file_index": i,
                        'iter_num': iter_num,
                        'accum_tokens': accum_tokens,
                        }, model_path)

        if iter_num >= config.max_iters:
            break            

if __name__ == "__main__":
    # 3200个样本, 4个1080ti并行46s，1个3090模型compile之后25s, flashAtt之后是23s
    # model5M，loss会先降后增，转折点1w step左右
    # model29M, step22w, loss在5w step之后基本持平，持续略微下降, 最终loss 3.49，没有先降后增的现象
    # lr_decay_iters改成8W, loss 3.51
    # float16 和 bfloat16 训练loss没有差别
    # lr 3e-4, 8wstep, 训练到71500时, loss3.55, 比6e-4同期高了0.03
    # batch_size=32, gradient_accumulation_steps 4->8, 收敛加速，但最终仍收敛于3.49
    # scaled_dot_product_attention(is_causal=True) 72s -> 62s
    # MLP -> silu + softgate, loss 3.51->3.44
    # 改成进程预处理数据，PreTrainedTokenizerFast后面不能fork进程
    # MLP intermediate_size 2->2.68, loss 3.44->3.40
    # rotary position embedding，step 6W时 loss 3.44->3.40，预计下降0.04个点
    # untied embedding, 无收益，step 2W时 loss 3.55->3.58
    # qkv bias false->true, 无收益
    # retriever_small, batch256, step16W, file_shuffle=True, loss3.29
    # 模型选择题能力评估0.25，用mmlu fine_tune之后，mmlu_score=0.29，证明模型没问题，是缺这方面的语料，以及模型能力不行
    # </s> attention mask, pretrain的时候好像无收益
    # 不断重复以及数字概率高的问题，通过重复训练少量数据，过拟合验证，具有一字不差复述的能力, loss 0.08
    # PaLM-540B 也有输出不断重复的问题 (Instruction SFT)
    # 加入RedPajama-Data-1T的数据源(Books, ArXiv, Wikipedia, StackExchange), retriever_tv2_110M_78B_loss2.66
    # stackexchange改成只要第一个答案, 去掉无答案的，去掉"Q:...A:..."，改成"...Response:..."
    # + instruct P3, MetaMathQA, atlas_math
    # + CoT, AlpacaCoT, CausalInstructions, CommonCrawl2023
    config = RetrieverConfig_medium()
    gpu_num = config.gpu_num
    need_prepare_first_dataset = True
    load_dataset = False
    load_model = False
    save_model = True
    flash = True
    eval = False
    distributed = True if not eval else False
    arch = retriever
    dtype = "bfloat16"
    model_root = "/home/work/disk/vision/retriever"
    model_path = "checkpoint/retriever_medium.pth.tar"
    model_backup_path = "checkpoint/model_backup.pth.tar"
    tokenizer_path = "tokenizer_models/tokenizer_v2_600G.json"
    token_dump_path = "checkpoint/tokens.pkl"
    data_root = "/home/work/disk/language-data"
    data_dirs = {  # total 1024 part per dataset, each part size
        # "english_CommonCrawl2023": [1, '\n\n\n'],  # 800M
        "english_c4": [1, '\n\n\n'],  # 770M
        # "english_books": [1, '\n▁\n▁\n'],  # 100M
        # "english_arxiv": [1, '\n▁\n▁\n'],  # 90M
        "english_wiki": [1, '\n▁\n▁\n'],  # 80M
        "english_stackexchange": [1, '\n▁\n▁\n'],  # 51M
        "english_P3_decontaminated": [1, '\n▁\n▁\n'],  # 42M
        "english_CausalInstructions": [1, '\n▁\n▁\n'],  # 20M
        "english_AlpacaCoT": [1, '\n▁\n▁\n'],  # 2.3M
        "english_CoT": [1, '\n▁\n▁\n'],  # 2M
        "english_atlas_math": [1, '\n▁\n▁\n'],  # 1.2M
        "english_MetaMathQA": [1, '\n▁\n▁\n'],  # 0.3M
    }

    if load_model: # backup origin model
        os.system(f"cp -f {os.path.join(model_root, model_path)} {os.path.join(model_root, model_backup_path)}")

    if distributed:
        mp.spawn(main, nprocs=gpu_num, args=(gpu_num, distributed, load_model, save_model, load_dataset, eval, config, arch, dtype, model_root, model_path, tokenizer_path, token_dump_path, data_root, data_dirs, need_prepare_first_dataset, flash))
    else:
        main(0, gpu_num, distributed, load_model, save_model, load_dataset, eval, config, arch, dtype, model_root, model_path, tokenizer_path, token_dump_path, data_root, data_dirs, need_prepare_first_dataset, flash)
