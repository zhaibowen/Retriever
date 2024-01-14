import os
import time
import math
import torch
import inspect
from model import retriever
from config import RetrieverConfig_small, RetrieverConfig_medium, RetrieverConfig_large, RetrieverConfig_medium_finetune
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import PreTrainedTokenizerFast
from dataloader import GPTDataset, RandSampler, DistRandSampler, MaskCollater
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
        fnames = list(map(lambda x: x.split('/')[5].split('_')[2]+x.split('/')[6].split('.')[1].split('-')[0], dataset.files[file_index+1]))
        print(f"preparing first dataset file: {', '.join(fnames)}")
        st = time.time()
        dataset.tokenize_with_packing(file_index+1, tokenizer, token_dump_path, is_batch=True)
        print(f"finished, consume: {int(time.time() - st)}s")

    for i, file in enumerate(dataset.files[file_index+1:], file_index+1):
        barrier.wait()
        dataset.tokenize_with_packing(i+1, tokenizer, token_dump_path, is_batch=False)

def main(gpu, gpu_num, distributed, load_model, save_model, load_dataset, config, arch, dtype, model_root, pretrain_model_path, model_path, tokenizer_path, token_dump_path, data_root, data_dirs, need_prepare_first_dataset, flash=True):
    model_path = os.path.join(model_root, model_path)
    if load_model:
        pretrain_model_path = model_path
    else:
        pretrain_model_path = os.path.join(model_root, pretrain_model_path)
    tokenizer_path = os.path.join(model_root, tokenizer_path)
    token_dump_path = os.path.join(model_root, token_dump_path)
    is_master = distributed == False or gpu == 0
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

    if load_dataset:
        checkpoint = torch.load(pretrain_model_path)
        train_dataset = checkpoint['train_dataset']
        trained_file_index = checkpoint['trained_file_index']
        iter_num = checkpoint['iter_num']
        accum_tokens = checkpoint['accum_tokens']
    else:
        train_dataset = GPTDataset(data_root, data_dirs, config.sequence_length, set_name="train", shuffle=True, mask_flag=True)
        trained_file_index = -1
        iter_num = 0
        accum_tokens = 0

    if is_master:
        barrier = Barrier(2)
        tp = Process(target=tokenizer_processor, args=(train_dataset, trained_file_index, tokenizer_path, token_dump_path, barrier, need_prepare_first_dataset))
        tp.start()

    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    model = arch(device, ptdtype, config, pretrained=True, model_path=pretrain_model_path, flash=flash)
    if distributed:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=gpu_num, rank=gpu)
    model.cuda(gpu)
    model = torch.compile(model)

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
            fnames = list(map(lambda x: x.split('/')[5].split('_')[2]+x.split('/')[6].split('.')[1].split('-')[0], file))
            print(f"processing {i}: {', '.join(fnames)}, "
                f"origin: {num_origin_samples}, "
                f"samples: {len(train_dataset)}, "
                f"accum_tokens: {int(accum_tokens)}M, "
                f"iter_num: {iter_num}")
        
        if distributed:
            train_sampler = DistRandSampler(train_dataset, batch_size=config.batch_size, drop_last=True)
        else:
            train_sampler = RandSampler(train_dataset, batch_size=config.batch_size, drop_last=True)
        train_loader = DataLoader(train_dataset, num_workers=2, pin_memory=True, collate_fn=MaskCollater, batch_sampler=train_sampler)

        optimizer.zero_grad(set_to_none=True)
        for j, data in enumerate(train_loader):
            lr = get_lr(iter_num, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            input_ids, labels, masks = data
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks = masks.to(ptdtype).to(device, non_blocking=True)
            with autocast(dtype=ptdtype):
                _, loss = model(input_ids, labels, masks)
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
    # 使用RedPajama-Data-Instruct P3，回答的answer太短了，而且评估指标没有提升，放到pretrain里面
    # 使用mmlu, STEM: 0.29, other (business, health, misc.): 0.31, social sciences: 0.30, humanities: 0.31
    # 使用alpaca，样本量太少，评估指标没有提升, 但是问答的质量明显好了很多
    # 使用NaturalQuestions/triviaQa，通识问答和阅读理解能力有所提升
    # + CoT, MathInstruct, AlpacaCoT(基本是中文), CausalInstructions
    # 只在mmlu train上finetune, mmlu eval 0.3, race 0.4.
    # 在instruction为主的数据集finetune，mmlu eval 0.25, race 0.27，mmlu样本太少，且label太少，被稀释了
    config = RetrieverConfig_medium()
    finetune_config = RetrieverConfig_medium_finetune()
    revised_params = list(filter(lambda x: x[0][0] != '_', inspect.getmembers(finetune_config)))
    for rp, value in revised_params:
        setattr(config, rp, value)
    gpu_num = config.gpu_num
    need_prepare_first_dataset = False
    load_dataset = False
    load_model = False # False: use pre-train model,  True: use finetune model
    save_model = True
    flash = True
    distributed = True
    arch = retriever
    dtype = "bfloat16"
    model_root = "/home/work/disk/vision/retriever"
    pretrain_model_path = "checkpoint/retriever_tv2_110M_78B_loss2.66.pth.tar"
    model_path = "checkpoint/instruct_retriever_medium.pth.tar"
    model_backup_path = "checkpoint/instruct_model_backup.pth.tar"
    tokenizer_path = "tokenizer_models/tokenizer_v2_600G.json"
    token_dump_path = "checkpoint/instruct_tokens.pkl"
    data_root = "/home/work/disk/language-data"
    data_dirs = {  # total 24 part per dataset, each part size
        "instruct_english_CausalInstructions": [1, '\n'],  # 910M
        "instruct_english_AlpacaCoT": [1, '\n'],  # 180M
        "instruct_english_CoT": [1, '\n'],  # 86M
        "instruct_english_atlas_math": [1, '\n'],  # 56M
        "instruct_english_MathInstruct": [1, '\n'],  # 8M
        "instruct_english_mmlu": [2, '\n'],  # 7M
    }

    if load_model: # backup origin model
        os.system(f"cp -f {os.path.join(model_root, model_path)} {os.path.join(model_root, model_backup_path)}")

    if distributed:
        mp.spawn(main, nprocs=gpu_num, args=(gpu_num, distributed, load_model, save_model, load_dataset, config, arch, dtype, model_root, pretrain_model_path, model_path, tokenizer_path, token_dump_path, data_root, data_dirs, need_prepare_first_dataset, flash))
    else:
        main(0, gpu_num, distributed, load_model, save_model, load_dataset, config, arch, dtype, model_root, pretrain_model_path, model_path, tokenizer_path, token_dump_path, data_root, data_dirs, need_prepare_first_dataset, flash)
