import os
import time
import copy
import json
import torch
import random
import pickle
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

def mask_generator(x):
    mask = np.zeros([x.shape[0], x.shape[0]], dtype=np.float32)
    indices = np.where(x == 2)[0]
    for i in indices:
        if i + 1 == x.shape[0]:
            continue
        mask[i+1 :, 0 : i+1] = -1
    return mask

class GPTDataset(Dataset):
    def __init__(self, data_root, data_dirs, sequence_length, set_name, ratio=1.0, shuffle=True, mask_flag=False):
        random.seed(1)
        data_paths = []
        splitters = []
        for data_dir, es in data_dirs.items():
            data_epochs, data_splitter = es
            files = self.get_all_files(data_root, data_dir, set_name, ratio)
            if set_name == "validation":
                data_epochs = 1
            splitters += [data_splitter] * data_epochs
            tmp_paths = [[] for _ in range(data_epochs)]
            knt = 0
            for _ in range(data_epochs):
                fc = copy.copy(files)
                if shuffle:
                    random.shuffle(fc)
                for f in fc:
                    tmp_paths[knt % data_epochs].append(f)
                    knt += 1
            data_paths += tmp_paths

        self.tokens = np.empty((1, sequence_length))
        self.labels = np.empty((1, sequence_length))
        self.files = list(zip(*data_paths))
        self.splitters = splitters
        self.sequence_length = sequence_length
        self.mask_flag = mask_flag

    @staticmethod
    def get_all_files(root, data_dir, set_name='train', ratio=1.0):
        data_path = os.path.join(root, data_dir)
        files = list(map(lambda x: os.path.join(data_path, x), os.listdir(data_path)))
        files = list(filter(lambda x: set_name in x, files))
        files.sort()
        return files[:int(len(files)*ratio)]
    
    def tokenize(self, index, tokenizer, token_dump_path, is_batch=False):
        if index >= len(self.files):
            return
        
        eos = tokenizer.encode('</s>')[0]
        data = []
        for file, splitter in zip(self.files[index], self.splitters):
            # print(file, splitter)
            with open(file, 'r') as f:
                dt = f.readlines()
                dt = ''.join(dt).split(splitter)
                dt = list(map(lambda x: x.strip(), dt))
                data += dt

        # truncate long text
        chunk = 20000
        data = list(map(lambda x: x[:chunk], data))
        
        random.shuffle(data)

        # split in chunks
        # chk_data = []
        # chunk = 100000
        # for sample in data:
        #     for i in range(0, len(sample), chunk):
        #         chk_data.append(sample[i : i + chunk])
        # data = chk_data

        if is_batch:
            tokens = tokenizer(data)['input_ids']
        else:
            step = 100
            tokens = []
            for i in range(0, len(data), step):
                tokens.extend(tokenizer(data[i:i+step])['input_ids'])
        tokens = list(map(lambda x: x + [eos] * 10, tokens))

        tokens = [elem for tl in tokens for elem in tl]
        truncate_len = len(tokens) // self.sequence_length * self.sequence_length
        tokens = tokens[:truncate_len]
        tokens = np.array(tokens, dtype=np.int32).reshape((-1, self.sequence_length))

        labels = tokens.copy()
        pickle.dump((len(data), tokens, labels), open(token_dump_path, 'wb'))

    def tokenize_with_packing(self, index, tokenizer, token_dump_path, is_batch=False):
        if index >= len(self.files):
            return
        
        eos = tokenizer.encode('</s>')[0]
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        data = []
        target = []
        for file in self.files[index]:
            with open(file, 'r') as f:
                dt = f.readlines()
                dt = list(map(lambda x: json.loads(x.strip()), dt))
                da = list(map(lambda x: f"{x['inputs']}\nResponse:", dt))
                db = list(map(lambda x: x['targets'], dt))
                data += da
                target += db
        if is_batch:
            tokens = tokenizer(data)['input_ids']
            label_tokens = tokenizer(target)['input_ids']
        else:
            step = 100
            tokens, label_tokens = [], []
            for i in range(0, len(data), step):
                tokens.extend(tokenizer(data[i:i+step])['input_ids'])
                label_tokens.extend(tokenizer(target[i:i+step])['input_ids'])
        label_tokens = list(map(lambda x: x + [eos], label_tokens))

        inputs = list(map(lambda x: x[0] + x[1], zip(tokens, label_tokens)))
        labels = list(map(lambda x: [IGNORE_INDEX]*len(x[0]) + x[1], zip(tokens, label_tokens)))
        
        zip_inputs_labels = list(zip(inputs, labels))
        random.shuffle(zip_inputs_labels)
        inputs, labels = zip(*zip_inputs_labels)

        # drop last sample
        token_lists = []
        padding_labels = []
        cur_tok = inputs[0]
        cur_lab = labels[0]
        for i, (ts, lbs) in enumerate(zip(inputs[1:], labels[1:])):
            if len(cur_tok) + len(ts) <= self.sequence_length:
                cur_tok += ts
                cur_lab += lbs
            else:
                if len(cur_tok) <= self.sequence_length:
                    pad_num = self.sequence_length - len(cur_tok)
                    cur_tok += [0] * pad_num
                    cur_lab += [IGNORE_INDEX] * pad_num
                else:
                    cur_tok = cur_tok[-self.sequence_length:]
                    cur_lab = cur_lab[-self.sequence_length:]
                token_lists.append(cur_tok)
                padding_labels.append(cur_lab)
                cur_tok = ts
                cur_lab = lbs

        token_lists = np.array(token_lists, dtype=np.int32)
        padding_labels = np.array(padding_labels, dtype=np.int32)
        pickle.dump((len(data), token_lists, padding_labels), open(token_dump_path, 'wb'))

    def load_samples(self, token_dump_path):
        num_origin_samples, self.tokens, self.labels = pickle.load(open(token_dump_path, 'rb'))
        return num_origin_samples
    
    def reset_samples(self):
        self.tokens = np.empty((1, self.sequence_length))
        self.labels = np.empty((1, self.sequence_length))

    def __len__(self):
        return self.tokens.shape[0]
    
    def __getitem__(self, index):
        if self.mask_flag:
            att_masks = mask_generator(self.tokens[index])
            return self.tokens[index], self.labels[index], att_masks
        else:
            return self.tokens[index], self.labels[index]

class RandSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last, shuffle=True):
        self.batch_size = batch_size
        self.order = list(range(len(data_source)))
        self.total_size = len(self.order) - len(self.order) % self.batch_size
        if not drop_last: self.total_size += batch_size

        if shuffle: random.shuffle(self.order)
        self.groups = []
        for i in range(0, self.total_size, self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    def __iter__(self):
        for group in self.groups:
            yield group

    def __len__(self):
        return len(self.groups)

class DistRandSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last, shuffle=True):
        self.rank = dist.get_rank()
        self.num_replicas = dist.get_world_size()
        self.batch_size = batch_size
        self.order = list(range(len(data_source)))
        self.total_size = len(self.order) - len(self.order) % (self.num_replicas * self.batch_size)
        if not drop_last: self.total_size += self.num_replicas * self.batch_size

        if shuffle:
            g = torch.Generator()
            g.manual_seed(-1)
            self.order = torch.randperm(len(self.order), generator=g).tolist()
        self.groups = []
        for i in range(self.rank * self.batch_size, self.total_size, self.num_replicas * self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    def __iter__(self):
        for group in self.groups:
            yield group

    def __len__(self):
        return len(self.groups)

def FixCollater(batch):
    inputs, labels = zip(*batch)
    inputs = torch.from_numpy(np.array(inputs)).to(torch.long)
    labels = torch.from_numpy(np.array(labels)).to(torch.long)
    return inputs, labels

def MaskCollater(batch):
    inputs, labels, att_masks = zip(*batch)
    inputs = torch.from_numpy(np.array(inputs)).to(torch.long)
    labels = torch.from_numpy(np.array(labels)).to(torch.long)
    att_masks = torch.from_numpy(np.array(att_masks)).to(torch.float32)
    return inputs, labels, att_masks

if __name__ == "__main__":
    model_root = "/home/work/disk/vision/retriever"
    model_dir = "tokenizer_models"
    model_name="tokenizer_300G.json"
    data_root = "/home/work/disk/language-data"
    data_dirs = {
        "english_c4": 1, 
        # "english_wikipedia": 2, 
        # "english_gutenberg": 2, 
        # "english_books3": 1
    }
    batch_size = 16
    sequence_length = 1024

    model_path = os.path.join(model_root, model_dir, model_name)
    train_dataset = GPTDataset(model_path, data_root, data_dirs, sequence_length, set_name="train", ratio=0.1, shuffle=False)
    for i, file in enumerate(train_dataset.files):
        st = time.time()
        origin = train_dataset.tokenize(file)
        print(f"processing {file[len(data_root)+1:]}, origin: {origin}, samples: {len(train_dataset)}, tokenize comsume: {int(time.time()-st)}s")
        train_sampler = RandSampler(train_dataset, batch_size=batch_size, drop_last=True)
        train_loader = DataLoader(train_dataset, num_workers=0, pin_memory=True, collate_fn=FixCollater, batch_sampler=train_sampler)
        
        for j, data in enumerate(train_loader):
            a = 1
        break
