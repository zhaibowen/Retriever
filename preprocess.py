import os
import gc
import re
import gzip
import json
import pandas as pd
import numpy as np
import zstandard as zstd
from langdetect import detect_langs
from tokenizers import Tokenizer
from tokenizers import normalizers, pre_tokenizers
from tokenizers import decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

def preview():
    # wiki = pd.read_parquet('/home/work/disk/language-data/wikipedia-20220301.en-0.005-validation/train-00000-of-00041.parquet')
    # gutenburg = pd.read_parquet('/home/work/disk/language-data/gutenberg_english/train-00000-of-00037-f5fce855b93d2d02.parquet')
    # books3 = pd.read_parquet('/home/work/disk/language-data/books3_english/train-00000-of-00213-312fd8d7a3c58a63.parquet')
    with gzip.open('/home/work/disk/language-data/c4/en/c4-train.00001-of-01024.json.gz', 'r') as f:
        json_bytes = f.read()
        json_str = json_bytes.decode('utf-8')
        json_list = json_str[1:-2].split('}\n{')
        print(len(json_list))
        for i, data in enumerate(json_list):
            if i % 10000 == 0: print(i)
            data = json.loads('{'+data+'}')
            if 'text' not in data or 'timestamp' not in data or 'url' not in data:
                print(data)

def shuffle_and_save_jsonl(root, tdir, data, template, label, knt, partition):
    block_num = knt // partition + 1
    parts = np.arange(partition)[None, :]
    parts = np.tile(parts, (block_num, 1))
    parts = parts.reshape(-1)
    np.random.shuffle(parts)
    parts = parts[:knt]

    print(f"total knt: {knt}, block num: {block_num}")

    tdir = os.path.join(root, tdir)
    for knt, x in data.iterrows():
        # if knt % 10000 == 0:
        #     print(knt)
        inputs = template.format_map(x).strip()
        targets = x[label].strip()
        text = {'inputs': inputs, 'targets': targets}
        text = json.dumps(text) + '\n'

        ti = parts[knt]
        tpath = os.path.join(tdir, f"train.{ti:02d}-of-{partition:02d}.txt")
        with open(tpath, '+a') as wf:
            wf.write(text)

def save_to_txt_wikipedia():
    root = '/home/work/disk/language-data'
    wikipedia_path = 'wikipedia-20220301.en-0.005-validation'
    wikipedia_files = os.listdir(os.path.join(root, wikipedia_path))
    wikipedia_files = list(filter(lambda x: 'parquet' in x, wikipedia_files))
    wikipedia_files.sort()

    target_dir = "english_wikipedia"
    for file in wikipedia_files:
        data = pd.read_parquet(os.path.join(root, wikipedia_path, file))['text']
        data = list(map(lambda x: x+'\r\n\r\n\r\n', data))
        with open(os.path.join(root, target_dir, file.replace('.parquet', '.txt')), 'w') as f:
            f.writelines(data)

def save_to_txt_gutenburg():
    root = '/home/work/disk/language-data'
    gutenberg_path = 'gutenberg_english'
    gutenberg_files = os.listdir(os.path.join(root, gutenberg_path))
    gutenberg_files = list(filter(lambda x: 'parquet' in x, gutenberg_files))
    gutenberg_files.sort()

    target_dir = "english_gutenberg"
    for file in gutenberg_files:
        data = pd.read_parquet(os.path.join(root, gutenberg_path, file))['TEXT'].to_list()
        with open(os.path.join(root, target_dir, file.replace('.parquet', '.txt')), 'w') as f:
            for book in data:
                book = list(filter(lambda x: len(x) > 0, book.split('\r\n')))
                book = list(map(lambda x: x.strip(), book))
                book = list(map(lambda x: x+' ' if x else '\n', book))
                book = ''.join(book)+'\r\n\r\n\r\n'
                f.write(book)

def detect_en(x, gutenberg_list):
    title, text = x['title'], x['text']
    if title in gutenberg_list:
        return False
    if len(text) < 1000:
        return False
    langs_list = detect_langs(text[:10000])
    if langs_list[0].lang != 'en' or langs_list[0].prob < 0.8:
        return False
    return True

def save_to_txt_books3():
    # 过滤掉books3里面非en的book，和与gutenburg重合的book（1450本）
    root = "/home/work/disk/language-data"
    gutenberg_list = set()

    source_dir = os.path.join(root, 'gutenberg_english')
    files = os.listdir(source_dir)
    files.sort()
    for i, file in enumerate(files):
        path = os.path.join(source_dir, file)
        books = pd.read_parquet(path)
        gutenberg_list.update(books['METADATA'].apply(lambda x: json.loads(x)['title']).tolist())
    print(f"gutenburg book nums: {len(gutenberg_list)}")

    source_dir = os.path.join(root, 'the_pile_books3_minus_gutenberg')
    target_dir = os.path.join(root, 'english_books3')
    files = os.listdir(source_dir)
    files.sort()
    for file in files:
        path = os.path.join(source_dir, file)
        books = pd.read_parquet(path)
        origin_shape = books.shape[0]
        filter_index = books.apply(detect_en, axis=1, args=(gutenberg_list,))
        books_filtered = books[filter_index]
        # books_filtered = books
        filter_shape = books_filtered.shape[0]
        print(file, f"origin: {origin_shape}, filter: {filter_shape}, ratio: {filter_shape/origin_shape:.2f}")

        data = books_filtered['text'].to_list()
        with open(os.path.join(root, target_dir, file.replace('.parquet', '.txt')), 'w') as f:
            for book in data:
                book = book.replace('\n\n', '\n')
                book = list(filter(lambda x: not re.match('^ *[|0-9.-]+ *$', x), book.split('\n')))
                book = '\n'.join(book)
                f.write(book+'\r\n\r\n\r\n')

def save_to_txt_c4():
    root = '/home/work/disk/language-data'
    c4_path = 'c4/en'
    c4_files = os.listdir(os.path.join(root, c4_path))
    c4_files = list(filter(lambda x: '.json.gz' in x, c4_files))
    c4_files.sort()

    target_dir = "english_c4"
    for file in c4_files:
        with gzip.open(os.path.join(root, c4_path, file), 'r') as f:
            with open(os.path.join(root, target_dir, file.replace('.json.gz', '.txt')), 'w') as tf:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                json_list = json_str[1:-2].split('}\n{')
                placeholder_knt = 0
                for i, data in enumerate(json_list):
                    data = json.loads('{'+data+'}')
                    if 'text' not in data or 'timestamp' not in data or 'url' not in data:
                        print(data)
                        continue
                    if 'placeholder page' in data['text']:
                        placeholder_knt += 1
                        continue
                    tf.write(data['text'] + '\r\n\r\n\r\n')
                print(file, "placeholder_knt:", placeholder_knt)
        exit()

def save_to_txt_mmlu():
    root = '/home/work/disk/language-data'
    mmlu_path = 'mmlu/data/auxiliary_train'
    mmlu_files = os.listdir(os.path.join(root, mmlu_path))
    mmlu_files = list(filter(lambda x: 'parquet' in x, mmlu_files))
    mmlu_files.sort()

    datas = []

    for i, file in enumerate(mmlu_files):
        print(i)
        path = os.path.join(root, mmlu_path, file)
        data = pd.read_parquet(path)
        datas.append(data)
        # if i > 3:
        #     break

    data = pd.concat(datas)
    data['A'] = data['choices'].apply(lambda x: x[0])
    data['B'] = data['choices'].apply(lambda x: x[1])
    data['C'] = data['choices'].apply(lambda x: x[2])
    data['D'] = data['choices'].apply(lambda x: x[3])
    data['label'] = data['answer'].apply(lambda x: chr(ord('A')+x))

    shuffled_data = data.sample(frac=1).reset_index(drop=True)

    template = (
        "{question}\n"
        "A: {A}\n"
        "B: {B}\n"
        "C: {C}\n"
        "D: {D}\n"
        "Answer: {label}\n"
    )
    
    step = 100000
    save_path = 'mmlu/train_data'
    knt = -1
    for i in range(0, shuffled_data.shape[0], step):
        knt += 1
        with open(os.path.join(root, save_path, f"train-{knt:04d}.txt"), 'w') as f:
            df = shuffled_data[i : i+step]
            samples = df.apply(lambda x: template.format_map(x), axis=1)
            for j, s in enumerate(samples):
                f.writelines(s+'\n\n\n')
                # if j > 10:
                #     break

def process_jsonl(root, spath, tdir, spliter='\n\n\n', pknt=100000, partition=1024):
    print(f"processing {spath}")
    spath = os.path.join(root, spath)
    knt = 0
    with open(spath, 'r') as f:
        x = f.readline()
        while x:
            knt += 1
            if knt % pknt == 0:
                print(knt)
                # break
            x = f.readline()

    block_num = knt // partition + 1
    parts = np.arange(partition)[None, :]
    parts = np.tile(parts, (block_num, 1))
    parts = parts.reshape(-1)
    np.random.shuffle(parts)
    parts = parts[:knt]

    print(f"total knt: {knt}, block num: {block_num}")

    knt = 0
    tdir = os.path.join(root, tdir)
    with open(spath, 'r') as f:
        x = f.readline()
        while x:
            text = json.loads(x)['text']+spliter
            ti = parts[knt]
            tpath = os.path.join(tdir, f"train.{ti:04d}-of-1024.txt")
            with open(tpath, '+a') as wf:
                wf.write(text)
            
            knt += 1
            if knt % pknt == 0:
                print(knt)
                # break
            x = f.readline()

def process_jsonl_stackexchange(root, spath, tdir, spliter='\n\n\n', pknt=100000, partition=1024):
    print(f"processing {spath}")
    spath = os.path.join(root, spath)
    knt = 0
    with open(spath, 'r') as f:
        x = f.readline()
        while x:
            knt += 1
            if knt % pknt == 0:
                print(knt)
                # break
            x = f.readline()

    block_num = knt // partition + 1
    parts = np.arange(partition)[None, :]
    parts = np.tile(parts, (block_num, 1))
    parts = parts.reshape(-1)
    np.random.shuffle(parts)
    parts = parts[:knt]

    print(f"total knt: {knt}, block num: {block_num}")

    knt = 0
    tdir = os.path.join(root, tdir)
    with open(spath, 'r') as f:
        x = f.readline()
        while x:
            text = json.loads(x)['text']
            text = text.split('\nA:')
            if len(text) < 2:
                x = f.readline()
                continue
            text = text[0][3:] + '\nResponse:' + text[1] + spliter
            ti = parts[knt]
            tpath = os.path.join(tdir, f"train.{ti:04d}-of-1024.txt")
            with open(tpath, '+a') as wf:
                wf.write(text)
            
            knt += 1
            if knt % pknt == 0:
                print(knt)
                # break
            x = f.readline()
    
def process_jsonl_dir(root, spath, tdir, spliter='\n\n\n', pknt=100000):
    print(f"processing {spath}")
    spath = os.path.join(root, spath)
    tfiles = os.listdir(spath)
    tfiles = list(map(lambda x: os.path.join(spath, x), tfiles))

    knt = 0
    for tf in tfiles:
        with open(tf, 'r') as f:
            x = f.readline()
            while x:
                knt += 1
                if knt % pknt == 0:
                    print(knt)
                    # break
                x = f.readline()
        # if knt % pknt == 0:
        #     break

    partition = 1024
    block_num = knt // partition + 1
    parts = np.arange(partition)[None, :]
    parts = np.tile(parts, (block_num, 1))
    parts = parts.reshape(-1)
    np.random.shuffle(parts)
    parts = parts[:knt]

    print(f"total knt: {knt}, block num: {block_num}")

    knt = 0
    tdir = os.path.join(root, tdir)
    for tf in tfiles:
        with open(tf, 'r') as f:
            x = f.readline()
            while x:
                text = json.loads(x)['text']+spliter
                ti = parts[knt]
                tpath = os.path.join(tdir, f"train.{ti:04d}-of-{partition}.txt")
                with open(tpath, '+a') as wf:
                    wf.write(text)
                
                knt += 1
                if knt % pknt == 0:
                    print(knt)
                    # break
                x = f.readline()
        # if knt % pknt == 0:
        #     break

def process_instruct_jsonl(root, spath, tdir, pknt=100000, partition=1024, spliter='\n▁\n▁\n'):
    print(f"processing {spath}")
    spath = os.path.join(root, spath)
    knt = 0
    with open(spath, 'r') as f:
        x = f.readline()
        while x:
            knt += 1
            if knt % pknt == 0:
                print(knt)
                # break
            x = f.readline()

    block_num = knt // partition + 1
    parts = np.arange(partition)[None, :]
    parts = np.tile(parts, (block_num, 1))
    parts = parts.reshape(-1)
    np.random.shuffle(parts)
    parts = parts[:knt]

    print(f"total knt: {knt}, block num: {block_num}")

    knt = 0
    tdir = os.path.join(root, tdir)
    with open(spath, 'r') as f:
        x = f.readline()
        while x:
            x = json.loads(x)
            inputs = x['inputs'].strip()
            targets = x['targets'].strip()
            # text = {'inputs': inputs, 'targets': targets}
            # text = json.dumps(text) + '\n'
            text = inputs.split('Answer:')[0].strip() + '\nResponse:' + targets + spliter

            ti = parts[knt]
            tpath = os.path.join(tdir, f"train.{ti:04d}-of-{partition:04d}.txt")
            with open(tpath, '+a') as wf:
                wf.write(text)
            
            knt += 1
            if knt % pknt == 0:
                print(knt)
                # break
            x = f.readline()

def save_to_jsonl_mmlu(root, spath, tdir, partition=64):
    mmlu_files = os.listdir(os.path.join(root, spath))
    mmlu_files = list(filter(lambda x: 'parquet' in x, mmlu_files))
    mmlu_files.sort()

    datas = []
    knt = 0
    for i, file in enumerate(mmlu_files):
        print(i)
        path = os.path.join(root, spath, file)
        data = pd.read_parquet(path)
        datas.append(data)
        knt += data.shape[0]

    data = pd.concat(datas)
    data['A'] = data['choices'].apply(lambda x: x[0])
    data['B'] = data['choices'].apply(lambda x: x[1])
    data['C'] = data['choices'].apply(lambda x: x[2])
    data['D'] = data['choices'].apply(lambda x: x[3])
    data['label'] = data['answer'].apply(lambda x: chr(ord('A')+x))

    template = (
        "The following are multiple-choice questions, please choose the correct answer.\n"
        "{question}\n"
        "A: {A}\n"
        "B: {B}\n"
        "C: {C}\n"
        "D: {D}"
    )
    label = 'label'
    shuffle_and_save_jsonl(root, tdir, data, template, label, knt, partition)
        
def save_to_jsonl_alpaca(root, spath, tdir, partition=64):
    path = os.path.join(root, spath)
    data = pd.read_parquet(path)
    knt = data.shape[0]

    template = (
        "{instruction}\n"
        "{input}"
    )
    label = 'output'
    shuffle_and_save_jsonl(root, tdir, data, template, label, knt, partition)

def save_to_jsonl_naturalquestion(root, spath, tdir, partition=64):
    path = os.path.join(root, spath)
    data = pd.read_parquet(path)
    data = data[data.apply(lambda x: x['context'][:3]=='<P>', axis=1)]
    data['context2'] = data['context'].apply(lambda x: x[3:-4].strip())
    data['answer'] = data['answers'].apply(lambda x: x[0])
    data = data.reset_index(drop=True)
    knt = data.shape[0]

    template = (
        "{context2}\n"
        "{question}"
    )
    label = 'answer'
    shuffle_and_save_jsonl(root, tdir, data, template, label, knt, partition)

    template = (
        "{question}"
    )
    label = 'answer'
    shuffle_and_save_jsonl(root, tdir, data, template, label, knt, partition)

def save_to_jsonl_triviaqa(root, spath, tdir, partition=64):
    path = os.path.join(root, spath)
    data = pd.read_parquet(path)
    data['answer'] = data['answers'].apply(lambda x: x[0])
    knt = data.shape[0]

    template = (
        "{question}"
    )
    label = 'answer'
    shuffle_and_save_jsonl(root, tdir, data, template, label, knt, partition)

def save_to_jsonl_CoT(root, spath, tdir, partition=4):
    path = os.path.join(root, spath)

    template = (
        "{source}"
    )
    label = 'rationale'

    with open(path, 'r') as f:
        data = json.load(f)
        data = list(data.values())
        data = list(map(lambda x: {'inputs': template.format_map(x), 'targets': x[label]}, data))
        data = list(map(lambda x: json.dumps(x) + '\n', data))

    knt = len(data)
    block_num = knt // partition + 1
    parts = np.arange(partition)[None, :]
    parts = np.tile(parts, (block_num, 1))
    parts = parts.reshape(-1)
    np.random.shuffle(parts)
    parts = parts[:knt]

    print(f"total knt: {knt}, block num: {block_num}")

    tdir = os.path.join(root, tdir)
    for knt, text in enumerate(data):
        if knt % 100000 == 0:
            print(knt)

        ti = parts[knt]
        tpath = os.path.join(tdir, f"train.{ti:02d}-of-{partition:02d}.txt")
        with open(tpath, '+a') as wf:
            wf.write(text)

def save_to_jsonl_MathInstruct(root, spath, tdir, partition=4):
    path = os.path.join(root, spath)

    template = (
        "{instruction}"
    )
    label = 'output'

    with open(path, 'r') as f:
        data = json.load(f)
        data = list(map(lambda x: {'inputs': template.format_map(x), 'targets': x[label]}, data))
        data = list(map(lambda x: json.dumps(x) + '\n', data))

    knt = len(data)
    block_num = knt // partition + 1
    parts = np.arange(partition)[None, :]
    parts = np.tile(parts, (block_num, 1))
    parts = parts.reshape(-1)
    np.random.shuffle(parts)
    parts = parts[:knt]

    print(f"total knt: {knt}, block num: {block_num}")

    tdir = os.path.join(root, tdir)
    for knt, text in enumerate(data):
        if knt % 100000 == 0:
            print(knt)

        ti = parts[knt]
        tpath = os.path.join(tdir, f"train.{ti:02d}-of-{partition:02d}.txt")
        with open(tpath, '+a') as wf:
            wf.write(text)

def save_to_jsonl_MetaMathQA(root, spath, tdir, partition=1024, spliter='\n▁\n▁\n'):
    path = os.path.join(root, spath)

    template = (
        "{query}\n"
        "Response:{response}"
    )

    with open(path, 'r') as f:
        data = json.load(f)
        data = list(map(lambda x: template.format_map(x), data))
        data = list(map(lambda x: x.split('####')[0].strip(), data))

    knt = len(data)
    block_num = knt // partition + 1
    parts = np.arange(partition)[None, :]
    parts = np.tile(parts, (block_num, 1))
    parts = parts.reshape(-1)
    np.random.shuffle(parts)
    parts = parts[:knt]

    print(f"total knt: {knt}, block num: {block_num}")

    tdir = os.path.join(root, tdir)
    for knt, text in enumerate(data):
        if knt % 100000 == 0:
            print(knt)

        ti = parts[knt]
        tpath = os.path.join(tdir, f"train.{ti:04d}-of-{partition:04d}.txt")
        with open(tpath, '+a') as wf:
            wf.write(text+spliter)

def save_to_jsonl_atlas_math(root, spath, tdir, partition=1024, spliter='\n▁\n▁\n'):
    path = os.path.join(root, spath)

    template = (
        "{instruction}\n"
        "Response:{answer}"
    )

    with open(path, 'r') as f:
        data = f.readlines()
        data = list(map(lambda x: template.format_map(json.loads(x)), data))

    knt = len(data)
    block_num = knt // partition + 1
    parts = np.arange(partition)[None, :]
    parts = np.tile(parts, (block_num, 1))
    parts = parts.reshape(-1)
    np.random.shuffle(parts)
    parts = parts[:knt]

    print(f"total knt: {knt}, block num: {block_num}")

    tdir = os.path.join(root, tdir)
    for knt, text in enumerate(data):
        if knt % 100000 == 0:
            print(knt)

        ti = parts[knt]
        tpath = os.path.join(tdir, f"train.{ti:04d}-of-{partition:04d}.txt")
        with open(tpath, '+a') as wf:
            wf.write(text+spliter)

def save_to_jsonl_AlpacaCoT(root, spath, tdir, partition=4):
    path = os.path.join(root, spath)
    files = os.listdir(path)
    files.sort()

    template = (
        "{instruction}\n"
        "{input}"
    )
    label = 'output'
    for file in files:
        data = pd.read_parquet(os.path.join(path, file))
        knt = data.shape[0]
        data = data[data['instruction'] != data['output']]
        shuffle_and_save_jsonl(root, tdir, data, template, label, knt, partition)

def process_zst_jsonl(root, spath, tdir, partition=1024, spliter='\n▁\n▁\n'):
    path = os.path.join(root, spath)
    files = os.listdir(path)
    files.sort()
    tdir = os.path.join(root, tdir)

    for i, file in enumerate(files):
        file = os.path.join(path, file)
        with open(file, "rb") as f:
            data = f.read()
            dctx = zstd.ZstdDecompressor()
            decompressed = dctx.decompress(data)
            decompressed = decompressed.decode('utf8')
            data = decompressed.split('{"pred_label":')
            data = list(map(lambda x: '{"pred_label":' + x.strip(), data[1:]))
            data = list(map(lambda x: json.loads(x)['text'], data))
            knt = len(data)

            block_num = knt // partition + 1
            parts = np.arange(partition)[None, :]
            parts = np.tile(parts, (block_num, 1))
            parts = parts.reshape(-1)
            np.random.shuffle(parts)
            parts = parts[:knt]

            print(f"{i}, total knt: {knt}, block num: {block_num}")

            knt = 0
            for text in data:
                text = text + spliter
                ti = parts[knt]
                tpath = os.path.join(tdir, f"train.{ti:04d}-of-1024.txt")
                with open(tpath, '+a') as wf:
                    wf.write(text)
                knt += 1
            
def save_to_jsonl_CausalInstructions(root, spath, tdir, partition=4):
    path = os.path.join(root, spath)
    files = os.listdir(path)
    files.sort()
    tdir = os.path.join(root, tdir)
    max_len = 5000

    template = (
        "{instruction}\n"
        "{input}"
    )
    label = 'output'
    for i, file in enumerate(files):
        data = pd.read_parquet(os.path.join(path, file))
        x1 = data.shape[0]
        data = data[data.apply(lambda x: len(x['instruction']) + len(x['input']) + len(x['output']) < max_len, axis=1)]
        x2 = data.shape[0]
        data = data[data.apply(lambda x: '[HM]: Instruction:' not in (x['instruction'] + x['input']), axis=1)]
        x3 = data.shape[0]
        data = data[data['instruction'] != data['output']]
        x4 = data.shape[0]
        print(i, x1, x2, x3, x4)
        data = data.reset_index(drop=True)
        knt = data.shape[0]
        shuffle_and_save_jsonl(root, tdir, data, template, label, knt, partition)

def save_to_jsonl_finetune_atlas_math(root, spath, tdir, partition=24):
    path = os.path.join(root, spath)
    with open(path, 'r') as f:
        data = f.readlines()
        data = list(map(lambda x: json.loads(x), data))
    knt = len(data)

    template = [("{instruction}")] * 8 + [("{input}")] + [("{input} =")]
    label = 'answer'

    block_num = knt // partition + 1
    parts = np.arange(partition)[None, :]
    parts = np.tile(parts, (block_num, 1))
    parts = parts.reshape(-1)
    np.random.shuffle(parts)
    parts = parts[:knt]

    print(f"total knt: {knt}, block num: {block_num}")

    tdir = os.path.join(root, tdir)
    for knt, x in enumerate(data):
        if knt % 100000 == 0:
            print(knt)
        inputs = template[knt % len(template)].format_map(x).strip()
        targets = x[label].strip()
        text = {'inputs': inputs, 'targets': targets}
        text = json.dumps(text) + '\n'

        ti = parts[knt]
        tpath = os.path.join(tdir, f"train.{ti:02d}-of-{partition:02d}.txt")
        with open(tpath, '+a') as wf:
            wf.write(text)

def save_to_txt_CoT(root, spath='CoT_collection_en/CoT_collection_en.json', tdir='english_CoT', partition=1024, spliter='\n▁\n▁\n'):
    path = os.path.join(root, spath)

    template = (
        "{source}\n"
        "Response:{rationale}"
    )

    with open(path, 'r') as f:
        data = json.load(f)
        data = list(data.values())
        data = list(map(lambda x: template.format_map(x), data))
        data = list(map(lambda x: x + spliter, data))

    knt = len(data)
    block_num = knt // partition + 1
    parts = np.arange(partition)[None, :]
    parts = np.tile(parts, (block_num, 1))
    parts = parts.reshape(-1)
    np.random.shuffle(parts)
    parts = parts[:knt]

    print(f"total knt: {knt}, block num: {block_num}")

    tdir = os.path.join(root, tdir)
    for knt, text in enumerate(data):
        if knt % 100000 == 0:
            print(knt)

        ti = parts[knt]
        tpath = os.path.join(tdir, f"train.{ti:04d}-of-{partition:04d}.txt")
        with open(tpath, '+a') as wf:
            wf.write(text)

def save_to_txt_AlpacaCoT(root, spath, tdir, partition=1024, spliter='\n▁\n▁\n'):
    path = os.path.join(root, spath)
    files = os.listdir(path)
    files.sort()

    template = (
        "{instruction}\n"
        "{input}\n"
        "Response:{output}"
    )
    for file in files:
        data = pd.read_parquet(os.path.join(path, file))
        data = data[data['instruction'] != data['output']]
        data = data.apply(lambda x: template.format_map(x), axis=1)
        data = data.to_list()
        data = list(map(lambda x: x + spliter, data))
        knt = len(data)
        
        block_num = knt // partition + 1
        parts = np.arange(partition)[None, :]
        parts = np.tile(parts, (block_num, 1))
        parts = parts.reshape(-1)
        np.random.shuffle(parts)
        parts = parts[:knt]

        print(f"total knt: {knt}, block num: {block_num}")

        tdir = os.path.join(root, tdir)
        for knt, text in enumerate(data):
            if knt % 100000 == 0:
                print(knt)

            ti = parts[knt]
            tpath = os.path.join(tdir, f"train.{ti:04d}-of-{partition:04d}.txt")
            with open(tpath, '+a') as wf:
                wf.write(text)
            
def save_to_txt_CausalInstructions(root, spath, tdir, partition=1024, spliter='\n▁\n▁\n'):
    path = os.path.join(root, spath)
    files = os.listdir(path)
    files.sort()
    tdir = os.path.join(root, tdir)
    max_len = 5000

    template = (
        "{instruction}\n"
        "{input}\n"
        "Response:{output}"
    )
    
    for i, file in enumerate(files):
        data = pd.read_parquet(os.path.join(path, file))
        x1 = data.shape[0]
        data = data[data.apply(lambda x: len(x['instruction']) + len(x['input']) + len(x['output']) < max_len, axis=1)]
        x2 = data.shape[0]
        data = data[data.apply(lambda x: '[HM]: Instruction:' not in (x['instruction'] + x['input']), axis=1)]
        x3 = data.shape[0]
        data = data[data['instruction'] != data['output']]
        x4 = data.shape[0]
        print(i, x1, x2, x3, x4)
        
        data = data.apply(lambda x: template.format_map(x), axis=1)
        data = data.to_list()
        data = list(map(lambda x: x + spliter, data))
        knt = len(data)
        
        block_num = knt // partition + 1
        parts = np.arange(partition)[None, :]
        parts = np.tile(parts, (block_num, 1))
        parts = parts.reshape(-1)
        np.random.shuffle(parts)
        parts = parts[:knt]

        print(f"total knt: {knt}, block num: {block_num}")

        tdir = os.path.join(root, tdir)
        for knt, text in enumerate(data):
            if knt % 100000 == 0:
                print(knt)

            ti = parts[knt]
            tpath = os.path.join(tdir, f"train.{ti:04d}-of-{partition:04d}.txt")
            with open(tpath, '+a') as wf:
                wf.write(text)

if __name__ == "__main__":
    # 数据预览
    # preview()

    # 保存成txt格式，供训练使用
    # save_to_txt_wikipedia()
    # save_to_txt_gutenburg()
    # save_to_txt_books3()
    # save_to_txt_c4()
    # save_to_txt_mmlu()

    # RedPajama-Data-1T的arxiv, wiki, books, stackexchange, 所有文件都存储成1024份，和c4保持一致
    # root = '/home/work/disk8T/language-data/RedPajama-Data-1T'
    # process_jsonl_stackexchange(root, spath='stackexchange.jsonl', tdir='english_stackexchange', spliter='\n▁\n▁\n')
    # process_jsonl(root, spath='wiki.jsonl', tdir='english_wiki', spliter='\n▁\n▁\n')
    # process_jsonl(root, spath='book.jsonl', tdir='english_books', spliter='\n▁\n▁\n', pknt=10000)
    # process_jsonl_dir(root, spath='arxiv', tdir='english_arxiv', spliter='\n▁\n▁\n')

    # root = '/home/work/disk8T/language-data'
    # process_instruct_jsonl(root, spath='RedPajama-Data-Instruct/P3_decontaminated.jsonl', tdir='english_P3_decontaminated', partition=1024, spliter='\n▁\n▁\n')
    # process_zst_jsonl(root, spath='RedPajama-Data-1T/CommonCrawl2019', tdir='english_CommonCrawl2019', partition=1024, spliter='\n▁\n▁\n')
    # process_zst_jsonl(root, spath='RedPajama-Data-1T/CommonCrawl2023', tdir='english_CommonCrawl2023', partition=1024, spliter='\n▁\n▁\n')

    root = '/home/work/disk/language-data'
    # save_to_jsonl_MetaMathQA(root, spath='MetaMathQA/MetaMathQA-395K.json', tdir='english_MetaMathQA', partition=1024, spliter='\n▁\n▁\n')
    # save_to_jsonl_atlas_math(root, spath='atlas_math/atlas-math-train.jsonl', tdir='english_atlas_math', partition=1024, spliter='\n▁\n▁\n')
    # save_to_txt_CoT(root, spath='CoT_collection_en/CoT_collection_en.json', tdir='english_CoT', partition=1024, spliter='\n▁\n▁\n')
    # save_to_txt_AlpacaCoT(root, spath='Alpaca-CoT', tdir='english_AlpacaCoT', partition=1024, spliter='\n▁\n▁\n')
    save_to_txt_CausalInstructions(root, spath='CausalInstructions', tdir='english_CausalInstructions', partition=1024, spliter='\n▁\n▁\n')

    # finetune
    # root = '/home/work/disk/language-data'
    # save_to_jsonl_mmlu(root, spath='mmlu/data/auxiliary_train', tdir='instruct_english_mmlu', partition=24)
    # save_to_jsonl_alpaca(root, spath='alpaca/train.parquet', tdir='instruct_english_alpaca', partition=24)
    # save_to_jsonl_naturalquestion(root, spath='naturalquestionsshortqa/data/train.parquet', tdir='instruct_english_NQ', partition=24)
    # save_to_jsonl_triviaqa(root, spath='triviaqa/data/train.parquet', tdir='instruct_english_triviaqa', partition=24)
    # save_to_jsonl_CoT(root, spath='CoT_collection_en/CoT_collection_en.json', tdir='instruct_english_CoT', partition=24)
    # save_to_jsonl_MathInstruct(root, spath='MathInstruct/MathInstruct.json', tdir='instruct_english_MathInstruct', partition=24)
    # save_to_jsonl_AlpacaCoT(root, spath='Alpaca-CoT', tdir='instruct_english_AlpacaCoT', partition=24)
    # save_to_jsonl_CausalInstructions(root, spath='CausalInstructions', tdir='instruct_english_CausalInstructions', partition=24)
    # save_to_jsonl_finetune_atlas_math(root, spath='atlas_math/atlas-math-train.jsonl', tdir='instruct_english_atlas_math', partition=24)