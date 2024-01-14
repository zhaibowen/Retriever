import os
import json
import pandas as pd
from tokenizers import Tokenizer
from tokenizers import normalizers, pre_tokenizers
from tokenizers import decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

def get_all_files(root, target_dirs, ratio=0.3):
    result = []
    for dir in target_dirs:
        abdir_path = os.path.join(root, dir)
        files = list(map(lambda x: os.path.join(abdir_path, x), os.listdir(abdir_path)))
        files = list(filter(lambda x: 'train' in x, files))
        files.sort()
        result.extend(files[:int(len(files)*ratio)])
    return result

def bpe_tokenization(model_root, data_root):
    tokenizer = Tokenizer(BPE(unk_token="<unk>", fuse_unk=False, byte_fallback=True))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Replace(' ', '▁')
    ]) 
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex('▁'), behavior='merged_with_next'),
        pre_tokenizers.Split(Regex('\d|[\u2E80-\u2FDF\u3040-\u318F\u31A0-\u31BF\u31F0-\u31FF\u3400-\u4DB5\u4E00-\u9FFF\uA960-\uA97F\uAC00-\uD7FF]'), behavior='isolated'),
        pre_tokenizers.Split(Regex('▁*(\w+|[^\w\s]+)'), behavior='isolated')
    ])
    tokenizer.decoder = decoders.Sequence([
        decoders.Replace('▁', ' '),
        decoders.ByteFallback()
    ])
    trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>"], 
                         vocab_size=32000-256,
                         limit_alphabet=2000)

    # tokenizer.train(files=['/home/work/disk/vision/retriever/text.txt'], trainer=trainer)

    # 1040G
    target_dirs = ['english_c4', 'english_arxiv', 'english_books', 'english_stackexchange', 'english_wiki']
    train_files = get_all_files(data_root, target_dirs, ratio=0.6)
    tokenizer.train(files=train_files, trainer=trainer) # 683462 Mo, 187167335 pairs

    save_path = os.path.join(model_root, 'tokenizer.json')
    tokenizer.save(save_path)

    with open('/home/work/disk/vision/retriever/tokenizer.json', 'r') as f:
        x = json.load(f)
        new_vocab = {}
        new_vocab['<unk>'] = 0
        new_vocab['<s>'] = 1
        new_vocab['</s>'] = 2

        for i in range(256):
            hexn = hex(i)[2:].upper()
            s = f"<0x{hexn:>02s}>"
            new_vocab[s] = i + 3

        for k, v in x['model']['vocab'].items():
            if k not in ['<unk>', '<s>', '</s>']:
                new_vocab[k] = v + 256

        x['model']['vocab'] = new_vocab

    with open('/home/work/disk/vision/retriever/tokenizer2.json', 'w') as f:
        json.dump(x, f, indent=2, ensure_ascii=False)

    tokenizer = Tokenizer.from_file(os.path.join(model_root, 'tokenizer2.json'))
    output = tokenizer.encode("Hello, y'all! How are you 😁\n您好，我叫小张 Ã?\nwhat is the meal today ")
    print(output.tokens)
    output = tokenizer.decode(output.ids)
    print(output)

def bpe_validate(model_root, data_root):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file='/home/work/disk/vision/retriever/tokenizer_models/tokenizer_v2_600G.json', eos_token='</s>')

    query = "Hello, y'all! How are you 😁\nwhat is the weather like today?\nThis year is 2023"
    query2 = "what is your name"
    output = tokenizer([query, query2], max_length=10, return_overflowing_tokens=True, stride=0, truncation=False)['input_ids']
    # print(output.tokens)
    # output = tokenizer.decode(output.ids)
    print(output)

    c4_dir = '/home/work/disk/language-data/english_stackexchange'
    c4_files = list(map(lambda x: os.path.join(c4_dir, x), os.listdir(c4_dir)))
    c4_files = list(filter(lambda x: 'train' in x, c4_files))
    c4_files.sort()
    # print(c4_files)

    for file in c4_files:
        data = open(file, 'r').readlines()
        char_knt = sum(list(map(lambda x: len(x), data)))
        word_knt = list(filter(lambda x: x.strip(), data))
        word_knt = sum(list(map(lambda x: len(x.split(' ')), word_knt)))
        output = tokenizer(data)['input_ids']
        output = sum(list(map(lambda x: len(x), output)))
        print(f"char: {char_knt}, word: {word_knt}, token: {output}, c2t: {char_knt/output:.3f}, w2t: {word_knt/output:.3f}")

        # char: 97143672, word: 16298401, token: 22874292, c2t: 4.247, w2t: 0.713

if __name__ == "__main__":
    model_root = "/home/work/disk/vision/retriever"
    data_root = '/home/work/disk/language-data'

    # bpe_tokenization(model_root, data_root)
    bpe_validate(model_root, data_root)

# [11:20:25] Pre-processing files (683462 Mo)         ████████████████████████████████████████████████████████████████████                100%
# [00:09:32] Tokenize words                           ████████████████████████████████████████████████████████████████████ 187167335/187167335
# [04:33:37] Count pairs                              ████████████████████████████████████████████████████████████████████ 187167335/187167335
# [08:28:39] Compute merges                           ████████████████████████████████████████████████████████████████████ 29741    /    29741