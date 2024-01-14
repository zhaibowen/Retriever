import os
import pandas as pd
from tokenizers import Tokenizer
from tokenizers import normalizers, pre_tokenizers
from tokenizers import decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

def count_word_the():
    root = '/home/work/disk/language-data'
    for target_dir in ['english_wikipedia']:
        abdir_path = os.path.join(root, target_dir)
        files = list(map(lambda x: os.path.join(abdir_path, x), os.listdir(abdir_path)))
        files = list(filter(lambda x: 'train' in x, files))
        files.sort()
        knt = 0
        for file in files:
            with open(file, 'r') as f:
                lines = f.readlines()
                lines = list(map(lambda x: len(x.split('alaal'))-1, lines))
                knt += sum(lines)
            break
        print(target_dir, knt)

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
    tokenizer = Tokenizer(BPE(unk_token="<unk>", fuse_unk=True))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC()
    ]) 
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(' '), behavior='merged_with_next'),
        pre_tokenizers.Split(Regex('\d|[\u2E80-\u2FDF\u3040-\u318F\u31A0-\u31BF\u31F0-\u31FF\u3400-\u4DB5\u4E00-\u9FFF\uA960-\uA97F\uAC00-\uD7FF]'), behavior='isolated'),
        pre_tokenizers.Split(Regex(' *(\w+|[^\w\s]+)'), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])
    tokenizer.decoder = decoders.Sequence([
        decoders.ByteLevel(),
    ])
    trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>"], 
                         vocab_size=32000,
                         initial_alphabet=pre_tokenizers.ByteLevel().alphabet())

    # tokenizer.train_from_iterator(text_iterator(), trainer=trainer) # memory leak
    # tokenizer.train(files=['/home/work/disk/vision/retriever/text.txt'], trainer=trainer)

    target_dirs = ['english_gutenberg', 'english_wikipedia', 'english_books3', 'english_c4'] #'english_gutenberg', 'english_wikipedia', 'english_books3', 'english_c4'
    train_files = get_all_files(data_root, target_dirs, ratio=0.35)
    tokenizer.train(files=train_files, trainer=trainer) # 325242 Mo, 69323261 pairs

    save_path = os.path.join(model_root, 'tokenizer.json')
    tokenizer.save(save_path)
    output = tokenizer.encode("Hello, y'all! How are you 😁您 Ã?\nwhat is the meal today")
    print(output.tokens)
    output = tokenizer.decode(output.ids)
    print(output)

def bpe_validate(model_root, model_dir, model_name, data_root):
    # tokenizer = Tokenizer.from_file(os.path.join(model_root, model_dir, model_name))
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_root, model_dir, model_name), eos_token='</s>')

    query = "Hello, y'all! How are you 😁\nwhat is the weather like today?\nThis year is 2023"
    query2 = "what is your name"
    output = tokenizer([query, query2], max_length=10, truncation=False, stride=2, return_overflowing_tokens=True, return_length=True)
    # print(output.tokens)
    # output = tokenizer.decode(output.ids)
    print(output)

    # c4_dir = '/home/work/disk/language-data/english_c4'
    # c4_files = list(map(lambda x: os.path.join(c4_dir, x), os.listdir(c4_dir)))
    # c4_files = list(filter(lambda x: 'validation' in x, c4_files))
    # c4_files.sort()
    # # print(c4_file)

    # for file in c4_files:
    #     data = open(file, 'r').readlines()
    #     char_knt = sum(list(map(lambda x: len(x), data)))
    #     word_knt = list(filter(lambda x: x.strip(), data))
    #     word_knt = sum(list(map(lambda x: len(x.split(' ')), word_knt)))
    #     output = tokenizer(data)['input_ids']
    #     output = sum(list(map(lambda x: len(x), output)))
    #     print(f"char: {char_knt}, word: {word_knt}, token: {output}, c2t: {char_knt/output:.3f}, w2t: {word_knt/output:.3f}")

        # output = tokenizer.encode(''.join(data))
        # print(f"char: {char_knt}, word: {word_knt}, token: {len(output)}, c2t: {char_knt/len(output):.3f}, w2t: {word_knt/len(output):.3f}")

    # char: 97143672, word: 16298401, token: 22590465, c2t: 4.300, w2t: 0.721
    # char: 97955635, word: 16419664, token: 22693871, c2t: 4.316, w2t: 0.724
    # char: 99279297, word: 16661649, token: 23020616, c2t: 4.313, w2t: 0.724
    # 训练token 300G，70B tokens
    # 训练数据868G，200B tokens

if __name__ == "__main__":
    model_root = "/home/work/disk/vision/retriever"
    model_dir = "tokenizer_models"
    model_name="tokenizer_300G.json"
    # model_name="tokenizer_u32.json"
    data_root = '/home/work/disk/language-data'

    # hugging face的BPE训练出来有问题，一些常用词合成不出来，如"the"->"t","h","e"
    # 查了下，应该是overflow的问题
    # english_gutenberg 179465157
    # english_wikipedia 168020500
    # english_books3 863365173
    # 一共是12,1085,0830 > uint32 的 1/4
    # he也没有，h + e 估计超了
    # count_word_the()

    # BPE tokenizer
    # 把rust源码中的u32->u64, i32->i64，解决溢出的问题
    # pip install -e . 安装到pip里，version 0.13.3
    # bpe_tokenization(model_root, data_root)

    # 验证模型是否有问题
    bpe_validate(model_root, model_dir, model_name, data_root)