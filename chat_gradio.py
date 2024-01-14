import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import time
import torch
import gradio as gr
from functools import partial
from model import retriever
from config import RetrieverConfig_medium
from transformers import PreTrainedTokenizerFast

def response_func(message, history, model, tokenizer):
    begin_time = time.time()
    query_length = len(message)
    for q, a in history:
        query_length += len(q) + len(a)
    print(message, history)
    response = model.chat(tokenizer, message, history)
    print(f"query_length: {query_length}, response_length: {len(response)}, consume {time.time() - begin_time:.1f}s")
    return response

def main(gpu, arch, dtype, model_root, model_path, tokenizer_path):
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    tokenizer_path = os.path.join(model_root, tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    model_path = os.path.join(model_root, model_path)
    model = arch(device, ptdtype, config, pretrained=True, model_path=model_path, flash=True)
    model.cuda(gpu)
    # model = torch.compile(model)
    model = model.eval()
    
    demo = gr.ChatInterface(partial(response_func, model=model, tokenizer=tokenizer), title="Chat Bot")
    demo.launch(server_port=7870)

if __name__ == "__main__":
    arch = retriever
    config = RetrieverConfig_medium()
    dtype = "bfloat16"
    model_root = "/home/work/disk/vision/retriever"
    # model_path = "checkpoint/retriever_35M_42B_loss3.29.pth.tar"
    # model_path = "checkpoint/retriever_110M_100B_loss2.94.pth.tar"
    model_path = "checkpoint/instruct_retriever_medium.pth.tar"
    tokenizer_path = "tokenizer_models/tokenizer_v2_600G.json"
    main(0, arch, dtype, model_root, model_path, tokenizer_path)