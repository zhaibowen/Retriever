import os
import sys
from mmlu_categories import subcategories

# find . -type f -empty -delete

data_path = "/home/work/disk/language-data/mmlu/data/auxiliary_train"

for sc in subcategories.keys():
    p = os.path.join(data_path, sc+'_train.parquet')
    if os.path.exists(p):
        continue

    x = "https://huggingface.co/datasets/cais/mmlu/resolve/refs%2Fconvert%2Fparquet/{}/auxiliary_train/0000.parquet".format(sc)

    command = f"wget {x} -O {p}"
    print(command)
    os.system(command)