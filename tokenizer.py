import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import Qwen2ForCausalLM
from typing import Optional
import torch
import torch.nn as nn
from tokenizers.pre_tokenizers import Whitespace, Sequence, Split, ByteLevel

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# 读取parquet文件
df = pd.read_parquet('./data/train-00000-of-00192.parquet')
# 判断df中的text是否为空字符串
empty_text = df['text'].apply(lambda x: x == '')
# 删除df中text为空字符串的行
df = df[~empty_text]

# 查看df的前5行
print(df.head())
