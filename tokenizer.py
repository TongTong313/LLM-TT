import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import Qwen2ForCausalLM
from typing import Optional
import torch
import torch.nn as nn
from tokenizers.pre_tokenizers import Whitespace, Sequence, Split, ByteLevel
from tokenizers.normalizers import NFC
from tokenizers import Regex

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# 读取parquet文件
# df = pd.read_parquet('./data/train-00000-of-00192.parquet')

# 随便给一段话
# 创建一段包含需要标准化的中文文本
text = "中国是一个伟大的国家，它有着悠久的历史。我今天心情很好，想学习一会"
# text = "hello world, I'm a student. I'm 20 years old"

pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
pre_tokenizer_split = Split(pattern=Regex(pattern), behavior='isolated')
pre_tokenizer_bytelevel = ByteLevel(add_prefix_space=False,
                                    trim_offsets=False,
                                    use_regex=False)

pre_tokenizer = Sequence([pre_tokenizer_split, pre_tokenizer_bytelevel])

text_1 = pre_tokenizer.pre_tokenize_str(text)
print(text_1)

# 表示16进制e4 b8 ad Unicode字符就是ä¸Ń  ä ¸
a = 0xe4
b = 0xb8
c = 0xad
print(chr(a), chr(b), chr(c))

# normalizer = NFC()

# 标准化之前的文本
# print("原始文本:", text)

# 标准化之后的文本
# normalized_text = normalizer.normalize_str(text)

# print("NFC标准化后:", normalized_text)

# 读取txt文件
# df = pd.read_csv('./data/tokenizer_train_data.txt', sep='\t', names=['text'])

# # 取一条数据
# text = df['text'][0]
# print(text)

# tokenizer.pre_tokenizer = Whitespace()

# # 判断df中的text是否为空字符串
# empty_text = df['text'].apply(lambda x: x == '')
# # 删除df中text为空字符串的行
# df = df[~empty_text]

# # 查看df的前5行
# print(df.head())
