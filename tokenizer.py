# 读取parquet文件
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import Qwen2ForCausalLM 

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

df = pd.read_parquet('./data/train-00000-of-00192.parquet')
# 判断df中的text是否为空字符串
empty_text = df['text'].apply(lambda x: x == '')
# 删除df中text为空字符串的行
df = df[~empty_text]

print(df.head())

trainer.train(df['text'])
