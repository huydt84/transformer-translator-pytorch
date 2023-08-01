from unicodedata import name
from tqdm import tqdm
from constants import *
from custom_data import *
import sentencepiece as spm
import random

NUM_GEN = 500000
src_sp = spm.SentencePieceProcessor()
trg_sp = spm.SentencePieceProcessor()
src_sp.load(f"{SP_DIR}/{src_model_prefix}.model")
trg_sp.load(f"{SP_DIR}/{trg_model_prefix}.model")

with open("data/src/javi/train_aug_mix.ja", "r", encoding="utf-8") as f:
    ja = f.readlines()

with open("data/trg/javi/train_aug_mix.vi", "r", encoding="utf-8") as f:
    vi = f.readlines()

ordinal_number_1 = random.sample(range(0, len(ja)), NUM_GEN)
ordinal_number_2 = random.sample(range(0, len(ja)), NUM_GEN)

ordinal_number_javi = [[i, j] for (i, j) in zip(ordinal_number_1, ordinal_number_2)]
for (i, j) in tqdm(ordinal_number_javi):
    sentence_ja = ja[i].strip() + "。" + ja[j].strip()
    sentence_ja = sentence_ja.replace("。。", "。")

    tokenized_ja = src_sp.EncodeAsIds(sentence_ja)

    symbol_eos = (",", ":", ";", "-", "_", ".", "?", "!")
    if vi[i].strip().endswith(symbol_eos):
        sentence_vi = vi[i].strip() + " " + vi[j].strip().capitalize()
    else:
        sentence_vi = vi[i].strip() + ". " + vi[j].strip().capitalize()

    tokenized_vi = trg_sp.EncodeAsIds(sentence_vi)

    if (len(tokenized_ja)<=127 and len(tokenized_vi)<=126):
        with open("merge_train.ja", "a", encoding="utf-8") as f_ja:
            f_ja.write(sentence_ja.strip())
            f_ja.write("\n")
        with open("merge_train.vi", "a", encoding="utf-8") as f_vi:
            f_vi.write(sentence_vi.strip())
            f_vi.write("\n")
    

    

