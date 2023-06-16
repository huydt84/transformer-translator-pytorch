from constants import *
from tqdm import tqdm

import os
import sentencepiece as spm

train_frac = 0.8

def train_sp(is_src=True):
    template = "--input={} \
                --pad_id={} \
                --bos_id={} \
                --eos_id={} \
                --unk_id={} \
                --model_prefix={} \
                --vocab_size={} \
                --character_coverage={} \
                --model_type={}"

    if is_src:
        this_input_file = f"{DATA_DIR}/{SRC_RAW_DATA_NAME}"
        this_model_prefix = f"{SP_DIR}/{src_model_prefix}"
        vocab_size = sp_src_vocab_size
    else:
        this_input_file = f"{DATA_DIR}/{TRG_RAW_DATA_NAME}"
        this_model_prefix = f"{SP_DIR}/{trg_model_prefix}"
        vocab_size = sp_trg_vocab_size

    config = template.format(this_input_file,
                            pad_id,
                            sos_id,
                            eos_id,
                            unk_id,
                            this_model_prefix,
                            vocab_size,
                            character_coverage,
                            model_type)

    print(config)

    if not os.path.isdir(SP_DIR):
        os.makedirs(SP_DIR)

    print(spm)
    spm.SentencePieceTrainer.Train(config)


if __name__=='__main__':
    train_sp(is_src=True)
    # train_sp(is_src=False)

    