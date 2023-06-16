import torch

# Path or parameters for data
DATA_DIR = 'data'
SP_DIR = f'{DATA_DIR}/sp'
SRC_DIR = 'src/javi'
TRG_DIR = 'trg/javi'
ONNX_DIR = 'onnx_test'
SRC_RAW_DATA_NAME = 'raw_data.src'
TRG_RAW_DATA_NAME = 'raw_data.trg'
SRC_TRAIN_NAME = 'train_aug_mix.ja'
SRC_VALID_NAME = 'dev2010.ja-vi.ja'
SRC_TEST_NAME = 'tatoeba.ja'
TRG_TRAIN_NAME = 'train_aug_mix.vi'
TRG_VALID_NAME = 'dev2010.ja-vi.vi'
TRG_TEST_NAME = 'tatoeba.vi'

# Parameters for sentencepiece tokenizer
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
src_model_prefix = 'src_sp'
trg_model_prefix = 'trg_sp'
sp_src_vocab_size = 64000
sp_trg_vocab_size = 64000
character_coverage = 1.0
model_type = 'bpe'

# Parameters for Transformer & training
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
# device = 'cpu'
learning_rate = 1e-4
betas = (0.9, 0.98)
eps = 1e-9
warmup_step = 200
batch_size = 16
seq_len = 128
num_heads = 8
encoder_num_layers = 6
decoder_num_layers = 1
d_model = 512
d_ff = 1024
d_k = d_model // num_heads
drop_out_rate = 0.1
num_epochs = 18
beam_size = 3
ckpt_dir = "sq128_8_6_1_512_1024"
