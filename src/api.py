# Importing Necessary modules
import os
from fastapi import FastAPI
from constants import *
import onnxruntime
import sentencepiece as spm
import numpy as np

# Declaring our FastAPI instance
app = FastAPI()
src_sp = spm.SentencePieceProcessor()
trg_sp = spm.SentencePieceProcessor()
encoder = onnxruntime.InferenceSession(os.path.join(ONNX_DIR, "encoder.onnx"))
decoder = onnxruntime.InferenceSession(os.path.join(ONNX_DIR, "decoder.onnx"))
src_sp.load(f"{SP_DIR}/{src_model_prefix}.model")
trg_sp.load(f"{SP_DIR}/{trg_model_prefix}.model")

def translate(input_sentence):
    tokenized = src_sp.EncodeAsIds(input_sentence)
    src = np.expand_dims(pad_or_truncate(tokenized + [eos_id]), axis=0).astype('int64')  # (1, L)
    e_mask = np.expand_dims((src != pad_id), axis=1)  # (1, 1, L)
    e_output_input = {encoder.get_inputs()[0].name: src, encoder.get_inputs()[1].name: e_mask}
    e_output = encoder.run(None, e_output_input)[0]
    result = greedy_search(e_output, e_mask)
    
    return result

def pad_or_truncate(tokenized_text):
    if len(tokenized_text) < seq_len:
        left = seq_len - len(tokenized_text)
        padding = [pad_id] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[:seq_len]

    return tokenized_text
def greedy_search(e_output, e_mask):
    last_words = [pad_id] * seq_len
    last_words[0] = sos_id
    cur_len = 1

    for i in range(seq_len):
        lw_expand = np.expand_dims(last_words, axis=0)
        d_mask = np.expand_dims((lw_expand != pad_id), axis=1)  # (1, 1, L)
        nopeak_mask = np.ones((1, seq_len, seq_len)).astype('bool')
        nopeak_mask = np.tril(nopeak_mask)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

        decoder_input = {decoder.get_inputs()[0].name: lw_expand,
                         decoder.get_inputs()[1].name: e_output,
                         decoder.get_inputs()[2].name: e_mask,
                         decoder.get_inputs()[3].name: d_mask}
        decoder_output = decoder.run(None, decoder_input)[0]  # (1, L, trg_vocab_size)
        output = np.argmax(decoder_output, axis=-1)
        last_word_id = output[0][i].item()

        if i < seq_len - 1:
            last_words[i + 1] = last_word_id
            cur_len += 1
        if last_word_id == eos_id:
            break

    if last_words[-1] == pad_id:
        decoded_output = last_words[1:cur_len]

    else:
        decoded_output = last_words[1:]
    decoded_output = trg_sp.decode_ids(decoded_output)

    return decoded_output

# Defining path operation for root endpoint
@app.get("/predict/{text}")
async def predict(text: str):
    result = translate(text)
    return result