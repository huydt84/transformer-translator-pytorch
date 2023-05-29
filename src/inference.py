from unicodedata import name
from tqdm import tqdm
from constants import *
from custom_data import *
from transformer import *
from data_structure import *
from torch import nn
from Sophia import SophiaG 

import torch
import sys, os
import numpy as np
import argparse
import datetime
import copy
import heapq
import sentencepiece as spm
from underthesea import text_normalize
from torchmetrics import BLEUScore

def inference(model, input_sentence, method):
    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.load(f"{SP_DIR}/{src_model_prefix}.model")
    trg_sp.load(f"{SP_DIR}/{trg_model_prefix}.model")

    tokenized = src_sp.EncodeAsIds(input_sentence)
    src_data = torch.LongTensor(pad_or_truncate(tokenized + [eos_id])).unsqueeze(0).to(device) # (1, L)
    e_mask = (src_data != pad_id).unsqueeze(1).to(device) # (1, 1, L)

    e_output = model.encoder(src_data, e_mask) # (1, L, d_model)

    if method == 'greedy':
        # print("Greedy decoding selected.")
        result = greedy_search(model, e_output, e_mask, trg_sp)
    elif method == 'beam':
        # print("Beam search selected.")
        result = beam_search(model, e_output, e_mask, trg_sp)
    else:
        raise ValueError("Method unsupported. Only support 'greedy' and 'beam' search")

    return result

def greedy_search(model, e_output, e_mask, trg_sp):
    last_words = torch.LongTensor([pad_id] * seq_len).to(device) # (L)
    last_words[0] = sos_id # (L)
    cur_len = 1

    for i in range(seq_len):
        d_mask = (last_words.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

        output = model.decoder(
            last_words.unsqueeze(0),
            e_output,
            e_mask,
            d_mask
        ) # (1, L, trg_vocab_size)

        output = torch.argmax(output, dim=-1) # (1, L)
        last_word_id = output[0][i].item()
        
        if i < seq_len-1:
            last_words[i+1] = last_word_id
            cur_len += 1
        
        if last_word_id == eos_id:
            break

    if last_words[-1].item() == pad_id:
        decoded_output = last_words[1:cur_len].tolist()
    else:
        decoded_output = last_words[1:].tolist()
    decoded_output = trg_sp.decode_ids(decoded_output)
    
    return decoded_output

def beam_search(model, e_output, e_mask, trg_sp):
    cur_queue = PriorityQueue()
    for k in range(beam_size):
        cur_queue.put(BeamNode(sos_id, -0.0, [sos_id]))
    
    finished_count = 0
    
    for pos in range(seq_len):
        new_queue = PriorityQueue()
        for k in range(beam_size):
            node = cur_queue.get()
            if node.is_finished:
                new_queue.put(node)
            else:
                trg_input = torch.LongTensor(node.decoded + [pad_id] * (seq_len - len(node.decoded))).to(device) # (L)
                d_mask = (trg_input.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
                nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)
                nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
                d_mask = d_mask & nopeak_mask # (1, L, L) padding false
                
                output = model.decoder(
                    trg_input.unsqueeze(0),
                    e_output,
                    e_mask,
                    d_mask
                ) # (1, L, trg_vocab_size)
                
                output = torch.topk(output[0][pos], dim=-1, k=beam_size)
                last_word_ids = output.indices.tolist() # (k)
                last_word_prob = output.values.tolist() # (k)
                
                for i, idx in enumerate(last_word_ids):
                    new_node = BeamNode(idx, -(-node.prob + last_word_prob[i]), node.decoded + [idx])
                    if idx == eos_id:
                        new_node.prob = new_node.prob / float(len(new_node.decoded))
                        new_node.is_finished = True
                        finished_count += 1
                    new_queue.put(new_node)
        
        cur_queue = copy.deepcopy(new_queue)
        
        if finished_count == beam_size:
            break
    
    decoded_output = cur_queue.get().decoded
    
    if decoded_output[-1] == eos_id:
        decoded_output = decoded_output[1:-1]
    else:
        decoded_output = decoded_output[1:]
        
    return trg_sp.decode_ids(decoded_output)

def calculate_bleu(model, method, list_src, list_trg):
    references, predictions = [], []
    for (src, trg) in tqdm(zip(list_src, list_trg)):
        pred = inference(model, src, method)
        pred = text_normalize(pred)
        predictions.append(pred)
        trg = text_normalize(trg)
        references.append([trg])      
    bleu = BLEUScore()
    return bleu(predictions, references) 

if __name__=='__main__':
    model = Transformer(src_vocab_size=sp_src_vocab_size, trg_vocab_size=sp_trg_vocab_size, d_model=d_model).to(device)

    print("Loading checkpoint...")
    checkpoint = torch.load("saved_model_sophia/best_ckpt.tar")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    input_sentence = "Hello my name is John"
    print(inference(model, input_sentence, method="greedy"))

    print(f"Getting source/target data...")
    with open(f"{DATA_DIR}/{SRC_DIR}/{SRC_TEST_NAME}", 'r') as f:
        src_text_list = f.readlines()

    with open(f"{DATA_DIR}/{TRG_DIR}/{TRG_TEST_NAME}", 'r') as f:
        trg_text_list = f.readlines()

    print(calculate_bleu(model, "greedy", src_text_list, trg_text_list))

    

