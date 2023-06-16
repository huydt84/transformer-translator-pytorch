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
import pytorch_warmup as warmup
from transformers import get_cosine_schedule_with_warmup


class Manager():
    def __init__(self, is_train=True, ckpt_name=None):
        if is_train:
            # Load loss function
            print("Loading loss function...")
            self.criterion = nn.NLLLoss()

            # Load dataloaders
            print("Loading dataloaders...")
            self.train_loader = get_data_loader(SRC_TRAIN_NAME, TRG_TRAIN_NAME)
            self.valid_loader = get_data_loader(SRC_VALID_NAME, TRG_VALID_NAME)
        
        # Load Transformer model & Adam optimizer
        print("Loading Transformer model & Adam optimizer...")
        self.model = Transformer(src_vocab_size=sp_src_vocab_size, trg_vocab_size=sp_trg_vocab_size, d_model=d_model).to(device)
        # self.optim = SophiaG(self.model.parameters(), lr=learning_rate, betas=(0.965, 0.99), rho = 0.03, weight_decay=1e-1)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, betas=betas, eps=eps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optim,
            num_warmup_steps=warmup_step,
            num_training_steps=len(self.train_loader) * num_epochs,
        )

        self.scaler = torch.cuda.amp.GradScaler()
        self.best_loss = sys.float_info.max
        
        print(sum(p.numel() for p in self.model.parameters()))

        if ckpt_name is not None:
            assert os.path.exists(f"{ckpt_dir}/{ckpt_name}"), f"There is no checkpoint named {ckpt_name}."

            print("Loading checkpoint...")
            checkpoint = torch.load(f"{ckpt_dir}/{ckpt_name}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.best_loss = checkpoint['loss']
        else:
            print("Initializing the model...")
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                    
        print("Setting finished.")

    def train(self):
        print("Training starts.")

        for epoch in range(1, num_epochs+1):
            self.model.train()
            
            train_losses = []
            start_time = datetime.datetime.now()

            for i, batch in tqdm(enumerate(self.train_loader)):
                src_input, trg_input, trg_output = batch
                src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

                e_mask, d_mask = self.make_mask(src_input, trg_input)

                output = self.model(src_input, trg_input, e_mask, d_mask) # (B, L, vocab_size)

                trg_output_shape = trg_output.shape

                self.optim.zero_grad() 
                with torch.autocast(device_type='cuda', dtype=torch.float16):   
                    loss = self.criterion(
                        output.view(-1, sp_src_vocab_size),
                        trg_output.view(trg_output_shape[0] * trg_output_shape[1])
                    )

                self.scaler.scale(loss).backward() 
                self.scaler.unscale_(self.optim)
            
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.scaler.step(self.optim)
                self.scaler.update()
                
                self.scheduler.step()

                train_losses.append(loss.item())
                
                del src_input, trg_input, trg_output, e_mask, d_mask, output
                torch.cuda.empty_cache()

            end_time = datetime.datetime.now()
            training_time = end_time - start_time
            seconds = training_time.seconds
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60

            mean_train_loss = np.mean(train_losses)
            print(f"#################### Epoch: {epoch} ####################")
            print(f"Train loss: {mean_train_loss} || One epoch training time: {hours} hrs {minutes} mins {seconds}secs")

            valid_loss, valid_time = self.validation()
            
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss

            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
                
            state_dict = {
                'model_state_dict': self.model.state_dict(),
                'optim_state_dict': self.optim.state_dict(),
                'loss': valid_loss
            }
            torch.save(state_dict, f"{ckpt_dir}/ckpt_{epoch}_javi2.tar")

            print(f"Best valid loss: {self.best_loss}")
            print(f"Valid loss: {valid_loss} || One epoch validating time: {valid_time}")

        print(f"Training finished!")
        
    def validation(self):
        print("Validation processing...")
        self.model.eval()
        
        valid_losses = []
        start_time = datetime.datetime.now()

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.valid_loader)):
                src_input, trg_input, trg_output = batch
                src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

                e_mask, d_mask = self.make_mask(src_input, trg_input)

                output = self.model(src_input, trg_input, e_mask, d_mask) # (B, L, vocab_size)

                trg_output_shape = trg_output.shape
                loss = self.criterion(
                    output.view(-1, sp_trg_vocab_size),
                    trg_output.view(trg_output_shape[0] * trg_output_shape[1])
                )

                valid_losses.append(loss.item())

                del src_input, trg_input, trg_output, e_mask, d_mask, output
                torch.cuda.empty_cache()

        end_time = datetime.datetime.now()
        validation_time = end_time - start_time
        seconds = validation_time.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        mean_valid_loss = np.mean(valid_losses)
        
        return mean_valid_loss, f"{hours}hrs {minutes}mins {seconds}secs"

    def inference(self, input_sentence, method):
        self.model.eval()

        src_sp = spm.SentencePieceProcessor()
        trg_sp = spm.SentencePieceProcessor()
        src_sp.load(f"{SP_DIR}/{src_model_prefix}.model")
        trg_sp.load(f"{SP_DIR}/{trg_model_prefix}.model")

        print("Preprocessing input sentence...")
        tokenized = src_sp.EncodeAsIds(input_sentence)
        src_data = torch.LongTensor(pad_or_truncate(tokenized + [eos_id])).unsqueeze(0).to(device) # (1, L)
        e_mask = (src_data != pad_id).unsqueeze(1).to(device) # (1, 1, L)

        start_time = datetime.datetime.now()

        print("Encoding input sentence...")
        e_output = self.model.encoder(src_data, e_mask) # (1, L, d_model)

        if method == 'greedy':
            print("Greedy decoding selected.")
            result = self.greedy_search(e_output, e_mask, trg_sp)
        elif method == 'beam':
            print("Beam search selected.")
            result = self.beam_search(e_output, e_mask, trg_sp)

        end_time = datetime.datetime.now()

        total_inference_time = end_time - start_time
        seconds = total_inference_time.seconds
        minutes = seconds // 60
        seconds = seconds % 60

        print(f"Input: {input_sentence}")
        print(f"Result: {result}")
        print(f"Inference finished! || Total inference time: {minutes}mins {seconds}secs")
        
    def greedy_search(self, e_output, e_mask, trg_sp):
        last_words = torch.LongTensor([pad_id] * seq_len).to(device) # (L)
        last_words[0] = sos_id # (L)
        cur_len = 1

        for i in range(seq_len):
            d_mask = (last_words.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
            nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)  # (1, L, L)
            nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
            d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

            output = self.model.decoder(
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
    
    def beam_search(self, e_output, e_mask, trg_sp):
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
                    
                    output = self.model.decoder(
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
        

    def make_mask(self, src_input, trg_input):
        e_mask = (src_input != pad_id).unsqueeze(1)  # (B, 1, L)
        d_mask = (trg_input != pad_id).unsqueeze(1)  # (B, 1, L)

        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

        return e_mask, d_mask


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help="train or inference?")
    parser.add_argument('--ckpt_name', required=False, help="best checkpoint file")
    parser.add_argument('--input', type=str, required=False, help="input sentence when inferencing")
    parser.add_argument('--decode', type=str, required=False, default="greedy", help="greedy or beam?")

    args = parser.parse_args()

    if args.mode == 'train':
        if args.ckpt_name is not None:
            manager = Manager(is_train=True, ckpt_name=args.ckpt_name)
        else:
            manager = Manager(is_train=True)

        manager.train()
    elif args.mode == 'inference':
        assert args.ckpt_name is not None, "Please specify the model file name you want to use."
        assert args.input is not None, "Please specify the input sentence to translate."
        assert args.decode == 'greedy' or args.decode =='beam', "Please specify correct decoding method, either 'greedy' or 'beam'."
       
        manager = Manager(is_train=False, ckpt_name=args.ckpt_name)
        manager.inference(args.input, args.decode)

    else:
        print("Please specify mode argument either with 'train' or 'inference'.")