import sentencepiece as spm
import numpy as np
from constants import *
from data_structure import *
from tqdm import tqdm
import onnxruntime
import copy
import os

class Translator():
    def __init__(self, session) -> None:
        self.encoder, self.decoder = session
        self.src_sp = spm.SentencePieceProcessor()
        self.trg_sp = spm.SentencePieceProcessor()
        self.src_sp.load(f"data/sp/src_sp.model")
        self.trg_sp.load(f"data/sp/trg_sp.model")
        

    def translate(self, input_sentence, method="greedy"):
        tokenized = self.src_sp.EncodeAsIds(input_sentence)
        src = np.expand_dims(self.pad_or_truncate(tokenized + [eos_id]), axis=0).astype('int64') # (1, L)
        e_mask = np.expand_dims((src != pad_id), axis=1) # (1, 1, L)
       
        e_output_input = { self.encoder.get_inputs()[0].name: src, self.encoder.get_inputs()[1].name: e_mask}
        e_output = self.encoder.run(None, e_output_input)[0]
        
        if method == "greedy":
            result = self.greedy_search(e_output, e_mask)
        elif method == 'beam':
        # print("Beam search selected.")
            result = self.beam_search(e_output, e_mask)
        else:
            raise ValueError("Method unsupported. Only support 'greedy' and 'beam' search")

        return result
        
    def greedy_search(self, e_output, e_mask):
        last_words = [pad_id] * seq_len
        last_words[0] = sos_id
        cur_len = 1

        for i in range(seq_len):
            lw_expand = np.expand_dims(last_words, axis=0)
            d_mask = np.expand_dims((lw_expand != pad_id), axis=1) # (1, 1, L)
            nopeak_mask = np.ones((1, seq_len, seq_len)).astype('bool')
            nopeak_mask = np.tril(nopeak_mask) # (1, L, L) to triangular shape
            d_mask = d_mask & nopeak_mask # (1, L, L) padding false

            decoder_input = {self.decoder.get_inputs()[0].name: lw_expand,
                                    self.decoder.get_inputs()[1].name: e_output,
                                    self.decoder.get_inputs()[2].name: e_mask,
                                    self.decoder.get_inputs()[3].name: d_mask}
            decoder_output = self.decoder.run(None, decoder_input)[0] # (1, L, trg_vocab_size)

            output = np.argmax(decoder_output, axis=-1)
            last_word_id = output[0][i].item()

            if i < seq_len-1:
                last_words[i+1] = last_word_id
                cur_len += 1
            
            if last_word_id == eos_id:
                break

        if last_words[-1] == pad_id:
            decoded_output = last_words[1:cur_len]
        else:
            decoded_output = last_words[1:]
        decoded_output = self.trg_sp.decode_ids(decoded_output)
        
        return decoded_output

    def beam_search(self, e_output, e_mask):
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
                    trg_input = node.decoded + [pad_id] * (seq_len - len(node.decoded)) # (L)
                    trg_input_expand = np.expand_dims(trg_input, axis=0) # (1, L)
                    d_mask = np.expand_dims((trg_input_expand != pad_id), axis=1) # (1, 1, L)
                    nopeak_mask = np.ones((1, seq_len, seq_len)).astype('bool')
                    nopeak_mask = np.tril(nopeak_mask) # (1, L, L) to triangular shape
                    d_mask = d_mask & nopeak_mask # (1, L, L) padding false
                    
                    decoder_input = {self.decoder.get_inputs()[0].name: trg_input_expand,
                                    self.decoder.get_inputs()[1].name: e_output,
                                    self.decoder.get_inputs()[2].name: e_mask,
                                    self.decoder.get_inputs()[3].name: d_mask}
                    output = self.decoder.run(None, decoder_input)[0] # (1, L, trg_vocab_size)

                    # output = self.model.decoder(
                    #     trg_input_expand,
                    #     e_output,
                    #     e_mask,
                    #     d_mask
                    # ) # (1, L, trg_vocab_size)
                    
                    output_prob, output_ind = self.topk(output[0][pos], k=beam_size, axis=-1)
                    last_word_ids = output_ind.tolist() # (k)
                    last_word_prob = output_prob.tolist() # (k)
                    
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
            
        return self.trg_sp.decode_ids(decoded_output)

    def pad_or_truncate(self, tokenized_text):
        if len(tokenized_text) < seq_len:
            left = seq_len - len(tokenized_text)
            padding = [pad_id] * left
            tokenized_text += padding
        else:
            tokenized_text = tokenized_text[:seq_len]

        return tokenized_text

    def topk(self, array, k, axis=-1, sorted=True):
        # Use np.argpartition is faster than np.argsort, but do not return the values in order
        # We use array.take because you can specify the axis
        partitioned_ind = (
            np.argpartition(array, -k, axis=axis)
            .take(indices=range(-k, 0), axis=axis)
        )
        # We use the newly selected indices to find the score of the top-k values
        partitioned_scores = np.take_along_axis(array, partitioned_ind, axis=axis)
        
        if sorted:
            # Since our top-k indices are not correctly ordered, we can sort them with argsort
            # only if sorted=True (otherwise we keep it in an arbitrary order)
            sorted_trunc_ind = np.flip(
                np.argsort(partitioned_scores, axis=axis), axis=axis
            )
            
            # We again use np.take_along_axis as we have an array of indices that we use to
            # decide which values to select
            ind = np.take_along_axis(partitioned_ind, sorted_trunc_ind, axis=axis)
            scores = np.take_along_axis(partitioned_scores, sorted_trunc_ind, axis=axis)
        else:
            ind = partitioned_ind
            scores = partitioned_scores
        
        return scores, ind

if __name__=='__main__':
    EP_list = ['CUDAExecutionProvider', "CPUExecutionProvider"]
    
    encoder = onnxruntime.InferenceSession(os.path.join(ONNX_DIR, "encoder.onnx"), providers=EP_list)
    decoder = onnxruntime.InferenceSession(os.path.join(ONNX_DIR, "decoder.onnx"), providers=EP_list)
    
    session = (encoder, decoder)

    translator = Translator(session)
    
    with open("data/trg/jaen/train.en", "r", encoding="utf-8") as f:
        en_lst = f.readlines()[:300000]
        
    vi_lst = []    
    for s in tqdm(en_lst):
        vi_lst.append(translator.translate(s.strip(), method="greedy"))
        
    with open("data/trg/jaen/train.vi", "w", encoding="utf-8") as f:
        f.write("\n".join(vi_lst))