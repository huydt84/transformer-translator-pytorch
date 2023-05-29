# transformer-translator-pytorch
This is a machine translation project using the basic **Transformer** introduced in *Attention is all you need*[[1]](#1).

The original author used English-French corpus provided by "European Parliament Proceedings Parallel Corpus 1996-2011"[[2]](#2).

I used PhoMT [[3]](#3) dataset for English-Vietnamese translation task

# Update 29/5/2023:
- Reformat model to only encoder-decoder
- Mixed-precision model training
- Add inference with Bleu score calculation
- Convert model to ONNX representation

<br/>

---

### Configurations

You can set various hyperparameters in `src/constants.py` file.

The description of each variable is as follows.

<br/>

**Parameters for data**

Argument | Type | Description | Default
---------|------|---------------|------------
`DATA_DIR` | `str` | Name of the parent directory where data files are stored. | `data` 
`SP_DIR` | `str` | Path for the directory which contains the sentence tokenizers and vocab files. | `f'{DATA_DIR}/sp'` 
`SRC_DIR` | `str` | Name of the directory which contains the source data files. | `src/envi` 
`TRG_DIR` | `str` | Name of the directory which contains the target data files. | `trg/envi` 
`ONNX_DIR` | `str` | Name of the directory which contains ONNX models | `onnx_test`
`SRC_RAW_DATA_NAME` | `str` | Name of the source raw data file. | `raw_data.src` 
`TRG_RAW_DATA_NAME` | `str` | Name of the target raw data file. | `raw_data.trg` 
`SRC_TRAIN_NAME` | `str` | Name of the source train data file. | `train.en` 
`SRC_VALID_NAME` | `str` | Name of the source validation data file. | `dev.en` 
`SRC_TEST_NAME` | `str` | Name of the source test data file. | `test.en` 
`TRG_TRAIN_NAME` | `str` | Name of the target train data file. | `train.vi` 
`TRG_VALID_NAME` | `str` | Name of the target validation data file. | `dev.vi` 
`TRG_TEST_NAME` | `str` | Name of the target test data file. | `test.vi` 

<br/>

**Parameters for Sentencepiece**

| Argument             | Type    | Description                                                  | Default   |
| -------------------- | ------- | ------------------------------------------------------------ | --------- |
| `pad_id`             | `int`   | The index of pad token.                                      | `0`       |
| `sos_id`             | `int`   | The index of start token.                                    | `1`       |
| `eos_id`             | `int`   | The index of end token.                                      | `2`       |
| `unk_id`             | `int`   | The index of unknown token.                                  | `3`       |
| `src_model_prefix`   | `str`   | The file name prefix for the source language tokenizer & vocabulary. | `src_sp`  |
| `trg_model_prefix`   | `str`   | The file name prefix for the target language tokenizer & vocabulary. | `trg_sp`  |
| `sp_src_vocab_size`  | `int`   | The size of vocabulary of source language.                   | `64000`   |
| `sp_trg_vocab_size`  | `int`   | The size of vocabulary of target language.                   | `64000`   |
| `character_coverage` | `float` | The value for character coverage (between `0.0` and `1.0`).  | `1.0`     |
| `model_type`         | `str`   | The type of sentencepiece model. (`unigram`, `bpe`, `char`, or `word`) | `bpe` |

<br/>

**Parameters for the transformer & training**

| Argument        | Type           | Description                                                  | Default                                                      |
| --------------- | -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `device`        | `torch.device` | The device type. (CUDA or CPU)                               | `torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')` |
| `learning_rate` | `float`        | The learning rate.                                           | `1e-4`                                                       |
| `betas`         | `tuple`        | Exponential moving average of gradient and its square        | `(0.9, 0.98)`                                                |
| `eps`           | `float`        | Small number added to denominator to prevent divided by 0.   | `1e-4`                                                       |
| `batch_size`    | `int`          | The batch size.                                              | `16`                                                         |
| `seq_len`       | `int`          | The maximum length of a sentence.                            | `256`                                                        |
| `num_heads`     | `int`          | The number of heads for Multi-head attention.                | `8`                                                          |
| `num_layers`    | `int`          | The number of layers in the encoder & the decoder.           | `3`                                                          |
| `d_model`       | `int`          | The size of hidden states in the model.                      | `512`                                                        |
| `d_ff`          | `int`          | The size of intermediate  hidden states in the feed-forward layer. | `1024`                                                       |
| `d_k`           | `int`          | The size of dimension which a single head should take. (Make sure that `d_model` is divided into `num_heads`.) | `d_model // num_heads`                                       |
| `drop_out_rate` | `float`        | The dropout rate.                                            | `0.1`                                                        |
| `num_epochs`    | `int`          | The total number of iterations.                              | `10`                                                         |
| `beam_size`     | `int`          | The beam size. (Only used when the beam search is used at inference time.) | `3`                                                          |
| `ckpt_dir`      | `str`          | The path for saved checkpoints.                              | `saved_model`                                                |

<br/>

<hr style="background: transparent; border: 0.5px dashed;"/>

### How to run

1. Download the dataset from ["PhoMT: A High-Quality and Large-Scale Benchmark Dataset for Vietnamese-English Machine Translation"](https://github.com/VinAIResearch/PhoMT) 
   
   Make `DATA_DIR` directory in the root directory and put raw texts in it. Raw text can be training set, or any corpus in corresponding languages.
   
   Name each ``SRC_RAW_DATA_NAME`` and ``TRG_RAW_DATA_NAME``.

   Save dataset in folder so that the structure of `DATA_DIR` directory likes below. I create `envi` folder to discriminate with other language pairs added in the future
   
   - `data`
     - `src`
       - `envi`
         - `train.en`
         - `dev.en`
         - `test.en`
     - `trg`
       - `envi`         
         - `train.vi`
         - `dev.vi`
         - `test.vi`
     - `raw_data.src`
     - `raw_data.tar`   
   
   Of course, you can use additional datasets and just make sure that the formats/names of raw data files are same as those of above datasets. 

   
   <br/>
   
2. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

3. Run `src/sentencepiece_train.py`.

   ```shell
   python src/sentencepiece_train.py
   ```

   Then there would be `SP_DIR` directory containing two sentencepiece models and two vocab files.

   Each model and vocab files are for source language and target language.

   In default setting, the structure of whole data directory should be like below. 

   - `data`
      - `sp`
         - `src_sp.model`
         - `src_sp.vocab`
         - `tar_sp.model`
         - `tar_sp.vocab`
     - `src`
       - `envi`
         - `train.en`
         - `dev.en`
         - `test.en`
     - `trg`
       - `envi`         
         - `train.vi`
         - `dev.vi`
         - `test.vi`
     - `raw_data.src`
     - `raw_data.tar`  

   You can delete vocab files, I modified the code so that `.vocab` file is not mandatory.

   <br/>

4. Run below command to train a transformer model for machine translation.

   ```shell
   python src/main.py --mode='train' --ckpt_name=CHECKPOINT_NAME
   ```

   - `--mode`: You have to specify the mode among two options, 'train' or 'inference'.
   - `--ckpt_name`: This specify the checkpoint file name. This would be the name of trained checkpoint and you can continue your training with this model in the case of resuming training. If you want to conduct training first, this parameter should be omitted. When testing, this would be the name of the checkpoint you want to test. (default: `None`)

   <br/>

5. Run below command to conduct an inference with the trained model.

   ```shell
   python src/main.py --mode='inference' --ckpt_name=CHECKPOINT_NAME --input=INPUT_TEXT --decode=DECODING_STRATEGY
   ```

   - `--input`: This is an input sequence you want to translate.
   - `--decode`: This makes the decoding algorithm into either greedy method or beam search. Make this parameter 'greedy' or 'beam'.  (default: `greedy`)

   <br/>
   
<hr style="background: transparent; border: 0.5px dashed;"/>

### References

<a id="1">[1]</a> 
*Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008)*. ([http://papers.nips.cc/paper/7181-attention-is-all-you-need](http://papers.nips.cc/paper/7181-attention-is-all-you-need))

<a id="2">[2]</a>
*Koehn, P. (2005, September). Europarl: A parallel corpus for statistical machine translation. In MT summit (Vol. 5, pp. 79-86)*. ([http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.5497&rep=rep1&type=pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.5497&rep=rep1&type=pdf))

<a id="3">[3]</a>
*Long Doan and Linh The Nguyen and Nguyen Luong Tran and Thai Hoang and Dat Quoc Nguyen. (2021). PhoMT: A High-Quality and Large-Scale Benchmark Dataset for Vietnamese-English Machine Translation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (pp. 4495-4503)*. ([https://aclanthology.org/2021.emnlp-main.369/](https://aclanthology.org/2021.emnlp-main.369/))

<br/>

---
