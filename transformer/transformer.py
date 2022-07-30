import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)


import numpy as np
import pickle as pkl
from transformer.modules import Encoder
from transformer.modules import Decoder

def filter_seq(seq):
    chars2remove = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

    return ''.join([c for c in seq if c not in chars2remove])

def lowercase_seq(seq):
    return seq.lower()



# First step: prepare the data
def import_multi30k_dataset(path = '/home/rustam/Coding/python/transformer/dataset/'):
    
    ret = []
    filenames = ["train", "val", "test"]
    # en_words, de_words = [], []

    for filename in filenames:

        examples = []

        en_path = os.path.join(path, filename + '.en')
        de_path = os.path.join(path, filename + '.de')

        en_file = [l.strip() for l in open(en_path, 'r', encoding='utf-8')]
        de_file = [l.strip() for l in open(de_path, 'r', encoding='utf-8')]

        assert len(en_file) == len(de_file)

        for i in range(len(en_file)):
            if en_file[i] != '' and de_file[i] != '':
                en_seq, de_seq = en_file[i], de_file[i]
                # en_seq, de_seq = filter_seq(en_file[i]), filter_seq(de_file[i])
                # en_seq, de_seq = lowercase_seq(en_seq), lowercase_seq(de_seq)
                # en_words.extend(en_seq.split()), de_words.extend(de_seq.split())

                examples.append({'en': en_seq, 'de': de_seq})
        # print(len(set(en_words)), len(set(de_words)))
        ret.append(examples)
    # print(len(set(en_words)), len(set(de_words)))

    return tuple(ret)

train_data, val_data, test_data = import_multi30k_dataset()



def clear_dataset(*data):

    for dataset in data:
        for example in dataset:
            example['en'] = filter_seq(example['en'])
            example['de'] = filter_seq(example['de'])

            example['en'] = lowercase_seq(example['en'])
            example['de'] = lowercase_seq(example['de'])

            example['en'] = example['en'].split()
            example['de'] = example['de'].split()

    return data

train_data, val_data, test_data = clear_dataset(train_data, val_data, test_data)
# print(train_data)

BATCH_SIZE = 64

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

PAD_INDEX = 0
SOS_INDEX = 1
EOS_INDEX = 2
UNK_INDEX = 3

toks_and_inds = {PAD_TOKEN: PAD_INDEX, SOS_TOKEN: SOS_INDEX, EOS_TOKEN: EOS_INDEX, UNK_TOKEN: UNK_INDEX}

def build_vocab(dataset, toks_and_inds, min_freq = 1):
    en_vocab = toks_and_inds.copy(); en_vocab_freqs = {}
    de_vocab = toks_and_inds.copy(); de_vocab_freqs = {}

    for example in dataset:
        for word in example['en']:
            if word not in en_vocab_freqs:
                en_vocab_freqs[word] = 0
            en_vocab_freqs[word] += 1
        for word in example['de']:
            if word not in de_vocab_freqs:
                de_vocab_freqs[word] = 0
            de_vocab_freqs[word] += 1

    for example in dataset:
        for word in example['en']:
            if word not in en_vocab and en_vocab_freqs[word] >= min_freq:
                en_vocab[word] = len(en_vocab)
        for word in example['de']:
            if word not in de_vocab and de_vocab_freqs[word] >= min_freq:
                de_vocab[word] = len(de_vocab)

    return en_vocab, de_vocab

train_data_vocabs = build_vocab(train_data, toks_and_inds, min_freq = 2)
print(len(train_data_vocabs[0]), len(train_data_vocabs[1]))


def add_tokens(dataset, batch_size):
    print("datalength:", len(dataset))

    for example in dataset:
        example['en'] = [SOS_TOKEN] + example['en'] + [EOS_TOKEN]
        example['de'] = [SOS_TOKEN] + example['de'] + [EOS_TOKEN]
        
    data_batches = np.array_split(dataset, np.arange(batch_size, len(dataset), batch_size))

    for batch in data_batches:
        max_en_seq_len, max_de_seq_len = 0, 0

        for example in batch:
            max_en_seq_len = max(max_en_seq_len, len(example['en']))
            max_de_seq_len = max(max_de_seq_len, len(example['de']))

        for example in batch:
            example['en'] = example['en'] + [PAD_TOKEN] * (max_en_seq_len - len(example['en']))
            example['de'] = example['de'] + [PAD_TOKEN] * (max_de_seq_len - len(example['de']))


    return data_batches

train_data = add_tokens(train_data, batch_size = BATCH_SIZE)
print(f"batch number: {len(train_data)}")

def build_dataset(dataset, vocabs):
    # tokens_dataset = []
    source, target = [], []
    for batch in dataset:
        # batch_tokens = []
        source_tokens, target_tokens = [], []
        for example in batch:
            en_inds = [vocabs[0][word] if word in vocabs[0] else UNK_INDEX for word in example['en']]
            de_inds = [vocabs[1][word] if word in vocabs[1] else UNK_INDEX for word in example['de']]
            # batch_tokens.append({'en': en_inds, 'de': de_inds})
            source_tokens.append(en_inds)
            target_tokens.append(de_inds)

        # tokens_dataset.append(np.asarray(batch_tokens))
        source.append(np.asarray(source_tokens))
        target.append(np.asarray(target_tokens))
    return source, target#tokens_dataset

source, target = build_dataset(train_data, train_data_vocabs)
# print(train_data[0])
# print(source[0])


# Second step: build transformer
class Transformer():

    def __init__(self, encoder, decoder, pad_idx) -> None:
        self.encoder = encoder
        self.decoder = decoder

        self.pad_idx = pad_idx

    def load(self, path):

        pickle_encoder = open(f'{path}/encoder.pkl', 'rb')
        pickle_decoder = open(f'{path}/decoder.pkl', 'rb')
        self.encoder = pkl.load(pickle_encoder)
        self.decoder = pkl.load(pickle_decoder)
        pickle_encoder.close()
        pickle_decoder.close()

        print(f'Loaded from "{path}"')

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        pickle_encoder = open(f'{path}/encoder.pkl', 'wb')
        pickle_decoder = open(f'{path}/decoder.pkl', 'wb')
        pkl.dump(self.encoder, pickle_encoder)
        pkl.dump(self.decoder, pickle_decoder)
        pickle_encoder.close()
        pickle_decoder.close()
        
        print(f'Saved to "{path}"')

    def get_pad_mask(self, x):
        #x: (batch_size, seq_len)
        return (x != self.pad_idx).astype(int)[:, np.newaxis, :]

    def get_sub_mask(self, x):
        #x: (batch_size, seq_len)
        seq_len = x.shape[1]
        return np.triu(np.ones((seq_len, seq_len)), k = 1).astype(int)

    def forward(self, src, trg):
        #src: (batch_size, source_seq_len)
        #tgt: (batch_size, target_seq_len)

        # src_mask: (batch_size, 1, seq_len)
        # tgt_mask: (batch_size, seq_len, seq_len)
        src_mask = self.get_pad_mask(src)

        trg_mask = self.get_pad_mask(trg) & self.get_sub_mask(trg)
        # print("trg mask, src mask", trg_mask.shape, src_mask.shape)

        enc_src = self.encoder.forward(src, src_mask)
        # print("enc:", enc_src.shape)
        out, attention = self.decoder.forward(trg, trg_mask, enc_src, src_mask)
        # output: (batch_size, target_seq_len, vocab_size)
        # attn: (batch_size, n_heads, target_seq_len, source_seq_len)
        return out, attention
        


INPUT_DIM = len(train_data_vocabs[0])#10
OUTPUT_DIM = len(train_data_vocabs[1])#5
HID_DIM = 256#512
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
FF_SIZE = 2048
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1


encoder = Encoder(INPUT_DIM, ENC_HEADS, ENC_LAYERS, HID_DIM, FF_SIZE, ENC_DROPOUT)
decoder = Decoder(OUTPUT_DIM, DEC_HEADS, DEC_LAYERS, HID_DIM, FF_SIZE, DEC_DROPOUT)

model = Transformer(encoder, decoder, PAD_INDEX)

# array = np.array([1, 2, 3, 4, 0, 0]).reshape(2, 3)
# array2 = np.array([1, 2, 3, 4, 0, 0]).reshape(2, 3)

# import time
# start_time = time.time()
# out, attention = model.forward(source[0], target[0])
# print(out.shape)
# print(out)
# print(time.time() - start_time)

model.save('transformer/saved models/test_transformer')
# model.load('transformer/saved models/test_transformer')

