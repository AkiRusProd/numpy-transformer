import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np


class DataPreparator():

    def __init__(self, tokens, indexes):

        self.PAD_TOKEN = tokens[0]
        self.SOS_TOKEN = tokens[1]
        self.EOS_TOKEN = tokens[2]
        self.UNK_TOKEN = tokens[3]

        self.PAD_INDEX = indexes[0]
        self.SOS_INDEX = indexes[1]
        self.EOS_INDEX = indexes[2]
        self.UNK_INDEX = indexes[3]

        self.toks_and_inds = {self.PAD_TOKEN: self.PAD_INDEX, self.SOS_TOKEN: self.SOS_INDEX, self.EOS_TOKEN: self.EOS_INDEX, self.UNK_TOKEN: self.UNK_INDEX}
        self.vocabs = None

    def prepare_data(self, path = 'dataset/', batch_size = 1, min_freq = 10):
        
        train_data, val_data, test_data = self.import_multi30k_dataset(path)
        train_data, val_data, test_data = self.clear_dataset(train_data, val_data, test_data)
        print(f"train data sequences num = {len(train_data)}")

        self.vocabs = self.build_vocab(train_data, self.toks_and_inds, min_freq)
        print(f"EN vocab length = {len(self.vocabs[0])}; DE vocab length = {len(self.vocabs[1])}")

        train_data = self.add_tokens(train_data, batch_size)
        print(f"batch num = {len(train_data)}")

        train_source, train_target = self.build_dataset(train_data, self.vocabs)

        test_data = self.add_tokens(test_data, batch_size)
        test_source, test_target = self.build_dataset(test_data, self.vocabs)

        val_data = self.add_tokens(val_data, batch_size)
        val_source, val_target = self.build_dataset(val_data, self.vocabs)

        return (train_source, train_target), (test_source, test_target), (val_source, val_target)

    def get_vocabs(self):
        return self.vocabs

    def filter_seq(self, seq):
        chars2remove = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

        return ''.join([c for c in seq if c not in chars2remove])

    def lowercase_seq(self, seq):
        return seq.lower()

    
    def import_multi30k_dataset(self, path):
        
        ret = []
        filenames = ["train", "val", "test"]

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

                    examples.append({'en': en_seq, 'de': de_seq})
        
            ret.append(examples)
   
        return tuple(ret)


    def clear_dataset(self, *data):

        for dataset in data:
            for example in dataset:
                example['en'] = self.filter_seq(example['en'])
                example['de'] = self.filter_seq(example['de'])

                example['en'] = self.lowercase_seq(example['en'])
                example['de'] = self.lowercase_seq(example['de'])

                example['en'] = example['en'].split()
                example['de'] = example['de'].split()

        return data



    def build_vocab(self, dataset, toks_and_inds, min_freq = 1):
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


    def add_tokens(self, dataset, batch_size):
        for example in dataset:
            example['en'] = [self.SOS_TOKEN] + example['en'] + [self.EOS_TOKEN]
            example['de'] = [self.SOS_TOKEN] + example['de'] + [self.EOS_TOKEN]
            
        data_batches = np.array_split(dataset, np.arange(batch_size, len(dataset), batch_size))

        for batch in data_batches:
            max_en_seq_len, max_de_seq_len = 0, 0

            for example in batch:
                max_en_seq_len = max(max_en_seq_len, len(example['en']))
                max_de_seq_len = max(max_de_seq_len, len(example['de']))

            for example in batch:
                example['en'] = example['en'] + [self.PAD_TOKEN] * (max_en_seq_len - len(example['en']))
                example['de'] = example['de'] + [self.PAD_TOKEN] * (max_de_seq_len - len(example['de']))


        return data_batches


    def build_dataset(self, dataset, vocabs):
        
        source, target = [], []
        for batch in dataset:
            
            source_tokens, target_tokens = [], []
            for example in batch:
                en_inds = [vocabs[0][word] if word in vocabs[0] else self.UNK_INDEX for word in example['en']]
                de_inds = [vocabs[1][word] if word in vocabs[1] else self.UNK_INDEX for word in example['de']]
                
                source_tokens.append(en_inds)
                target_tokens.append(de_inds)

            source.append(np.asarray(source_tokens))
            target.append(np.asarray(target_tokens))

        return source, target
