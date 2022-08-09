import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)


import numpy as np
import pickle as pkl
from tqdm import tqdm
from transformer.modules import Encoder
from transformer.modules import Decoder
from transformer.optimizers import Adam, Nadam, Momentum, RMSProp, SGD
from transformer.losses import CategoricalCrossEntropy, BinaryCrossEntropy, MSE, CrossEntropy, TorchCrossEntropy
import matplotlib.pyplot as plt

def filter_seq(seq):
    chars2remove = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

    return ''.join([c for c in seq if c not in chars2remove])

def lowercase_seq(seq):
    return seq.lower()


def import_multi30k_dataset(path = '/home/rustam/Coding/python/numpy-transformer/dataset/'): #/dataset
    
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

# EPOCHS = 1
BATCH_SIZE = 2

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



class Transformer():

    def __init__(self, encoder, decoder, pad_idx) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

        self.optimizer = Adam()
        self.loss_function = CategoricalCrossEntropy()

    def set_optimizer(self):
        encoder.set_optimizer(self.optimizer)
        decoder.set_optimizer(self.optimizer)

    def compile(self, optimizer, loss_function):
        self.optimizer = optimizer
        self.loss_function = loss_function
        

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
        subsequent_mask = np.triu(np.ones((seq_len, seq_len)), k = 1).astype(int)
        subsequent_mask = np.logical_not(subsequent_mask)
        return subsequent_mask

    def forward(self, src, trg):
        #src: (batch_size, source_seq_len)
        #tgt: (batch_size, target_seq_len)

        # src_mask: (batch_size, 1, seq_len)
        # tgt_mask: (batch_size, seq_len, seq_len)
        src_mask = self.get_pad_mask(src)

        trg_mask = self.get_pad_mask(trg) & self.get_sub_mask(trg)

        enc_src = self.encoder.forward(src, src_mask)

        out, attention = self.decoder.forward(trg, trg_mask, enc_src, src_mask)
        # output: (batch_size, target_seq_len, vocab_size)
        # attn: (batch_size, n_heads, target_seq_len, source_seq_len)
        return out, attention

    def backward(self, error):
        error = self.decoder.backward(error)
        error = self.encoder.backward(self.decoder.encoder_error)

    def update_weights(self):
        self.encoder.update_weights()
        self.decoder.update_weights()


    def fit(self, source, target, epochs, save_every_epoch, save_path):
        self.set_optimizer()
        
        loss_history = []
        for epoch in range(epochs):
            tqdm_range = tqdm((zip(source, target)), total = len(source))
            for source_batch, target_batch in tqdm_range:#zip(source, target):
                
                # print(source_batch.shape, target_batch[:,:-1].shape)
                output, attention = self.forward(source_batch, target_batch[:,:-1])
               
                _output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])
                # print(_output)
                error = self.loss_function.derivative(_output, target_batch[:, 1:].flatten())#[:, np.newaxis]
                # print(error)
                loss_history.append(self.loss_function.loss(_output, target_batch[:, 1:].flatten()).mean())#[:, np.newaxis]

                self.backward(error.reshape(output.shape))
                self.update_weights()

                tqdm_range.set_description(
                        f"training | loss: {loss_history[-1]:.7f} | epoch {epoch + 1}/{epochs}" #loss: {loss:.4f}
                    )
        return loss_history


# INPUT_DIM = len(train_data_vocabs[0])#10
# OUTPUT_DIM = len(train_data_vocabs[1])#5
# HID_DIM = 256#512
# ENC_LAYERS = 3
# DEC_LAYERS = 3
# ENC_HEADS = 8
# DEC_HEADS = 8
# FF_SIZE = 2048
# ENC_DROPOUT = 0.1
# DEC_DROPOUT = 0.1


# encoder = Encoder(INPUT_DIM, ENC_HEADS, ENC_LAYERS, HID_DIM, FF_SIZE, ENC_DROPOUT)
# decoder = Decoder(OUTPUT_DIM, DEC_HEADS, DEC_LAYERS, HID_DIM, FF_SIZE, DEC_DROPOUT)

# print("batch0 shape", source[0].shape, target[0].shape)

# model = Transformer(encoder, decoder, PAD_INDEX)
# model.compile(optimizer = Adam(), loss_function = CategoricalCrossEntropy(ignore_index=PAD_INDEX))
# model.fit([source[0]], [target[0]], epochs = 10, save_every_epoch = 1, save_path = 'saved models/#2FgS6_transformer')



INPUT_DIM = 10#len(train_data_vocabs[0])#10
OUTPUT_DIM = 10#len(train_data_vocabs[1])#5
# INPUT_DIM = 10
# OUTPUT_DIM = 10
HID_DIM = 256#512
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
FF_SIZE = 512#2048
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1


encoder = Encoder(INPUT_DIM, ENC_HEADS, ENC_LAYERS, HID_DIM, FF_SIZE, ENC_DROPOUT)
decoder = Decoder(OUTPUT_DIM, DEC_HEADS, DEC_LAYERS, HID_DIM, FF_SIZE, DEC_DROPOUT)



array = np.array([1, 8, 3, 4, 2, 0]).reshape(2, 3)
array2 = np.array([1, 8, 3, 4, 2, 0]).reshape(2, 3)
model = Transformer(encoder, decoder, PAD_INDEX)
#lr = 0.00005; 1e-4
model.compile(optimizer = Adam(alpha = 0.0005), loss_function = CrossEntropy(ignore_index=PAD_INDEX))#alpha = 1e-4, beta=0.9, beta2=0.98, epsilon = 1e-9
loss_history = model.fit([array], [array2], epochs = 1000, save_every_epoch = 1, save_path = 'saved models/#2FgS6_transformer')

#plot loss history
def plot_loss_history(loss_history):
    plt.plot(loss_history)
    plt.title('Loss history')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
        

plot_loss_history(loss_history)
# import time
# start_time = time.time()

# model.set_optimizer()
# out, attention = model.forward(array, array2)
# print(out.shape)
# print(out)
# print(time.time() - start_time)

# model.backward(out)
# model.update_weights()

# model.save('transformer/saved models/test_transformer')
# model.load('transformer/saved models/test_transformer')





#TESTS
import torch
import torch.nn as nn
from torch.autograd import Variable
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# output, attention = model.forward(array, array2[:,:-1])
# target = array2
# output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])
# t_target = torch.tensor([[1, 8, 3], [4, 2, 0]]).to(device)
# # target = array2[:, 1:].flatten()[:, np.newaxis]
# t_output = torch.tensor(output, requires_grad=True)#torch.from_numpy(output).float().to(device)
# # target = torch.from_numpy(target).float().to(device)
# # target = torch.from_numpy(target).float().to(device)
# # print(output.view(-1, output.size(-1)).shape, target[:, 1:].contiguous().view(-1).shape)
# # print(output.shape, target.shape)

# criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)

# loss = criterion(
#             t_output,  # (batch_size * (target_seq_len - 1), vocab_size)
#             t_target[:, 1:].contiguous().view(-1) # (batch_size * (target_seq_len - 1))
#         )
# print("torchloss", loss)
# print("model loss and der")
# print(model.loss_function.loss(output, target[:, 1:].flatten()[:, np.newaxis]).mean())
# print(model.loss_function.derivative(output, target[:, 1:].flatten()[:, np.newaxis]))
# # loss.backward()
# # print("loss")
# # print(loss)
# # print(loss.item())
# print(t_output.shape, t_target[:, 1:].contiguous().view(-1).shape)
# grad = torch.autograd.grad(loss, t_output, retain_graph=True)
# print(grad)


# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# output = loss(input, target)
# output.backward()

# print(output)


y = np.random.normal(0, 1, (2, 3))
t = np.random.normal(0, 1, (2, 3))

my_loss = MSE()
print(my_loss.loss(y, t).mean())
print(my_loss.derivative(y, t)/3)


criterion = nn.MSELoss()
y = torch.tensor(y, requires_grad=True).float()
t = torch.from_numpy(t).float()
torch_loss = criterion(y, t)
print(torch_loss)
# torch_loss.backward()
# print(torch_loss)
grad = torch.autograd.grad(torch_loss, y, retain_graph=True)
print(grad)




# u, v = torch.tensor([1.,2.], requires_grad=True), torch.tensor([3.,4.], requires_grad=True)
# print(v.shape)
# z = u.dot(v)         # tensor(11., grad_fn=<DotBackward>)
 
# z.backward()          # запускаем вычисление градиентов
 
# print(u.grad)         # tensor([3., 4.])
# print(v.grad) 


# print(torch.dot(z, torch.transpose(v, 0)))
u, v = np.array([[1.,2.]], ndmin=2), np.array([[3.,4., 5],[6,7,8]], ndmin=2)
# print(v.shape)
z = u.dot(v)         # 11.
# print(np.dot(z, v.T))
# print(np.dot(v, z.T))
print(np.dot(u.T, z))
print(np.dot(z.T, u))



x = torch.tensor([2.], requires_grad=True)


k = x
y = x * 3 + k

grad = torch.autograd.grad(y, x, retain_graph=True)
print(grad)