import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)


import numpy as np
try: 
    import cupy as cp
    is_cupy_available = True
    print('Cupy is available. Using Cupy for all computations.')
except:
    is_cupy_available = False
    print('CuPy is not available. Switching to NumPy.')
    
import pickle as pkl
from tqdm import tqdm
from transformer.modules import Encoder
from transformer.modules import Decoder
from transformer.optimizers import Adam, Nadam, Momentum, RMSProp, SGD, Noam
from transformer.losses import CrossEntropy
from transformer.prepare_data import DataPreparator
import matplotlib.pyplot as plt



DATA_TYPE = np.float32
BATCH_SIZE = 32

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

PAD_INDEX = 0
SOS_INDEX = 1
EOS_INDEX = 2
UNK_INDEX = 3

tokens  = (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
indexes = (PAD_INDEX, SOS_INDEX, EOS_INDEX, UNK_INDEX)

data_preparator = DataPreparator(tokens, indexes)

train_data, test_data, val_data = data_preparator.prepare_data(
                    path = 'dataset/', 
                    batch_size = BATCH_SIZE, 
                    min_freq = 2)

source, target = train_data

train_data_vocabs = data_preparator.get_vocabs()



class Seq2Seq():

    def __init__(self, encoder, decoder, pad_idx) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

        self.optimizer = Adam()
        self.loss_function = CrossEntropy()

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

    def forward(self, src, trg, training):
        src, trg = src.astype(DATA_TYPE), trg.astype(DATA_TYPE)
        #src: (batch_size, source_seq_len)
        #tgt: (batch_size, target_seq_len)

        # src_mask: (batch_size, 1, seq_len)
        # tgt_mask: (batch_size, seq_len, seq_len)
        src_mask = self.get_pad_mask(src)

        trg_mask = self.get_pad_mask(trg) & self.get_sub_mask(trg)

        enc_src = self.encoder.forward(src, src_mask, training)

        out, attention = self.decoder.forward(trg, trg_mask, enc_src, src_mask, training)
        # output: (batch_size, target_seq_len, vocab_size)
        # attn: (batch_size, heads_num, target_seq_len, source_seq_len)
        return out, attention

    def backward(self, error):
        error = self.decoder.backward(error)
        error = self.encoder.backward(self.decoder.encoder_error)

    def update_weights(self):
        self.encoder.update_weights()
        self.decoder.update_weights()


    def fit(self, source, target, epochs, save_every_epochs, save_path = None):
        self.set_optimizer()
        
        loss_history = []
        for epoch in range(epochs):
            tqdm_range = tqdm(enumerate(zip(source, target)), total = len(source))
            for batch_num, (source_batch, target_batch) in tqdm_range:
                
                output, attention = self.forward(source_batch, target_batch[:,:-1], training = True)
               
                _output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])

                loss_history.append(self.loss_function.loss(_output, target_batch[:, 1:].astype(np.int32).flatten()).mean())#[:, np.newaxis]
                error = self.loss_function.derivative(_output, target_batch[:, 1:].astype(np.int32).flatten())#[:, np.newaxis]


                self.backward(error.reshape(output.shape))
                self.update_weights()

                tqdm_range.set_description(
                        f"training | loss: {loss_history[-1]:.7f} | perplexity: {np.exp(loss_history[-1]):.7f} | epoch {epoch + 1}/{epochs}" #loss: {loss:.4f}
                    )

                if batch_num == (len(source) - 1):
                    if is_cupy_available:
                        epoch_loss = cp.mean(cp.array(loss_history[(epoch) * len(source) : (epoch + 1) * len(source) ]))
                    else:
                        epoch_loss = np.mean(loss_history[(epoch) * len(source) : (epoch + 1) * len(source) ])

                    tqdm_range.set_description(
                            f"training | avg loss: {epoch_loss:.7f} | avg perplexity: {np.exp(epoch_loss):.7f} | epoch {epoch + 1}/{epochs}"
                    )

            if (save_path is not None) and (epoch % save_every_epochs == 0):
                self.save(save_path + f'/{epoch}')
                
        return loss_history


    def evaluate(self, source, target):
        loss_history = []

        tqdm_range = tqdm(enumerate(zip(source, target)), total = len(source))
        for batch_num, (source_batch, target_batch) in tqdm_range:
            
            output, attention = self.forward(source_batch, target_batch[:,:-1], training = False)
            
            _output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])

            loss_history.append(self.loss_function.loss(_output, target_batch[:, 1:].astype(np.int32).flatten()).mean())
            
            tqdm_range.set_description(
                    f"testing | loss: {loss_history[-1]:.7f} | perplexity: {np.exp(loss_history[-1]):.7f}"
                )

            if batch_num == (len(source) - 1):
                epoch_loss = np.mean(loss_history)

                tqdm_range.set_description(
                        f"testing | avg loss: {epoch_loss:.7f} | avg perplexity: {np.exp(epoch_loss):.7f}"
                )

        return loss_history

    def predict(self, sentence, vocabs, max_length = 50):

        src_inds = [vocabs[0][word] if word in vocabs[0] else UNK_INDEX for word in sentence]
        src_inds = [SOS_INDEX] + src_inds + [EOS_INDEX]
        
        src = np.asarray(src_inds).reshape(1, -1)
        src_mask =  self.get_pad_mask(src)

        enc_src = self.encoder.forward(src, src_mask, training = False)

        trg_inds = [SOS_INDEX]

        for _ in range(max_length):
            trg = np.asarray(trg_inds).reshape(1, -1)
            trg_mask = self.get_pad_mask(trg) & self.get_sub_mask(trg)

            out, attention = self.decoder.forward(trg, trg_mask, enc_src, src_mask, training = False)
            
            trg_indx = out.argmax(axis=-1)[:, -1].item()
            trg_inds.append(trg_indx)

            if trg_indx == EOS_INDEX or len(trg_inds) >= max_length:
                break
        
        reversed_vocab = dict((v,k) for k,v in vocabs[1].items())
        decoded_sentence = [reversed_vocab[indx] if indx in reversed_vocab else UNK_TOKEN for indx in trg_inds]

        return decoded_sentence[1:], attention[0]




INPUT_DIM = len(train_data_vocabs[0])
OUTPUT_DIM = len(train_data_vocabs[1])
HID_DIM = 256  #512 in original paper
ENC_LAYERS = 3 #6 in original paper
DEC_LAYERS = 3 #6 in original paper
ENC_HEADS = 8
DEC_HEADS = 8
FF_SIZE = 512  #2048 in original paper
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

MAX_LEN = 5000


encoder = Encoder(INPUT_DIM, ENC_HEADS, ENC_LAYERS, HID_DIM, FF_SIZE, ENC_DROPOUT, MAX_LEN, DATA_TYPE)
decoder = Decoder(OUTPUT_DIM, DEC_HEADS, DEC_LAYERS, HID_DIM, FF_SIZE, DEC_DROPOUT, MAX_LEN, DATA_TYPE)



model = Seq2Seq(encoder, decoder, PAD_INDEX)


model.load("saved models/seq2seq_model/0")


model.compile(
                optimizer = Noam(
                                Adam(alpha = 1e-4, beta = 0.9, beta2 = 0.98, epsilon = 1e-9), #NOTE: alpha doesn`t matter for Noam scheduler
                                model_dim = HID_DIM,
                                scale_factor = 2,
                                warmup_steps = 4000
                            ) 
                , loss_function = CrossEntropy(ignore_index=PAD_INDEX)
            )
# loss_history = model.fit(source, target, epochs = 10, save_every_epochs = 1, save_path = "saved models/seq2seq_model")


def plot_loss_history(loss_history):
    plt.plot(loss_history)
    plt.title('Loss history')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.show()
        

# plot_loss_history(loss_history)



_, _, val_data = data_preparator.import_multi30k_dataset(path = "dataset/")
val_data = data_preparator.clear_dataset(val_data)[0]
sentences_num = 10

random_indices = np.random.randint(0, len(val_data), sentences_num)
sentences_selection = [val_data[i] for i in random_indices]

#Translate sentences from validation set
for i, example in enumerate(sentences_selection):
    print(f"\nExample №{i + 1}")
    print(f"Input sentence: { ' '.join(example['en'])}")
    print(f"Decoded sentence: {' '.join(model.predict(example['en'], train_data_vocabs)[0])}")
    print(f"Target sentence: {' '.join(example['de'])}")




def plot_attention(sentence, translation, attention, heads_num = 8, rows_num = 2, cols_num = 4):
    
    assert rows_num * cols_num == heads_num
    
    sentence = [SOS_TOKEN] + [word.lower() for word in sentence] + [EOS_TOKEN]

    fig = plt.figure(figsize = (15, 25))
    
    for h in range(heads_num):
        
        ax = fig.add_subplot(rows_num, cols_num, h + 1)
        ax.set_xlabel(f'Head {h + 1}')
        
        if is_cupy_available:
            ax.matshow(cp.asnumpy(attention[h]), cmap = 'inferno')
        else:
            ax.matshow(attention[h], cmap = 'inferno')

        ax.tick_params(labelsize = 7)

        ax.set_xticks(range(len(sentence)))
        ax.set_yticks(range(len(translation)))

        ax.set_xticklabels(sentence, rotation=90)
        ax.set_yticklabels(translation)


    plt.show()

#Plot Attention
sentence = sentences_selection[0]['en']#['a', 'trendy', 'girl', 'talking', 'on', 'her', 'cellphone', 'while', 'gliding', 'slowly', 'down', 'the', 'street']
print(f"\nInput sentence: {sentence}")
decoded_sentence, attention =  model.predict(sentence, train_data_vocabs)
print(f"Decoded sentence: {decoded_sentence}")

plot_attention(sentence, decoded_sentence, attention)