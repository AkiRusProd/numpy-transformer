# numpy-transformer
## What it is?
This is a numpy implementation of the Transformer (Seq2Seq) model in original paper "Attention is All You Need", that runs at CPU.

<p align="center">
<img src="images/The-Transformer-model-architecture.png" width=50% height=50%>
</p>

Some methods were borrowed from my [numpy-nn-model](https://github.com/AkiRusProd/numpy-nn-model) repository.

## Training
Since the model is implemented at numpy, it runs on the CPU. Therefore, I have to set more gentle conditions for training the model. Otherwise CPU training will take **hundreds or thousands of hours**

The model was trained for 10 epochs. On my machine, training one epoch takes 3-3.5 hours. Thus training of all epochs takes 32.5 hours.
> **UPDATE**: Now training one epoch takes 2 hours with the same parameters.
### Dataset:
The dataset on which the model was trained is [Multi30k](https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/datasets/multi30k.html) dataset.
To import this just run [extract_data.py](extract_data.py) script.


## Project Structure
#### Transformer model components:  
 - [transformer.py](transformer/transformer.py) - initializes and trains the model. Also here the model predicts sequences  
 - [encoder.py](transformer/modules/encoder.py) and [decoder.py](transformer/modules/decoder.py) - modules that make up the this Seq2Seq model  
 - [base layers](transformer/layers/base) - simple neural network layers like Dense, Dropout etc
 - [combined layers](transformer/layers/combined) - special transformer layers
#### Functional components:
 - [activations.py](transformer/activations.py) - list of activation functions for the model
 - [losses.py](transformer/losses.py) - list of loss functions for the model. But used only torch like Cross Entropy Loss
 - [optimizers.py](transformer/optimizers.py) - list of gradient optimizers for the model. In this case, Adam is used with the Noam scheduler
#### Special utils:
 - [extract_data.py](extract_data.py) - downloads and extracts dataset
 - [prepare_data.py](transformer/prepare_data.py) - import the data in a suitable form for the transformer
#### Others:
 - ~~[tests](transformer/tests) - junk scripts. Do not pay attention to them~~




## Examples
#### Examples of translated sentences of validation set:  

>Example №1  
*Input sentence: a man on his wedding day  
Decoded sentence: ein mann <unk> an seinem tag <eos>  
Target sentence: ein mann an seinem hochzeitstag*  

>Example №2  
*Input sentence: a man with sunglasses is operating a construction automobile and releasing gravel on to the ground  
Decoded sentence: ein mann mit sonnenbrille und sonnenbrille bedient einen <unk> mit <unk> <eos>  
Target sentence: ein mann mit sonnenbrille bedient ein baufahrzeug und lädt kies auf den boden ab*  

>Example №3  
*Input sentence: two dogs play by a tree  
Decoded sentence: zwei hunde spielen neben einem baum <eos>  
Target sentence: zwei hunde spielen unter einem baum*  

>Example №4  
*Input sentence: a man is standing by a group of video games in a bar  
Decoded sentence: ein mann steht bei einer <unk> in einer <unk> bar <eos>  
Target sentence: ein mann steht bei einigen spielautomaten in einer bar*  

>Example №5  
*Input sentence: a gloved hand holds what appears to be an oversize nail against a log  
Decoded sentence: ein <unk> hält eine <unk> gegen die hand <unk> <unk> <unk> <eos>  
Target sentence: eine hand mit handschuh hält einen übergroßen nagel gegen einen holzscheit*  

>Example №6  
*Input sentence: a professionally dressed woman standing at a podium debating or discussing something of importance  
Decoded sentence: eine <unk> gekleidete frau steht an einem podium und <unk> etwas <unk> <eos>  
Target sentence: eine professionell gekleidete frau steht an einem podium und debattiert oder diskutiert über etwas wichtiges*  

>Example №7  
*Input sentence: two people are silhouetted against a lake reflecting a painted sky  
Decoded sentence: zwei personen <unk> gegen einen see gegen einen see <unk> himmel <eos>  
Target sentence: die silhouetten von zwei personen auf einem see der einen gemalten himmel reflektiert*  

>Example №8  
*Input sentence: a group of people on the street setting up instruments  
Decoded sentence: eine gruppe von personen auf instrumenten <unk> auf der straße <eos>  
Target sentence: eine gruppe von menschen baut auf der straße instrumente auf*  

>Example №9  
*Input sentence: a teenage boy is stretching in the kitchen and you can see part of his stomach  
Decoded sentence: ein junge in einer küche streckt sich die <unk> eines <unk> und <unk> <unk> <eos>  
Target sentence: ein teenager streckt sich in der küche und sein bauch ist teilweise sichtbar*  

>Example №10  
*Input sentence: an elderly man sitting in a chair eating some snacks  
Decoded sentence: ein älterer mann sitzt auf einem stuhl und isst <unk> <eos>  
Target sentence: ein älterer mann sitzt in einem stuhl und isst ein paar snacks*  

#### Attention plots:

<p align="center">
<img src="images/Figure_1.png" width=100% height=100%>
</p>

<p align="center">
<img src="images/Figure_2.png" width=100% height=100%>
</p>

<p align="center">
<img src="images/Figure_3.png" width=100% height=100%>
</p>

## Training problems and improvements
To improve a results of the model, you need to increase batch size, number of epochs and decrease the minimum word frequency from 10 to at least 2. However, since the model runs on a processor, such training is impossible because there is not enough CPU and RAM power for extremely large arrays to train this model with satisfactory time. Any ideas are welcome.


### References:
 - https://arxiv.org/abs/1706.03762 - article "Attention is All You Need"

### TODO:
1) ~~add pretrained model (very soon)~~
2) clean up and refactor code
3) add cupy realization (under development)
