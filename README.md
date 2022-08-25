# numpy-transformer `(under development)`
This is a numpy implementation of the Transformer model in `"Attention is All You Need"`, that runs at CPU

<p align="center">
<img src="images/The-Transformer-model-architecture.png" width=50% height=50%>
</p>

Some methods were borrowed from my [numpy-nn-model](https://github.com/AkiRusProd/numpy-nn-model) repository

Since the model is implemented at numpy, it runs on the CPU. Therefore, I have to set more gentle conditions for training the model. Otherwise CPU training will take **hundreds or thousands of hours**


Example of translated sentences:

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

### References:
 - https://arxiv.org/abs/1706.03762 - article "Attention is All You Need"

### TODO:
1) ~~add pretrained model (very soon)~~
2) clean up and refactor code

