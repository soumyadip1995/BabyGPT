## BabyGPT :baby_chick:

Building on the intuition of Karpathy's [ng-video-lectures](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py) and [mingpt](https://github.com/soumyadip1995/minGPT/blob/master/mingpt), BabyGPT provides a working model of a GPT on a much smaller scale (around 28k parametres, fine-tuned). BabyGPT has been built from a [toyGPT](https://github.com/soumyadip1995/language-models/blob/main/Notebook/GPT_from_scratch.ipynb) which was made to understand transformers from scratch. It has been scaled down , as you will see below. Visit the notebooks. We scale up to transformers from simple Language models, attention mechanisms and finally BabyGPT. While [toyGPT](https://github.com/soumyadip1995/language-models/blob/main/Notebook/GPT_from_scratch.ipynb) has been built by separating all the layers of a transformer indiviually. In BabyGPT, the attention mechanism is implemented manually. 
The purpose of building smaller GPTs is to understand transformer functions at the bit level.

It goes bigram_lm, ngram_lm ---> Attention ---> gpt from scratch ---> babygpt.

Low rank approximation improves parametre efficiency. A LoRa_model.py has been added(15k parametres) less than BabyGPT..!!. All, we need to do is to compute a rank parametre and compute the attention accordingly. In the [LoRa notebook](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/lora.ipynb), an estimation of FLOPs has been done according to the [chinchilla paper](https://arxiv.org/pdf/2203.15556.pdf).


### We support Lit llama for BabyGPT :zap:

An implemetation of the [lit-llama](https://github.com/Lightning-AI/lit-llama) model has been added to BabyGPT. It still needs a bit more modification. You can find the notebook here -> [llama_implementation](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/llama_implementation.ipynb) . Run the model 
```python llama_model.py```. Training and generating tokens has been provided below.

### Files

```
BabyGPT
├── bigram_lm.py
├── ngram_lm.py
├── model.py
├── Lora_model.py
├── Llama_model.py
├── Attention
│   ├── dot product attention.py
│   ├── multi headed attention.py
│   ├── cross attention.py
│   ├── spatial attention.py
├── Notebook
│   ├── Dot product attention
│   ├── multiheaded attention
│   ├── gpt from scratch
│   ├── spatial transformer
│   ├── babyGPT
│   ├── LoRa
│   ├── llama_implementation
├── Train
|	├── babygpt_trainer.py
|	├── llama_trainer.py
├── transformers
|   ├── transformer_model.py
│   ├── babyGPT.py
├── text.txt
├── trainer.ipynb

```


### Run :running:

To run the bigram and ngram language models.
```python bigram_lm.py ``` and ```python ngram_lm.py ```.

To run babygpt
```python babygpt.py``` from transformers folder.

To run a simple transformer model
```python transformer_model.py``` 
To run a low rank approximation model
```python LoRa_model.py```

Run the different attention mechanisms from [Attention](https://github.com/soumyadip1995/BabyGPT/tree/main/Attention) folder.

### Train and Generate :running:


If you wish to get started on  BabyGPT and llama , but don't want to go through all the hassle of knowing about transformer models, you can simply start by running the code from the train folder.


To train and generate text from both BabyGPT model and the LLaMA model. Run
```python babygpt_trainer.py ``` and ```python llama_trainer.py ```
from the [train](https://github.com/soumyadip1995/BabyGPT/tree/main/train) folder.

Both have been trained on the Tesla T4 GPUs. You can increase or decrease the values of max_iters according to your wish. Takes a few minutes to train.

### Results :clipboard:

You can see the result from both the models in the [trainer](https://github.com/soumyadip1995/BabyGPT/blob/main/trainer.ipynb) notebook.

#### Result from the llama trainer.

```
``` number of parameters: 16221 ```
step 0: train loss 4.6937, val loss 4.6937
step 500: train loss 3.1929, val loss 3.1976
step 1000: train loss 2.7869, val loss 2.7994
.
.
.
step 10999: train loss 2.2015, val loss 2.2009

Oh a he spoments
Ohs and whet me.0
Jay, Dre hanf inche, buh, I'm herere a win'
So to saiidid
I I'mf neved, I donse efy fake'ind merile (I'd nobqu cad-shegeds..]
Get withank to I what baut the gudin' I'm no]
Now oun the cuckedid in
I gine we dey (Be be doup I cloe that in a treve I ack pith my!
I gine we dey (Be be doup I cloe that in a treve I ack pith my!
?
Thhide was ueves sueel The ray heB*Red, yeu the me won beall you cham?
This if to my ona um a lating ut Dof'
Just to to sarse mighior mey (namale me
Heats to ine Ack moy t'
She good whene core pruting yo
```

Seems like the model converges a bit, towards the end. Maybe that will need more modification. 
Spitting some Eminem yo..:smile:

### Data
The [data](https://github.com/soumyadip1995/BabyGPT/tree/main/data) folder contains the text document whioch has the lyrics to all of Eminem's songs.

### Running the notebooks


| Notebook                    | Description |
| -----------                 | ----------- |
| Dot product attention       | [colab](https://colab.research.google.com/github/soumyadip1995/language-models/blob/main/Notebook/dot_product_attention.ipynb)|
| Multi headed attention      | [colab](https://colab.research.google.com/github/soumyadip1995/language-models/blob/main/Notebook/Multi_head_attention.ipynb)|
| GPT from scratch            | [colab](https://colab.research.google.com/github/soumyadip1995/language-models/blob/main/Notebook/GPT_from_scratch.ipynb) (Approx 860k parametres)|
| Spatial Transformers        | [colab](https://github.com/soumyadip1995/language-models/blob/main/Notebook/Spatialtransformer.ipynb)|
| BabyGPT                     | [colab](https://github.com/soumyadip1995/language-models/blob/main/Notebook/BabyGPT.ipynb)(Approx 28k parametres)|
| LoRa                    | [colab](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/lora.ipynb)(Approx 15k parametres). less than babygpt|
| lit-llama for BabyGPT                   | [colab](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/llama_implementation.ipynb)(Approx 28k parametres)|



```text.txt ``` is based on Eminem's Stan. 



#### 1. Simple Language Models
1. [Bigram Language Model](https://github.com/soumyadip1995/language-models/blob/main/bigram_lm.py).
2. [N-gram Language Model](https://github.com/soumyadip1995/language-models/blob/main/ngram_lm.py).
     
#### 2. Attention Mechanisms
1. [Dot product attention](https://github.com/soumyadip1995/language-models/blob/main/Attention/dot_product_attention.py)
2. [Multi headed attention](https://github.com/soumyadip1995/language-models/blob/main/Attention/multi_headed_attention.py)
3. [Cross Attention](https://github.com/soumyadip1995/language-models/blob/main/Attention/cross_attention.py)
4. [Spatial Attention](https://github.com/soumyadip1995/language-models/blob/main/Attention/spatial_attention.py)
    
#### 3. Transformers
1. [GPTs from Scratch](https://github.com/soumyadip1995/language-models/blob/main/Notebook/GPT_from_scratch.ipynb)
2. [Spatial Transformers](https://github.com/soumyadip1995/language-models/blob/main/Notebook/Spatialtransformer.ipynb)
3. [Simple Transformer model](https://github.com/soumyadip1995/language-models/blob/main/model.py)
4. [BabyGPT](https://github.com/soumyadip1995/language-models/blob/main/Notebook/BabyGPT.ipynb)

#### 4. Others
1. [Low rank approximation](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/lora.ipynb)
2. [llama implementation](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/llama_implementation.ipynb)


### Acknowledgements

1. Pytorch GPT tutorial
2. Karpathy's youtube videos and tutorials
3. Karpathy's Mingpt
4. O' reilly notes for NLP
5. lit-llama repository.
6. llama from facebook
7. chinchilla paper
8. karpathy's nn zero to hero.

### TO DO

1. Building an Attention Engine.



