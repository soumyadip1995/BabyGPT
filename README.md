## BabyGPT

Building on the intuition of Karpathy's [ng-video-lectures](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py), BabyGPT provides a working model of a GPT on a much smaller scale (around 24k parametres). BabyGPT has been built from a [toyGPT](https://github.com/soumyadip1995/language-models/blob/main/Notebook/GPT_from_scratch.ipynb) which was made to understand transformers from scratch. It has been scaled down , as you will see below. A detailed explanation covering each aspect has been provided below. We scale up to transformers from simple Language models, attention mechanisms and finally BabyGPT. 

It goes bigram_lm, ngram_lm ---> Attention ---> gpt from scratch ---> babygpt.

Low rank approximation improves parametre efficiency. A LoRa model has been added(15k parametres) less thn BabyGPT..!!.

### Files

```
BabyGPT
├── bigram_lm.py
├── ngram_lm.py
├── model.py
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
├── transformers
|   ├── model.py
│   ├── babyGPT.py
├── text.txt

```


### Run

To run the bigram and ngram language models.
```python bigram_lm.py ``` and ```python ngram_lm.py ```.

To run babygpt
```python babygpt.py``` from transformers folder.

To run a simple transformer model
```python model.py``` 

#### Running the notebooks


| Notebook                    | Description |
| -----------                 | ----------- |
| Dot product attention       | [colab](https://colab.research.google.com/github/soumyadip1995/language-models/blob/main/Notebook/dot_product_attention.ipynb)|
| Multi headed attention      | [colab](https://colab.research.google.com/github/soumyadip1995/language-models/blob/main/Notebook/Multi_head_attention.ipynb)|
| GPT from scratch            | [colab](https://colab.research.google.com/github/soumyadip1995/language-models/blob/main/Notebook/GPT_from_scratch.ipynb) (Approx 860k parametres)|
| Spatial Transformers        | [colab](https://github.com/soumyadip1995/language-models/blob/main/Notebook/Spatialtransformer.ipynb)|
| BabyGPT                     | [colab](https://github.com/soumyadip1995/language-models/blob/main/Notebook/BabyGPT.ipynb)(Approx 24k parametres)|



```text.txt ``` is based on Eminem's Stan. 


## Table of Contents.
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



### TO DO
1. If somebody could write the generate() method for babygpt, that would be helpful.
2. Building an Attention Engine.
3. The LoRa model needs more modification.


