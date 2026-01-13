# BabyGPT :baby_chick:

<p align="center">
  <img src="https://github.com/soumyadip1995/BabyGPT/blob/main/image/gpt.png" alt="Sublime's custom image"/>
</p>

Table of contents
=================

<!--ts-->
   * [The models](#the-models)
   * [The Roadmap :rocket:](#the-roadmap-rocket)
   * [Low Rank Approximation](#low-rank-approximation)
      * [Quantization on low rank approximation](#quantization-on-low-rank-approximation)
   * [We support LLaMa for BabyGPT :zap: :zap:](#we-support-llama-for-babygpt-zap-zap)
      * [1.LLaMa version 1 :llama:](https://github.com/soumyadip1995/BabyGPT/blob/main/README.md#2-llama2-unicorn)
      * [2.LLaMa2 :unicorn:](https://github.com/soumyadip1995/BabyGPT/blob/main/README.md#2-llama2-unicorn)
           * [Tokenization](#tokenization)
      * [ LLaMA with Model FLOP Utilization(MFU) :zap:](https://github.com/soumyadip1995/BabyGPT/blob/main/README.md#llama-with-model-flop-utilizationmfu-zap)
   * [Quantization](#quantization)
   * [Our result](#our-result)
   * [Performance Benchmark](#performance-benchmark)
   * [Files](#files)
   * [Run :running:](#run-running)
   * [Auto Mixed Prescision](#auto-mixed-precision)
   * [Train and Generate :running:](#train-and-generate-running)
   * [Results :clipboard:](#results-clipboard)
   * [Data](#data)
   * [Running the notebooks](#running-the-notebooks)
   * [Mechanistic Interpretability](#mechanistic-interpretability)
   * [Chain Of Thought](#chain-of-thought)
   * [Acknowledgements](#acknowledgements)
   * [Licenses](#licenses)
<!--te-->

Building on the intuition of Karpathy's [ng-video-lectures](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py) and [mingpt](https://github.com/soumyadip1995/minGPT/blob/master/mingpt), BabyGPT provides a working model of a GPT on a much smaller scale (256 as well as 16 out channels, 5 layer GPT, fine-tuned). BabyGPT has been built from a [toyGPT](https://github.com/soumyadip1995/language-models/blob/main/Notebook/GPT_from_scratch.ipynb) which was made to understand transformers from scratch. It has been scaled down , as you will see below. Visit the notebooks. We scale up to transformers from simple Language models, attention mechanisms and finally BabyGPT. While [toyGPT](https://github.com/soumyadip1995/language-models/blob/main/Notebook/GPT_from_scratch.ipynb) has been built by separating all the layers of a transformer indiviually. In BabyGPT, the attention mechanism is implemented manually. 
The purpose of building smaller GPTs is to understand transformer functions at a much more granular level. Based on the above work , we have done a paper, you can check it in the paper folder above. 

## The models

To train small models we are using tinystories. You can download the weights from hugging face. We are setting max_iters to 5000 on a Tesla T4. For the OG model, we are using 256 out channels.

| model    | context length|n_layers |n_head        |n_embd    | train loss | val loss  | parametres| data      |
| ---------| --------------|---------| -------------|----------|------------|-----------|-----------|-----------|
| 15M      | 16            |4        | 4            |16        |2.4633      |2.4558     |13k        |[stories15M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin)|
| 42M      | 32            |8        | 8            |32        |2.3772      |2.3821     |1.01M      |[stories42M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin)|
|BabyGPT Original   | 64            |8        | 8            |256       |1.3954     |1.5959    |6.37M      |[data](https://github.com/soumyadip1995/BabyGPT/tree/main/data/ALL_eminem.txt)|

Note:- The 110M is being omitted for now. The RAM blew up..!! 

## The Roadmap :rocket:

If you wish to understand the nitty gritty of how transformers  work from scratch, this roadmap will guide you. We start from implementing simple language models bigram and ngram and then from there work our way up to building transformers , a GPT from scratch and then finally babyGPT. 

![alt_text](https://github.com/soumyadip1995/BabyGPT/blob/main/image/img3.JPG)

We implement a Low rank approximation as well as lit-lama  to  babyGPT as well. We finally train the models and generate tokens.

## Low rank approximation

Low rank approximation improves parametre efficiency(compression technique). A LoRa_model.py has been added(on 256 out channels) . We receive a parametre reduction of about 2. All, we need to do is to compute a rank parametre and compute the attention accordingly. In the [LoRa notebook](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/lora.ipynb), an estimation of FLOPs has been done according to the [chinchilla paper](https://arxiv.org/pdf/2203.15556.pdf).


### Quantization on low rank approximation

Quantization has also been performed on the lora model. A calculation of FLOPs has been added as well. For the BabyGPT model for 256 out channels we get 0.407 Peta FLOPs.  In the [LoRa notebook](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/lora.ipynb) we have added the quantization. In terms of size reduction, we are getting a reduction of a factor of 1.3 for now.


## We support LLaMa for BabyGPT :zap: :zap:

### 1. LLaMa version -1 :llama:

An implemetation of the [lit-llama](https://github.com/Lightning-AI/lit-llama) model has been ported to BabyGPT(based on llama- version 1). You can find the notebook here -> [llama_implementation](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/llama_implementation.ipynb) . Run the model ,MFU has also been added.
```llama\python llama_model_v1.py```. Training and generating tokens has been provided below. 

Note:- We have ported ```build_rope_cache()``` , ```apply_rope()``` and ```RMSNorm()```  from version 1. We are also not using the  version 1 weights or checkpoints(these are for even larger models 7B, 13B, 65B etc). You can download the weights and port llama to your own version.

### 2. LLaMa2 :unicorn:

We have ported [llama2](https://github.com/facebookresearch/llama/blob/main/llama/model.py) by meta into BabyGPT. You can find the implementation at ```llama\python llama2.py```. we have also provided a calculation of FLOPs along with the model.

The FLOPs to compute k, v cache is the flops to compute is :- 2 * 2 * num_layers * (embedded_dim) ^2. Find more information on how to compute memory and compute in [kipply's blog](https://kipp.ly/transformer-inference-arithmetic/#kv-cache)
MFU has been added to llama2

Note:- We are not using original llama weights by meta. We are also using arbitrary values for 70B. You can port it to your own model using your own weights.

#### Tokenization

Tokenization using sentencepiece has been done. We are exporting a ```tokenizer.bin``` unlike the tokenizer the quant folder. Run it in ```llama\python tokenizer.py```(meta-pieces added) . We can use the .bin file for further inference.

### LLaMA with Model FLOP Utilization(MFU) :zap:

We need efficient memory usage for LLMs. Hardware accelerators use a technique called Hardware FLOP Utilization for efficient trade-offs between memory usage and compute. This is typically done using an estimate of the ratio of FLOPs observed on a given
device to its theoretical peak FLOPs. MFU is the ratio of the observed throughput (tokens-per-second), relative to the theoretical maximum throughput of a system operating at peak FLOPs. The theoretical peak matmul of Tesla T4 is around 8.1 TFLOPS. Hence, we calculate the MFU of the LLaMA trainer model. See in the  [trainer](https://github.com/soumyadip1995/BabyGPT/blob/main/trainer.ipynb) notebook under LLaMA-trainer. We receive a MFU of :  0.0527723427% on 3.22M parametres. This would of course increase as the number of parametres increases. For a 530B parametre model, MPU is around 30% on A100 GPUs. We use Section B from the [PaLM](https://arxiv.org/pdf/2204.02311.pdf) paper for reference.


## Quantization 

LLMs  require  many GPUs to run, we need to find ways to reduce these requirements while preserving the model's performance. Various technologies have been developed that try to shrink the model size, you may have heard of quantization and distillation. It has been discovered that instead of using the 4-byte FP32 precision, we can get an almost identical inference outcome with 2-byte BF16/FP16 half-precision, which halves the model size.

To remediate that,  8-bit quantization was introduced. This method uses a quarter precision, thus needing only 1/4th of the model size! But it's not done by just dropping another half of the bits. There's a lot more to this topic. Look at hugging face quantization.

You can see [quant.md](https://github.com/soumyadip1995/BabyGPT/blob/main/quant/quant.md) on how to perform llama-quantization. You can look at [quantization Notebook](https://github.com/soumyadip1995/BabyGPT/blob/main/quant/Quantization.ipynb) for a beginner's introduction to quantization. Different benchmarkings has been done.  For ex:- On a GPU, the 7B parametre model on bfloat16 will take about 15GB. BabyGPT will take about a few kilobytes..!!!
```quantization.py``` has been obtained from lit-llama repo. A tokenizer using sentencepiece has been added as well. Diffrent kinds of weight operations can be performed from the tokenizer.model

### Our result
For post training quantization, we are able to reduce the model size by a factor of almost 4.
```
model_fp = BabyGPTmodel(config)
model_fp.eval()
model_int8 = torch.ao.quantization.quantize_dynamic(
    model_fp,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  
    
/// number of parameters: 3222637
12.9688 MB
3.4603 MB ////
```

***Note: Just for quantization we are using a bigger model with about 3.22M paramtres***

## Performance Benchmark

Performance benchmarking has been done on the BabyGPTmodel and the quantized model. Below are the results.

![alt_text](https://github.com/soumyadip1995/BabyGPT/blob/main/image/performance.png). 

It has been added to the [quantization Notebook](https://github.com/soumyadip1995/BabyGPT/blob/main/quant/Quantization.ipynb) in the quant folder.

## Files

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
│   ├── mixed precision
│   ├── mechanistic Interpretability
│   ├── Additional BabyGPT
│   ├── BabyGPT CoT
├── Train
|	├── babygpt_trainer.py
|	├── llama_trainer.py
├── transformers
|   ├── transformer_model.py
│   ├── babyGPT.py
├── Quant
│   ├── quantization.py
│   ├── quantization notebook
│   ├── tokenizer.model
│   ├── tokenizer.vocab
│   ├── tokenizer.py
│   ├── model.pth
│   ├── quant.md
├── llama
│   ├── llama2.py
│   ├── llama_model_v1.py
│   ├── tokenizer.py
│   ├── tokenizer.vocab
│   ├── tokenizer.bin
│   ├── tokenizer.model
├── text.txt
├── trainer.ipynb
├── requirements.txt

```


## Run :running:

Clone the Repo and run the following:

```
! git clone https://github.com/soumyadip1995/BabyGPT.git
```

To run the bigram and ngram language models.
```python bigram_lm.py ``` and ```python ngram_lm.py ```.

To run babygpt
```transformers\python babygpt.py``` from transformers folder.

To run a simple transformer model
```python transformer_model.py``` 

To run a low rank approximation model
```python LoRa_model.py```

To run the llama model
```llama\python llama_model_v1.py```

To run llama2
```llama\python llama2.py```

Run the different attention mechanisms from [Attention](https://github.com/soumyadip1995/BabyGPT/tree/main/Attention) folder.

### Auto Mixed Precision

A very preliminary auto mixed precision has been added. FP16/FP32 It can be achieved with a cuda enabled gpu. A combination of pytorch's ```autocast()``` and ```gradscalar()``` is used for mixed precision. See more in the pytorch tutorial. Unfortunately the gpu blew up during training and cpu for now only supports bfloat16. Takes a hell of a long time to train. If anyone can improve upon it that would be awesome. Check the [Mixed Precision](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/mixed_precision.ipynb) Notebook.


### Train and Generate :running:


If you wish to get started on  BabyGPT and llama , but don't want to go through all the hassle of knowing all about transformer models, you can simply start by running the code from the train folder.


To train and generate text from both BabyGPT model and the LLaMA model. Run
```train\python babygpt_trainer.py ``` and ```train\python llama_trainer.py ```
from the [train](https://github.com/soumyadip1995/BabyGPT/tree/main/train) folder.

Both have been trained on the Tesla T4 GPUs. You can increase or decrease the values of max_iters according to your wish. Takes a few minutes to train.

### Results :clipboard:

You can see the result from both the models in the [trainer](https://github.com/soumyadip1995/BabyGPT/blob/main/trainer.ipynb) notebook.

#### Result from the llama trainer.

Number of params = 3.22 M

```
``` number of parameters: 3222381 ```

step 0: train loss 4.6894, val loss 4.6895
step 500: train loss 2.1731, val loss 2.1832
step 1000: train loss 1.7580, val loss 1.8032
step 1500: train loss 1.5790, val loss 1.6645
step 2000: train loss 1.4482, val loss 1.5992
step 2500: train loss 1.3538, val loss 1.5874
step 3000: train loss 1.2574, val loss 1.5971
.
.
.
step 9000: train loss 0.5236, val loss 2.4614
step 9500: train loss 0.4916, val loss 2.5494
step 10000: train loss 0.4680, val loss 2.6631
step 10500: train loss 0.4448, val loss 2.6970
step 10999: train loss 0.4341, val loss 2.7462

Detroit, revior myself 'til I confused to get the big clead Mastles
Slaughterhouse on the blue, that's when he pine I'm hop with the cowprinton
robaly I want to a lox on my tempt

But now we can't never find a gift killed broke
Big before anyone could ever hear the first as I was cooped chill
But i this o for a big star
I said get chased up!
(Hello darkness, my old friend)[Eminem:]
If my legacy I acged buving in the tub (might what?)
I would know one [*Barrns, worried :]
Yeah, so kon bitch, it's
```



Seems like the model converges a bit early, towards the end. Maybe that will need more modification. 
Spitting some Eminem yo..:smile:

### Data
The [data](https://github.com/soumyadip1995/BabyGPT/tree/main/data) folder contains the text document which has the lyrics to all of Eminem's songs.

### Running the notebooks


| Notebook                    | Description |
| -----------                 | ----------- |
| Dot product attention       | [colab](https://colab.research.google.com/github/soumyadip1995/language-models/blob/main/Notebook/dot_product_attention.ipynb)|
| Multi headed attention      | [colab](https://colab.research.google.com/github/soumyadip1995/language-models/blob/main/Notebook/Multi_head_attention.ipynb)|
| GPT from scratch            | [colab](https://colab.research.google.com/github/soumyadip1995/language-models/blob/main/Notebook/GPT_from_scratch.ipynb) (Approx 860k parametres)|
| Spatial Transformers        | [colab](https://github.com/soumyadip1995/language-models/blob/main/Notebook/Spatialtransformer.ipynb)|
| BabyGPT                     | [colab](https://github.com/soumyadip1995/language-models/blob/main/Notebook/BabyGPT.ipynb)(16, 256 out channels)|
| LoRa                    | [colab](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/lora.ipynb)(256 out channels)|
| lit-llama for BabyGPT                   | [colab](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/llama_implementation.ipynb)(16 out channels for lit-llama )|
| trainer for Babygpt and llama                   | [colab](https://github.com/soumyadip1995/BabyGPT/blob/main/trainer.ipynb)(16 out channels for BabyGPT , 256 out channels for llama)|
| Mechanistic Interpretability Code                 | [colab](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/Mechanistic_Interpretability_code.ipynb)|
| Additional BabyGPT                  | [colab](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/Additional_BabyGPT.ipynb)|
| BabyGPT_CoT                 | [colab](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/BabyGPT_CoT.ipynb)|


```text.txt ``` is based on Eminem's Stan. 


## Mechanistic Interpretability

Features are one of the most fundamental properties of Neural Networks. If you think of a direction vector in a vector space of activations of neurons in a given layer , this can correspond to directions in within a feature space. Even though, indiviual neurons can be studied, these features can be connected by weights forming circuits.

The circuits can be thought of as computational graphs that consist of a  set of features, and the weighted edges that go between them in the original network. The question here is- are these vector spaces of neurons understandable ?. Can these indiviual neurons be studied ?. The answer lies in mechanistically interpreting these circuits at a deeper level.

Making use of the vector space  of neurons, in particular their weights, one can contextualize the weights in a broader context of the network *C.Olah et.al, 2018* . The challenge of contextualization is a recurring one in understanding neural networks: we can easily observe every activation, every weight, and every gradient; the challenge lies in determining what those values represent. In the [Mechanistic Interpretability](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/Mechanistic_Interpretability_code.ipynb) Notebook , we have used  an example where we are loading a pre-trained VGG16  and selecting a convolutional layer an channel inside of it. A random input image of a dog is used. We are optimizing the pixels (via gradient ascent) to maximize activation of that neuron channel. Then the resulting synthetic image is displayed. The result is usually a patterned texture (e.g., stripes, curves, eyes), showing what kind of input strongly excites that channel.  BabyGPT has also been analysed mechanistically in 
[BabyGPT_CoT](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/BabyGPT_CoT.ipynb)

### Reverse Engineering 

Mechanistic interpretability treats a neural network like a compiled program or electronic circuit. Instead of treating it as a black box (input → output correlations), we try to open the hood and explain the exact algorithms being implemented by the weights, neurons, and attention heads. This is reverse engineering in the same sense as analyzing a compiled binary. To demonstrate toy mechanistic interpretability, we can train BabyGPT to learn modular addition ```(a + b) mod N ``` . In the [Additional BabyGPT Notebook](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/Additional_BabyGPT.ipynb) we have implemented modular addition to show which attention heads/neurons implement addition and how to trace the “circuit.” This has been done using Attention Inspection:- To visualize which input token the last layer attends and linear probe.- probing to check whether  hidden states linearly encode ```(a + b) mod N ```.  This has been adopted from the works of *Nanda et al., Progress on Induction Heads, 2022*, toy mechanistic models. We have also implemented Linear Probe.

## Chain Of Thought

A step by Step CoT (addition operation) has been implemented with BabyGPT. [BabyGPT_CoT](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/BabyGPT_CoT.ipynb). WE have also implemented FLOPs per reasoning token with and without RoPE. We found a Around 2 % increase in FLOP utilization using RopE.  BabyGPT has also been implemented along with memory. 

### Acknowledgements

1. Pytorch GPT tutorial
2. Karpathy's youtube videos and tutorials
3. Karpathy's Mingpt
4. O' reilly notes for NLP
5. lit-llama repository.
6. llama from facebook
7. chinchilla paper
8. karpathy's nn zero to hero.
9. Lightning AI for Low rank Approximation.
10. IST das for gptq

### LICENSES

Licenses have been updated to include facebookresearch/llama, lit-llama and IST-das lab
You can use it under GNU, Apache and MIT.

### TO DO

1. look into triton
2. Fix the readme.md
3. Inference using libtorch or each separately
4. look into sentencepiece and tokenizer
5. look into tinystories




