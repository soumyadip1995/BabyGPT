# BabyGPT :baby_chick:

<p align="center">
  <img src="https://github.com/soumyadip1995/BabyGPT/blob/main/image/chg.png" alt="Sublime's custom image"/>
</p>

Building on the intuition of Karpathy's [ng-video-lectures](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py) and [mingpt](https://github.com/soumyadip1995/minGPT/blob/master/mingpt), BabyGPT provides a working model of a GPT on a much smaller scale (256 as well as 16 out channels, 5 layer GPT, fine-tuned). BabyGPT has been built from a [toyGPT](https://github.com/soumyadip1995/language-models/blob/main/Notebook/GPT_from_scratch.ipynb) which was made to understand transformers from scratch. It has been scaled down , as you will see below. Visit the notebooks. We scale up to transformers from simple Language models, attention mechanisms and finally BabyGPT. While [toyGPT](https://github.com/soumyadip1995/language-models/blob/main/Notebook/GPT_from_scratch.ipynb) has been built by separating all the layers of a transformer indiviually. In BabyGPT, the attention mechanism is implemented manually. 
The purpose of building smaller GPTs is to understand transformer functions at a much more granular level.


## The Roadmap :rocket:

If you wish to understand the nitty gritty of how transformers  work from scratch, this roadmap will guide you. We start from implementing simple language models and work our way up to building transformers , a GPT from scratch and then finally babyGPT. 

![alt_text](https://github.com/soumyadip1995/BabyGPT/blob/main/image/img3.JPG)

We implement a Low rank approximation as well as lit-lama  to  babyGPT as well. We finally train the models and generate tokens.

## Low rank approximation

Low rank approximation improves parametre efficiency(compression technique). A LoRa_model.py has been added(on 256 out channels) . We receive a parametre reduction of about 2. All, we need to do is to compute a rank parametre and compute the attention accordingly. In the [LoRa notebook](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/lora.ipynb), an estimation of FLOPs has been done according to the [chinchilla paper](https://arxiv.org/pdf/2203.15556.pdf).


### Quantization on low rank approximation

Quantization has also been performed on the lora model. A calculation of FLOPs has been added as well. For the BabyGPT model for 256 out channels we get 0.407 Peta FLOPs.  In the [LoRa notebook](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/lora.ipynb) we have added the quantization. In terms of size reduction, we are getting a reduction of a factor of 1.3 for now.


## We support LLaMa for BabyGPT :zap: :zap:


### 1. LLaMa version -1

An implemetation of the [lit-llama](https://github.com/Lightning-AI/lit-llama) model has been ported to BabyGPT(based on llama- version 1). You can find the notebook here -> [llama_implementation](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/llama_implementation.ipynb) . Run the model 
```llama\python llama_model_v1.py```. Training and generating tokens has been provided below. 

Note:- We have ported ```build_rope_cache()``` , ```apply_rope()``` and ```RMSNorm()```  from version 1. We are also not using the  version 1 weights or checkpoints(these are for even larger models 7B, 13B, 65B etc). You can download the weights and port llama to your own version.

### 1. LLaMa2

We have ported [llama2](https://github.com/facebookresearch/llama/blob/main/llama/model.py) by meta into BabyGPT. You can find the implementation at ```llama\python llama2.py```. we have also provided a calculation of FLOPs along with the model.

The FLOPs to compute k, v cache is the flops to compute is :- 2 * 2 * num_layers * (embedded_dim) ^2. Find more information on how to compute memory and compute in [kipply's blog](https://kipp.ly/transformer-inference-arithmetic/#kv-cache)

Note:- We are not using original llama weights by meta. We are also using arbitrary values for 70B. You can port it to your own model using your own weights.

### LLaMA with Model FLOP Utilization(MFU) :zap:

We need efficient memory usage for LLMs. Hardware accelerators use a technique called Hardware FLOP Utilization for efficient trade-offs between memory usage and compute. This is typically done using an estimate of the ratio of FLOPs observed on a given
device to its theoretical peak FLOPs. MFU is the ratio of the observed throughput (tokens-per-second), relative to the theoretical maximum throughput of a system operating at peak FLOPs. The theoretical peak matmul of Tesla T4 is around 8.1 TFLOPS. Hence, we calculate the MFU of the LLaMA trainer model. See in the  [trainer](https://github.com/soumyadip1995/BabyGPT/blob/main/trainer.ipynb) notebook under LLaMA-trainer. We receive a MFU of :  0.0527723427% on 3.22M parametres. This would of course increase as the number of parametres increases. For a 530B parametre model, MPU is around 30% on A100 GPUs. We use Section B from the [PaLM](https://arxiv.org/pdf/2204.02311.pdf) paper for reference.


## Quantization 

LLMs  require  many GPUs to run, we need to find ways to reduce these requirements while preserving the model's performance. Various technologies have been developed that try to shrink the model size, you may have heard of quantization and distillation. It has been discovered that instead of using the 4-byte FP32 precision, we can get an almost identical inference outcome with 2-byte BF16/FP16 half-precision, which halves the model size.

To remediate that,  8-bit quantization was introduced. This method uses a quarter precision, thus needing only 1/4th of the model size! But it's not done by just dropping another half of the bits. There's a lot more to this topic. Look at hugging face quantization.

You can see [quant.md](https://github.com/soumyadip1995/BabyGPT/blob/main/quant/quant.md) on how to perform llama-quantization. You can look at [quantization Notebook](https://github.com/soumyadip1995/BabyGPT/blob/main/quant/Quantization.ipynb) for a beginner's introduction to quantization. Different benchmarkings has been done.  For ex:- On a GPU, the 7B parametre model on bfloat16 will take about 15GB. BabyGPT will take about a few kilobytes..!!!
```quantization.py``` has been obtained from lit-llama repo.

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
│   ├── model.pth
│   ├── quant.md
├── llama
│   ├── llama2.py
│   ├── llama_model_v1.py
├── text.txt
├── trainer.ipynb

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

Run the different attention mechanisms from [Attention](https://github.com/soumyadip1995/BabyGPT/tree/main/Attention) folder.

### Auto Mixed Precision

A very preliminary auto mixed precision has been added. It can be achieved with a cuda enabled gpu. A combination of pytorch's autocast and gradscaler is used for mixed precision. See more in the pytorch tutorial. Unfortunately the gpu blew up during training and cpu for now only supports bfloat16. Takes a hell of a long time to train. If anyone can improve upon it that would be awesome. Check the [Mixed Precision](https://github.com/soumyadip1995/BabyGPT/blob/main/Notebook/mixed_precision.ipynb) Notebook.


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


```text.txt ``` is based on Eminem's Stan. 




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

1. Try out quantization for Lora 
2. analysing the different performance benchmarks for the models.



