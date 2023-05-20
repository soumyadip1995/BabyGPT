### Quantization

We are performing post training quantization. After training has been completed, we quantize the layers in order, instead of quantizing everything at once. Will elaborate more about quantization later on. 

## llama quantization

I haven't included the llama quantization yet. You can follow the instructions below if you wish to:

Downloading pretrained weights
Except for when you are training from scratch, you will need the pretrained weights from Meta.

### Original Meta weights
Download the model weights following the instructions on the official LLaMA repository.

Once downloaded, you should have a folder like this:

checkpoints/llama
├── 7B
│   ├── ...
│   └── consolidated.00.pth
├── 13B
│   ...
└── tokenizer.model
Convert the weights to the LLaMA format:

Covert the checkpoints.

Note: All scripts are prone toargument customization

### OpenLLaMA
OpenLM Research has released Apache 2.0 licensed weights obtained by training LLaMA on the 1.2 trillion token open-source RedPajama dataset.

Weights were released in preview on intermediate number of tokens (400B at the time of writing). In order to get them do:

 Make sure you have git-lfs installed (https://git-lfs.com): git lfs install

git clone https://huggingface.co/openlm-research/open_llama_7b_400bt_preview checkpoints/open-llama/7B

### TO DO:

The babygpt quantization is still a little bit dodgy. I will come back to it later. Tokenizer works, quantization.py works.