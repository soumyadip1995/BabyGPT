### from the lit-llama repo partially

import functools
from pathlib import Path
import pickle
import warnings
from io import BytesIO
from train.babygpt_trainer import BabyGPTmodel

import torch  
import torch.nn as nn

llama_model_sizes = {
    4096: "7B",  # 7B n_embd=4096
    5120: "13B",  # 13B n_embd=5120
    6656: "30B",  # 30B n_embd=6656
    8192: "65B",  # 65B n_embd=8192
}


def model_lookup(checkpoint: dict) -> str:
    """Returns the LLaMA model name from the checkpoint.
    
    Checks the width of the lm_head.weight matrix, as these uniquely identify the model.
    """
    embedding_size = checkpoint['BabyGPTmodel.blocks.weight'].shape[1]
    return llama_model_sizes[embedding_size]


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)



class EmptyInitOnDevice(torch.overrides.TorchFunctionMode):
    def __init__(self, device=None, dtype=None, quantization_mode=None):
        """
        Create tensors with given device and dtype and don't run initialization
           (but instead use "empty tensors", i.e. uninitialized memory).

            device: `torch.device` to work with
            dtype: `torch.dtype` to work with
            quantization_mode: optional string, quantization mode to work with, default `None`.
                 Available modes: `llm.int8` bitsnbytes LLM.int8 quantization (only on GPU)
                                  `qptq.int4`, `gptq.int8`: GPTQ pre-quantized models

        Example::
            with EmptyInitOnDevice("cuda", dtype=torch.bfloat16):
               model = LLaMA.from_name('7B')
            model.load_state_dict(torch.load('llama-lit/7B/lit-llama.pth'))"""

        self.quantization_mode = quantization_mode
        self.quantized_linear_cls = None
        if self.quantization_mode == 'llm.int8':
            if device.type != "cuda":
                raise ValueError("Quantization is only supported on the GPU.")
            from .quantization import Linear8bitLt
            self.quantized_linear_cls = Linear8bitLt
        elif self.quantization_mode == 'gptq.int4':
            from .quantization import ColBlockQuantizedLinear
            self.quantized_linear_cls = functools.partial(ColBlockQuantizedLinear, bits=4, tile_cols=-1)
        elif self.quantization_mode == 'gptq.int8':
            from .quantization import ColBlockQuantizedLinear
            self.quantized_linear_cls = functools.partial(ColBlockQuantizedLinear, bits=8, tile_cols=-1)
        elif self.quantization_mode is not None:
            raise RuntimeError(f"unknown quantization mode {self.quantization_mode}")
        self.device = device
        self.dtype = dtype

    def __enter__(self):
        if self.quantized_linear_cls != None:
            self.torch_linear_cls = torch.nn.Linear
            torch.nn.Linear = self.quantized_linear_cls
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.quantized_linear_cls != None:
            torch.nn.Linear = self.torch_linear_cls
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, "__module__", None) == "torch.nn.init":
            if "tensor" in kwargs:
                return kwargs["tensor"]
            else:
                return args[0]
        if (
            self.device is not None
            and func in torch.utils._device._device_constructors()
            and kwargs.get("device") is None
        ):
            kwargs["device"] = self.device
        if (
            self.dtype is not None
            and func in torch.utils._device._device_constructors()
            and kwargs.get("dtype") is None
        ):
            kwargs["dtype"] = self.dtype
        return func(*args, **kwargs)


# this is taken from torchhacks https://github.com/lernapparat/torchhacks