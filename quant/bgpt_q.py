import gc
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import math
from quantization import GPTQQuantizer
from tokenizer import Tokenizer
import torch
from train.babygpt_trainer import BabyGPTmodel
from utils import model_lookup, EmptyInitOnDevice

# @torch.no_grad()
# def blockwise_quantization(
#     BabyGPTmodel, working_device, *, bits=4, groupsize=-1):
#     # This is the classic post-training quantization
#     # of all linear layers. We quantize in order, i.e.
#     # when observing the inputs, we use the outputs
#     # of the previously quantized layers rather than
#     # doing them all at once.

#     print("Getting inputs for first block")
#     print(BabyGPTmodel)
#     print(BabyGPTmodel.config)

#     BabyGPTmodel.token.to(working_device)
#     inps = []
#     for batch in sample_inputs:
#         inps.append(BabyGPTmodel.token(batch[None].to(working_device)))
#     inps = torch.cat(inps, dim=0)
#     BabyGPTmodel.token.to("cpu")
#     torch.cuda.empty_cache()

#     print("Starting to quantize blocks")
#     outs = torch.zeros_like(inps)

#     # better than relying on enumeration? originally the code bundled
#     # the two mlp fc layers
#     # we could automate this with a lot of hooks and another iteration
#     submodules_to_process = [
#         "Attention.attention",
#         "Attention.projection"
#     ]

#     for i, block in enumerate(BabyGPTmodel.blocks):
#         block.to(working_device)

#         for name in submodules_to_process:
#             print(i, name, end=" ")
#             t0 = time.perf_counter()
#             print("collecting stats", end=" ")
#             sys.stdout.flush()
#             module = block.get_submodule(name)

#             gptq = GPTQQuantizer(
#                 module,
#                 bits=bits,
#                 groupsize=groupsize,
#                 actorder=(groupsize == -1),
#             )
#             handle = module.register_forward_hook(gptq.collect_input_stats)
#             for j in range(inps.size(0)):
#                 outs[j : j + 1] = block(inps[j : j + 1])

#             handle.remove()

#             print("quantizing", end=" ")
#             sys.stdout.flush()
#             q_module, error = gptq.quantize()

#             # replace the linear module with the quantized module
#             pname, dname = name.rsplit(".", 1)
#             setattr(block.get_submodule(pname), dname, q_module)

#             # cleanup in an attempt to not run out of memory
#             del gptq
#             gc.collect()
#             torch.cuda.empty_cache()
#             t1 = time.perf_counter()
#             print(f"time {int(t1 - t0 + 0.5)}s quantization error {error:.1f}")

#         for j in range(inps.size(0)):
#             outs[j : j + 1] = block(inps[j : j + 1])

#         block.cpu()
#         gc.collect()
#         torch.cuda.empty_cache()

#         # the outputs are the next block's inputs and we'll reuse the old inputs
#         inps, outs = outs, inps

#     BabyGPTmodel.ln_f.to(working_device)
#     for j in range(inps.size(0)):
#         outs[j : j + 1] = BabyGPTmodel.ln_f(inps[j : j + 1])
#     BabyGPTmodel.ln_f.to("cpu")
#     inps, outs = outs, inps

#     BabyGPTmodel.lm_head.to(working_device)
#     gptq = GPTQQuantizer(
#         BabyGPTmodel.lm_head,
#         bits=bits,
#         groupsize=groupsize,
#         actorder=(groupsize == -1),
#     )
#     handle = BabyGPTmodel.lm_head.register_forward_hook(gptq.collect_input_stats)
#     for j in range(inps.size(0)):
#         BabyGPTmodel.lm_head(inps[j : j + 1])
#     handle.remove()
#     q_module, error = gptq.quantize()
#     BabyGPTmodel.lm_head = q_module
#     BabyGPTmodel.lm_head.to("cpu")

# def main(quantize: Optional[str] = None):    

#     device = 'cuda'
#     if quantize == "gptq.int4":
#         bits = 4
#     elif quantize == "gptq.int8":
#         bits = 8
#     else:
#         raise RuntimeError(f"unknown/unsupported quantization mode {quantize}")


    
#     t0 = time.perf_counter()
#     blockwise_quantization(BabyGPTmodel, device, bits=bits)

#     t = time.perf_counter() - t0
#     print(f"\n\nTime for quantization: {t:.02f} sec total",
#         file=sys.stderr,
#     )
#     print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB",
#         file=sys.stderr,)

# if __name__ == "__main__":
#     from jsonargparse import CLI

#     torch.set_float32_matmul_precision("high")
#     CLI(main)





@torch.no_grad()
def llama_blockwise_quantization(
        BabyGPTmodel, sample_inputs, working_device, *, bits=4, groupsize=-1
):
    # This is the classic post-training quantization
    # of all linear layers. We quantize in order, i.e.
    # when observing the inputs, we use the outputs
    # of the previously quantized layers rather than
    # doing them all at once.

    print("Getting inputs for first block")
    print(BabyGPTmodel)
    print(BabyGPTmodel.config)

    BabyGPTmodel.token.to(working_device)
    inps = []
    for batch in sample_inputs:
        inps.append(BabyGPTmodel.token(batch[None].to(working_device)))
    inps = torch.cat(inps, dim=0)
    BabyGPTmodel.token.to("cpu")
    torch.cuda.empty_cache()

    print("Starting to quantize blocks")
    outs = torch.zeros_like(inps)

    # better than relying on enumeration? originally the code bundled
    # the two mlp fc layers
    # we could automate this with a lot of hooks and another iteration
    submodules_to_process = [
        "train.babypt_trainer.Attention.attention",
        "train.babypt_trainer.Attention.projection",
    ]

    for i, block in enumerate(BabyGPTmodel.blocks):
        block.to(working_device)

        for name in submodules_to_process:
            print(i, name, end=" ")
            t0 = time.perf_counter()
            print("collecting stats", end=" ")
            sys.stdout.flush()
            module = block.get_submodule(name)

            gptq = GPTQQuantizer(
                module,
                bits=bits,
                groupsize=groupsize,
                actorder=(groupsize == -1),
            )
            handle = module.register_forward_hook(gptq.collect_input_stats)
            for j in range(inps.size(0)):
                outs[j : j + 1] = block(inps[j : j + 1])

            handle.remove()

            print("quantizing", end=" ")
            sys.stdout.flush()
            q_module, error = gptq.quantize()

            # replace the linear module with the quantized module
            pname, dname = name.rsplit(".", 1)
            setattr(block.get_submodule(pname), dname, q_module)

            # cleanup in an attempt to not run out of memory
            del gptq
            gc.collect()
            torch.cuda.empty_cache()
            t1 = time.perf_counter()
            print(f"time {int(t1 - t0 + 0.5)}s quantization error {error:.1f}")

        for j in range(inps.size(0)):
            outs[j : j + 1] = block(inps[j : j + 1])

        block.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # the outputs are the next block's inputs and we'll reuse the old inputs
        inps, outs = outs, inps

    BabyGPTmodel.ln_f.to(working_device)
    for j in range(inps.size(0)):
        outs[j : j + 1] = BabyGPTmodel.ln_f(inps[j : j + 1])
    BabyGPTmodel.ln_f.to("cpu")
    inps, outs = outs, inps

    BabyGPTmodel.lm_head.to(working_device)
    gptq = GPTQQuantizer(
        BabyGPTmodel.lm_head,
        bits=bits,
        groupsize=groupsize,
        actorder=(groupsize == -1),
    )
    handle = BabyGPTmodel.lm_head.register_forward_hook(gptq.collect_input_stats)
    for j in range(inps.size(0)):
        BabyGPTmodel.lm_head(inps[j : j + 1])
    handle.remove()
    q_module, error = gptq.quantize()
    BabyGPTmodel.lm_head = q_module
    BabyGPTmodel.lm_head.to("cpu")


def main(
    *,
    checkpoint_path: Path = Path("C://Users//Soumyadip Nandi//Downloads//policy//language//model.pth"),
    output_path: Optional[Path] = None,
    tokenizer_path: Path = Path("C://Users//Soumyadip Nandi//Downloads//policy//language//tokenizer.model"),
    n_samples: int = 128,
    dtype: str = "float32",
    quantize: Optional[str] = None,
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        # compile: Whether to compile the model.
        checkpoint_path: The checkpoint path to load.
        output_path: Path to write the quantized model's state dict to.
        tokenizer_path: The tokenizer path to load.
        n_samples: Number of example inputs to use for statistics (default: 128)
        dtype: The dtype to use to load the model.
        quantize: Mode to quantize the model to:
            ``"gptq.int4"``: GPTQ 4-bit mode.
            Note that ``"llm.int8"```does not need a quantization step.
    """
    assert checkpoint_path.is_file()
    assert tokenizer_path.is_file()

    device = "cuda"

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    if quantize == "gptq.int4":
        bits = 4
    elif quantize == "gptq.int8":
        bits = 8
    else:
        raise RuntimeError(f"unknown/unsupported quantization mode {quantize}")

    # we avoid loading the entire model on the GPU and do this block by block
    with EmptyInitOnDevice(
        device="cpu",
        dtype=dtype,
    ):
        print("Loading model ...", file=sys.stderr)
        t0 = time.time()
        checkpoint = torch.load(checkpoint_path)
        name = llama_model_lookup(checkpoint)
        # model = LLaMA.from_name(name)
        model.load_state_dict(checkpoint)
        print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()

    tokenizer = Tokenizer(tokenizer_path)

    test_string = get_sample_data()
    encoded_text = tokenizer.encode(
        test_string,
        bos=True,
        eos=False,
    )
    block_size = 2048  # this is for compat with gptq, and indeed we get much worse beyond this (https://github.com/facebookresearch/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L30)
    encoded_text = encoded_text[: n_samples * block_size].reshape(n_samples, block_size)
    t0 = time.perf_counter()

    llama_blockwise_quantization(model, encoded_text, device, bits=bits)

    torch.save(model.state_dict(), output_path)

    t = time.perf_counter() - t0
    print(
        f"\n\nTime for quantization: {t:.02f} sec total",
        file=sys.stderr,
    )
    print(
        f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB",
        file=sys.stderr,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)