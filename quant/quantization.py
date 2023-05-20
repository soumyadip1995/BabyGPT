import torch
import torch.nn as nn 
import math
import tqdm
import numpy as np  


class GPTQQuantizer:
    # The algorithm and code has been taken from  https://github.com/IST-DASLab/gptq/
    # E. Frantar et al GPTQ: Accurate Post-training Compression for GPT, arXiv:2210.17323
    # portions copyright by the authors licensed under the Apache License 2.0


    def __init__(
        self,
        linear_module,
        *,
        bits,
        perchannel=True,
        sym=False,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False
    ):
        assert isinstance(linear_module, torch.nn.Linear)

        self.linear_module = linear_module
        self.dev = self.linear_module.weight.device
        self.rows = linear_module.weight.shape[0]
        self.columns = linear_module.weight.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.bits = bits
        self.maxq = 2**bits - 1
        self.perchannel = perchannel
        self.sym = sym
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.groupsize = groupsize
        self.actorder = actorder
        self.tile_cols = self.columns if groupsize == -1 else groupsize
        self.scales = torch.zeros(
            (self.rows, (self.columns + self.tile_cols - 1) // self.tile_cols),
            dtype=self.linear_module.weight.dtype,
            device=self.dev,
        )
        self.zeros = torch.zeros_like(self.scales)
        assert not (
            self.actorder and self.groupsize != -1
        ), "The permutation trick does not work for grouped quantization"

    @staticmethod
    def quantize_weight(x, scale, zero, maxq):
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        x_rec = scale * (q - zero)
        return x_rec

    def find_params_weight(self, x):
        dev = x.device

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        scale = (xmax - xmin) / self.maxq
        if self.sym:
            zero = torch.full_like(scale, (self.maxq + 1) / 2)
        else:
            zero = torch.round(-xmin / scale)

        if not self.perchannel:
            tmp = shape[0]
            scale = scale.repeat(tmp)
            zero = zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        scale = scale.reshape(shape)
        zero = zero.reshape(shape)
        return scale, zero

    def collect_input_stats(self, _1, inp, _2):
        inp = inp[0].detach()
        self.last_inp = inp
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def quantize(self):
        W = self.linear_module.weight.detach().to(dtype=torch.float, copy=True)

        scale, zero = self.find_params_weight(W)
        self.scales[:] = scale
        self.zeros[:] = zero

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        if self.actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if self.groupsize != -1:
                    if (i1 + i) % self.groupsize == 0:
                        scale, zero = self.find_params_weight(
                            W[:, (i1 + i) : (i1 + i + self.groupsize)]
                        )
                        self.scales[:, (i1 + i) // self.groupsize] = scale
                        self.zeros[:, (i1 + i) // self.groupsize] = zeros

                q = self.quantize_weight(w.unsqueeze(1), scale, zero, self.maxq)
                q = q.squeeze(1)
                assert q.dim() == 1
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if self.actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]

        weight = Q.reshape(self.linear_module.weight.shape).to(
            self.linear_module.weight.data.dtype
        )
        error = torch.sum(Losses).item()

        q_module = ColBlockQuantizedLinear(
            self.linear_module.in_features,
            self.linear_module.out_features,
            self.linear_module.bias is not None,
            bits=self.bits,
            tile_cols=self.groupsize,
        ).to(self.dev)
        q_module.scales = self.scales
        q_module.zeros = self.zeros
        q_module.pack_weight(weight)
        q_module.bias = self.linear_module.bias
        return q_module, error