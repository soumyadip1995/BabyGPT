## from lit-llama 
## from karpathy(export only)



import os
import struct

from logging import getLogger
from typing import List
from typing import Optional
from pathlib import Path

from sentencepiece import SentencePieceProcessor, SentencePieceTrainer


class Tokenizer:
    def __init__(self)-> None:
        
        model_path = "tokenizer.model"
        
        self.processor = SentencePieceProcessor(model_file = str(model_path))
        self.n_words : int = self.processor.vocab_size()
        self.bos_id: int = self.processor.bos_id()
        self.eos_id: int = self.processor.eos_id()
        self.pad_id: int = self.processor.pad_id()


    @property
    def vocab_size (self):
        self.n_words: int = self.processor.vocab_size()
        return self.n_words

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.processor.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.processor.decode(t)


    @staticmethod
    def train(input: str, destination: str, vocab_size=32000) -> None:
        model_prefix = os.path.join(destination, "tokenizer")
        SentencePieceTrainer.Train(input=input, model_prefix=model_prefix, vocab_size=vocab_size)

    def export(self):
        tokens , scores = [], []
        for i in range(self.n_words):
            t = self.processor.id_to_piece(i)
            s = self.processor.get_score(i)
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            elif len(t) == 6 and t.startswith('<0x') and t.endswith('>'):
                t = chr(int(t[3:5], 16)) # e.g. make '<0x01>' into '\x01'
            t = t.replace('▁', ' ') # sentencepiece uses this character as whitespace
            b = t.encode('utf-8') # bytes of this token, utf-8 encoded

            tokens.append(b)
            scores.append(s)

            # record the max token length
            max_token_length = max(len(t) for t in tokens)

            # write to a binary file
            with open("tokenizer.bin", 'wb') as f:
                f.write(struct.pack("I", max_token_length))
                for bytes, score in zip(tokens, scores):

                    f.write(struct.pack("fI", score, len(bytes)))
                    f.write(bytes)


if __name__ == "__main__":
   x = Tokenizer.train("C://Users//Soumyadip Nandi//Downloads//policy//BabyGPT//data//ALL_eminem.txt", "C://Users//Soumyadip Nandi//Downloads//policy//", vocab_size = 10000)
   t = Tokenizer()
   t.export()


       