from src.asr.decode import greedy_decode
import torch

def test_greedy_decode_shapes():
    logits = torch.zeros(5, 2, 4)  # (T,B,V)
    itos = {0:"<blank>",1:"a",2:"b",3:" "}
    out = greedy_decode(logits, itos)
    assert len(out) == 2
