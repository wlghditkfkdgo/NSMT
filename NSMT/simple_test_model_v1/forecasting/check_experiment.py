"""Checks attention axes, shared head, trainable identity and ETT loader parity."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

from run_ett import NSMT, ETTWindows
from forecasting.simple_test_model_v1 import SimpleTestModelV1


def check(device, backend, parity=True):
    torch.set_num_threads(2)
    results = []
    for axis, embedding in (("population", False), ("temporal", False), ("temporal", True), ("none", False)):
        torch.manual_seed(7)
        model = SimpleTestModelV1(seq_len=25, pred_len=5, patch_size=4, num_population=7,
                                  d_model=16, num_heads=4, head_dim=9, attention_axis=axis,
                                  learn_population_embedding=embedding, backend=backend).to(device)
        captured, handles = {}, []
        def record(name):
            def hook(module, inputs, output):
                assert inputs[0].shape[:2] == (7, 6), (name, inputs[0].shape)
                captured[name] = output.detach()
            return hook
        for name, module in model.named_modules():
            if isinstance(module, MultiStepLIFNode):
                handles.append(module.register_forward_hook(record(name)))
        incoming = []
        if axis != "none":
            handles.append(model.blocks[0].attn.attn_lif.register_forward_pre_hook(
                lambda module, inputs: incoming.append(inputs[0].detach())))
        x = torch.randn(2,25,3,device=device)
        prediction, aux = model(x, return_aux=True)
        assert prediction.shape == (2,5,3) and aux["pad_right"] == 3
        if axis != "none":
            q, k, v = (captured[f"blocks.0.attn.{name}.lif"].reshape(7,6,7,4,4) for name in ("q","k","v"))
            if axis == "temporal":
                scores = torch.einsum("tbkhd,sbkhd->bkhts",q,k)
                expected = torch.einsum("bkhts,sbkhd->tbkhd",scores,v).flatten(-2)
            else:
                scores = torch.einsum("tbihd,tbjhd->tbhij",q,k)
                expected = torch.einsum("tbhij,tbjhd->tbihd",scores,v).flatten(-2)
            torch.testing.assert_close(aux["attention"][0], scores)
            torch.testing.assert_close(incoming[0], expected)
        target = torch.randn_like(prediction)
        (prediction-target).square().mean().backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert model.head_compress.weight.grad.abs().sum() > 0
        if embedding:
            assert model.population_embedding.grad.abs().sum() > 0
        for handle in handles:
            handle.remove()
        # The two-stage head must preserve time and K ordering exactly.
        z = aux["block_spikes"][-1]
        per_patch = torch.stack([model.head_compress(z[t].flatten(1)) for t in range(7)], 1)
        direct_head = model.head(per_patch.flatten(1)).reshape(2,3,5).transpose(1,2)
        mean = x.mean(1,keepdim=True)
        std = (x.var(1,unbiased=False,keepdim=True)+1e-5).sqrt()
        torch.testing.assert_close(prediction, direct_head*std+mean)
        with torch.no_grad():
            first = model(x)
            model(torch.randn(1,25,1,device=device))
            torch.testing.assert_close(model(x), first, rtol=0,atol=0)
        results.append({"axis":axis,"embedding":embedding,"status":"passed"})
    if parity:
        sys.path.insert(0,str(NSMT / "model_v1/forecasting"))
        from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute
        root = NSMT / "forecasting/dataset/ETT-small"
        for name in ("ETTh1","ETTh2","ETTm1","ETTm2"):
            for horizon in (96,720):
                mine = ETTWindows(root/(name+".csv"),name,96,horizon,"cpu")
                cls = Dataset_ETT_hour if name.startswith("ETTh") else Dataset_ETT_minute
                for split in ("train","val","test"):
                    ref = cls(SimpleNamespace(),str(root),flag=split,size=[96,0,horizon],features="M",
                              data_path=name+".csv",timeenc=1,freq="h" if name.startswith("ETTh") else "t")
                    assert len(ref) == mine.counts[split]
                    for index in (0,len(ref)-1):
                        x,y,*_ = ref[index]
                        a = mine.arrays[split]
                        np.testing.assert_array_equal(a[index:index+96].numpy(),x.astype(np.float32))
                        np.testing.assert_array_equal(a[index+96:index+96+horizon].numpy(),y.astype(np.float32))
                results.append({"dataset":name,"horizon":horizon,"loader_parity":"passed"})
    return {"device":device,"backend":backend,"checks":results,"status":"passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--backend",default="torch")
    parser.add_argument("--skip-parity",action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(args.device,args.backend,not args.skip_parity),indent=2))
