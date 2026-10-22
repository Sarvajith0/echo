import torch
from transformers import ClapModel, ClapProcessor

def to_tensor(out):
    # older versions return a tensor, newer ones return a wrapper object
    return out if isinstance(out, torch.Tensor) else out.pooler_output


model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

inputs = proc(text=["calm piano music"], return_tensors="pt", padding=True)
with torch.no_grad():
    emb = to_tensor(model.get_text_features(**inputs))
print(emb.shape)   # torch.Size([1, 512])