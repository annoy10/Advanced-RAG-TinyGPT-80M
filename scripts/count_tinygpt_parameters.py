import torch.nn as nn
from pathlib import Path
import torch

ckpt_path=Path("transformer/checkpoint/tinygpt.pt")
ckpt=torch.load(ckpt_path,map_location="cpu",weights_only=False)
vocab_size=len(ckpt["vocabulary"])
d=512; layers=20; heads=8; block_size=256; ff=4*d
# Matches the supplied GPT architecture: token + position embeddings, untied vocab head,
# attention/FFN projections, two LayerNorms per block and final LayerNorm.
embedding=vocab_size*d
position=block_size*d
head=vocab_size*d+vocab_size
attention=4*(d*d+d)
feed_forward=d*ff+ff+ff*d+d
layer_norms=4*d
final_norm=2*d
total=embedding+position+head+layers*(attention+feed_forward+layer_norms)+final_norm
print(f"Vocabulary: {vocab_size:,}")
print(f"Architecture: d_model={d}, layers={layers}, heads={heads}, block_size={block_size}")
print(f"Trainable parameters: {total:,} ({total/1e6:.3f}M)")
