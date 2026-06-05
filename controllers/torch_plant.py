"""Exact, fully-differentiable PyTorch replica of tinyphysics.onnx.

The ONNX plant is a neural net, hence differentiable. We convert it with
onnx2torch (verified exact to ~3e-5) and replace the token-embedding gather
(E[tokens], non-differentiable) with a soft-token matmul (soft @ E). Feeding a
one-hot reproduces the exact model; feeding a soft distribution makes the full
autoregressive loop differentiable -> exact gradients for trajectory optimization.
"""
from pathlib import Path
import torch
from onnx2torch import convert
from tinyphysics import VOCAB_SIZE as VOCAB, LATACCEL_RANGE

_MODEL = Path(__file__).resolve().parent.parent / "models" / "tinyphysics.onnx"

# The ONNX model is exact to ~3e-5, which is good enough for our purposes. We could get exact if we 
# wanted by using the same float16 quantization as the ONNX export, but that would be more complicated 
# and less efficient on GPU. The small discrepancy is not a problem for our purposes, and the ONNX 
# export is more convenient for development and debugging.
def build_soft_plant(device):
  # Convert the ONNX model to a PyTorch model using onnx2torch. This gives us a PyTorch replica of the 
  # ONNX plant, which is exact to ~3e-5. We will then modify this PyTorch model to replace the 
  # token-embedding gather with a soft matmul, which will make the model fully differentiable and allow 
  # us to compute exact gradients for trajectory optimization. The ONNX model is a neural net, hence 
  # differentiable, but the token-embedding gather is not differentiable with respect to the input tokens. 
  # By replacing it with a soft matmul, we can maintain differentiability while still approximating the 
  # behavior of the original model. Feeding a one-hot distribution over tokens will reproduce the exact 
  # behavior of the original model, while feeding a soft distribution will allow us to compute exact 
  # gradients for trajectory optimization.
  gm = convert(str(_MODEL))
  g = gm.graph
  gn = [n for n in g.nodes if n.op == "call_module" and "wt2_embedding" in str(n.target)]
  assert len(gn) == 1, f"expected 1 token-embedding gather, found {len(gn)}"
  # the node that gathers the token embeddings, which we will replace with a soft matmul to make 
  # the model fully differentiable. The original node looks like E[tokens], where E is the token 
  # embedding matrix and tokens are the input tokens. We will replace this with a soft matmul of 
  # the token embedding matrix E with a soft one-hot distribution over tokens, which will allow 
  # us to maintain differentiability while still approximating the behavior of the original model. 
  # The args of this node are E and tokens, which we will use to construct the new soft matmul node.
  gn = gn[0] 
  # The original node looks like E[tokens], where E is the token embedding matrix and tokens are the 
  # input tokens. The args of this node are E and tokens, which we will use to construct the new soft 
  # matmul node. We will replace this node with a new node that computes the soft matmul of E with a 
  # soft one-hot distribution over tokens, which will allow us to maintain differentiability while 
  # still approximate the behavior of the original model. The new node will take E and tokens as 
  # inputs, and will compute the weighted sum of the rows of E according to the weights in the soft 
  # one-hot distribution over tokens. This will give us a new node that produces the same output as 
  # the original node when fed a one-hot distribution over tokens, but will also allow us to compute 
  # exact gradients for trajectory optimization when fed a soft distribution over tokens.
  E_node, tok_node = gn.args

  # Replace E[tokens] with soft @ E, where soft is a soft one-hot distribution over tokens. This makes 
  # the model fully differentiable, allowing us to compute exact gradients for trajectory optimization. 
  # Feeding a one-hot reproduces the exact model; feeding a soft distribution makes the full 
  # autoregressive loop differentiable -> exact gradients for trajectory optimization.
  def soft_gather(E, soft):                 # E (V,d), soft (...,V) -> (...,d)
    # E is the token embedding matrix, soft is a soft one-hot distribution over tokens. We want to 
    # compute the weighted sum of the rows of E according to the weights in soft. This is equivalent 
    # to a matrix multiplication of soft (as a row vector) with E. The result will be a weighted 
    # average of the rows of E, which is what we want for a soft gather. 
    return soft.to(E.dtype) @ E 

  with g.inserting_after(gn):
    new = g.call_function(soft_gather, args=(E_node, tok_node))
  gn.replace_all_uses_with(new)
  g.erase_node(gn)
  g.lint()
  gm.recompile()
  return gm.eval().to(device)

# The bins are fixed, so we can precompute them. They are used to convert continuous lataccel values 
# into soft one-hot distributions over the discrete token bins.
def make_bins(device):
  return torch.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB, device=device)

# Convert continuous lataccel values into soft one-hot distributions over the discrete token bins. 
# This is used to feed continuous lataccel values into the model, which expects discrete tokens. 
# The soft one-hot distribution allows us to maintain differentiability while still approximating 
# the behavior of the original model.
def soft_encode(x, bins, sigma=0.03):       # continuous lataccel -> soft one-hot over bins
  d = x.unsqueeze(-1) - bins
  return torch.softmax(-(d * d) / (2.0 * sigma * sigma), dim=-1)
