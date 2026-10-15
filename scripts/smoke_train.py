import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

torch.manual_seed(0)

model = BasicsTransformerLM(
    vocab_size=32,
    context_length=8,
    d_model=32,
    num_layers=1,
    num_heads=4,
    d_ff=64,
    rope_theta=10000.0,
).cpu()
model.train()

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

tokens = torch.randint(0, 32, (2, 9))
x = tokens[:, :-1]
y = tokens[:, 1:]

before = model.lm_head.weight.detach().clone()

optimizer.zero_grad(set_to_none=True)

logits = model(x)
loss = cross_entropy(logits, y)

loss.backward()

print("logits shape:", tuple(logits.shape))
print("loss:", loss.item())
print("head gradient norm:", model.lm_head.weight.grad.norm().item())

optimizer.step()

change = (model.lm_head.weight.detach() - before).abs().max().item()
print("head max parameter change:", change)
print("parameter count:", sum(p.numel() for p in model.parameters()))


def tensor_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


parameter_bytes = tensor_bytes(model.parameters())
gradient_bytes = tensor_bytes(p.grad for p in model.parameters() if p.grad is not None)
optimizer_bytes = tensor_bytes(
    value for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor)
)

print("parameter bytes:", parameter_bytes)
print("gradient bytes:", gradient_bytes)
print("optimizer tensor bytes:", optimizer_bytes)
