from types import MethodType
import triton
import triton.language as tl

@triton.jit
def _affine(X, W, B, Y, TOKENS: tl.constexpr, DIN: tl.constexpr, DOUT: tl.constexpr, HAS_BIAS: tl.constexpr, BM: tl.constexpr=16, BN: tl.constexpr=32, BK: tl.constexpr=32):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    batch = tl.program_id(2)
    acc = tl.full((BM, BN), 0, tl.float32)
    for offset in range(tl.cdiv(DIN, BK)):
        k = offset * BK + ks
        a = tl.load(X + batch * TOKENS * DIN + rows[:, None] * DIN + k[None, :], (rows[:, None] < TOKENS) & (k[None, :] < DIN), other=0)
        b = tl.load(W + cols[None, :] * DIN + k[:, None], (cols[None, :] < DOUT) & (k[:, None] < DIN), other=0)
        acc = tl.dot(a, b, acc, input_precision='ieee')
    if HAS_BIAS:
        acc = acc + tl.load(B + cols, cols < DOUT, other=0)[None, :]
    tl.store(Y + batch * TOKENS * DOUT + rows[:, None] * DOUT + cols[None, :], acc, (rows[:, None] < TOKENS) & (cols[None, :] < DOUT))

def install(model, torch):
    names = []

    def forward(self, x):
        if x.ndim != 3:
            return self._v32_original_forward(x)
        if x.dtype != torch.float32 or self.weight.dtype != torch.float32 or x.device.type != 'cuda':
            raise RuntimeError('Fixed V32 affine kernel requires FP32 CUDA inputs/weights')
        x = x.contiguous()
        w = self.weight.contiguous()
        batch, tokens, din = x.shape
        dout = w.shape[0]
        y = torch.empty((batch, tokens, dout), dtype=torch.float32, device=x.device)
        _affine[triton.cdiv(tokens, 16), triton.cdiv(dout, 32), batch](x, w, self.bias if self.bias is not None else w, y, TOKENS=tokens, DIN=din, DOUT=dout, HAS_BIAS=self.bias is not None, num_warps=4)
        return y
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, '_v32_original_forward'):
                raise RuntimeError('Another affine execution route is already active')
            module._v32_original_forward = module.forward
            module.forward = MethodType(forward, module)
            names.append(name)
    return names
