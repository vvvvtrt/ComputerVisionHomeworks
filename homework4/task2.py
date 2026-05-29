"""Triton LayerNorm: forward + backward + autotune + benchmark."""
import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bucketize_n(n: int) -> int:
    # bucketize hidden size для key autotune — чтобы кеш конфигов не раздувался
    # на каждое уникальное N
    return int(math.log2(max(n, 1)))


def _num_warps_for(block_size: int) -> int:
    # эвристика на дефолтный num_warps; реальный выбор делает autotune
    if block_size <= 512:
        return 2
    if block_size <= 2048:
        return 4
    if block_size <= 4096:
        return 8
    return 16


_FWD_CONFIGS = [
    triton.Config({}, num_warps=w) for w in (2, 4, 8, 16)
]

_BWD_CONFIGS = [
    triton.Config({}, num_warps=w) for w in (2, 4, 8, 16)
]


# ---------------------------------------------------------------------------
# forward kernel
# ---------------------------------------------------------------------------

@triton.autotune(configs=_FWD_CONFIGS, key=["N_bucket"])
@triton.jit
def _layernorm_fwd_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    Mean_ptr, Rstd_ptr,
    stride_xm, stride_ym,
    N, N_bucket,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    X_ptr += row * stride_xm
    Y_ptr += row * stride_ym

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # все промежуточные суммы — в fp32, иначе среднее/дисперсия теряют точность
    # при bf16/fp16 входе
    x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(Mean_ptr + row, mean)
    tl.store(Rstd_ptr + row, rstd)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_centered * rstd * w + b
    tl.store(Y_ptr + cols, y, mask=mask)


# ---------------------------------------------------------------------------
# backward kernel — один проход по строке: считает dx и atomic-аккумулирует dw/db
# ---------------------------------------------------------------------------

# reset_to_zero обязателен: autotune прогоняет kernel несколько раз для замера
# конфигов, а atomic_add по DW/DB иначе накапливается между попытками и портит
# результат первого реального вызова
@triton.autotune(
    configs=_BWD_CONFIGS,
    key=["N_bucket"],
    reset_to_zero=["DW_ptr", "DB_ptr"],
)
@triton.jit
def _layernorm_bwd_kernel(
    X_ptr, W_ptr, DY_ptr, DX_ptr,
    DW_ptr, DB_ptr,
    Mean_ptr, Rstd_ptr,
    stride_xm, stride_dym, stride_dxm,
    N, N_bucket,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    X_ptr += row * stride_xm
    DY_ptr += row * stride_dym
    DX_ptr += row * stride_dxm

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(Mean_ptr + row).to(tl.float32)
    rstd = tl.load(Rstd_ptr + row).to(tl.float32)

    x_hat = (x - mean) * rstd
    dy_hat = dy * w

    # dx = (rstd/N) * (N*dy_hat - sum(dy_hat) - x_hat * sum(dy_hat * x_hat))
    sum_dy_hat = tl.sum(tl.where(mask, dy_hat, 0.0), axis=0)
    sum_dy_hat_xhat = tl.sum(tl.where(mask, dy_hat * x_hat, 0.0), axis=0)
    dx = (rstd / N) * (N * dy_hat - sum_dy_hat - x_hat * sum_dy_hat_xhat)
    tl.store(DX_ptr + cols, dx, mask=mask)

    # atomic-аккумуляция dW/dB в fp32-буфер (atomic_add по bf16/fp16 ненадёжен)
    tl.atomic_add(DW_ptr + cols, dy * x_hat, mask=mask)
    tl.atomic_add(DB_ptr + cols, dy, mask=mask)


# ---------------------------------------------------------------------------
# python wrappers
# ---------------------------------------------------------------------------

def _flatten_to_2d(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.reshape(-1, n)


def layernorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not x.is_contiguous():
        x = x.contiguous()

    n = weight.shape[0]
    x_2d = _flatten_to_2d(x, n)
    m = x_2d.shape[0]

    y = torch.empty_like(x_2d)
    mean = torch.empty(m, device=x.device, dtype=torch.float32)
    rstd = torch.empty(m, device=x.device, dtype=torch.float32)

    # BLOCK_SIZE >= N — строка целиком влезает в один тайл; autotune перебирает
    # только num_warps, а размер тайла адаптивен по N
    block_size = triton.next_power_of_2(n)

    _layernorm_fwd_kernel[(m,)](
        x_2d, y, weight, bias,
        mean, rstd,
        x_2d.stride(0), y.stride(0),
        N=n,
        N_bucket=_bucketize_n(n),
        eps=eps,
        BLOCK_SIZE=block_size,
    )
    return y.view_as(x), mean, rstd


def layernorm_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not dy.is_contiguous():
        dy = dy.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()

    n = weight.shape[0]
    x_2d = _flatten_to_2d(x, n)
    dy_2d = _flatten_to_2d(dy, n)
    m = x_2d.shape[0]

    dx_2d = torch.empty_like(x_2d)
    # fp32-аккумуляторы, ниже кастуем обратно в dtype weight/bias
    dw_fp32 = torch.zeros(n, device=x.device, dtype=torch.float32)
    db_fp32 = torch.zeros(n, device=x.device, dtype=torch.float32)

    block_size = triton.next_power_of_2(n)

    _layernorm_bwd_kernel[(m,)](
        x_2d, weight, dy_2d, dx_2d,
        dw_fp32, db_fp32,
        mean, rstd,
        x_2d.stride(0), dy_2d.stride(0), dx_2d.stride(0),
        N=n,
        N_bucket=_bucketize_n(n),
        BLOCK_SIZE=block_size,
    )

    return dx_2d.view_as(x), dw_fp32.to(weight.dtype), db_fp32.to(weight.dtype)


# ---------------------------------------------------------------------------
# autograd + nn.Module
# ---------------------------------------------------------------------------

class _LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        y, mean, rstd = layernorm_forward(x, weight, bias, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, mean, rstd = ctx.saved_tensors
        dx, dw, db = layernorm_backward(dy, x, weight, mean, rstd)
        return dx, dw, db, None


class TritonLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5,
                 device=None, dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        # поддерживаем только нормализацию по последней оси (как в SPEC)
        if len(normalized_shape) != 1:
            raise ValueError("TritonLayerNorm supports 1-D normalized_shape only")
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(normalized_shape, **factory))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, **factory))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _LayerNormFn.apply(x, self.weight, self.bias, self.eps)


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------

def _check_correctness():
    device = "cuda"
    torch.manual_seed(0)

    shapes_and_dtypes = [
        ((8, 512, 1024), torch.float32),
        ((8, 512, 1024), torch.bfloat16),
        ((4, 128, 199), torch.bfloat16),     # не-степень-двойки
        ((2, 64, 1023), torch.float32),      # не-степень-двойки
        ((16, 256, 4096), torch.bfloat16),
    ]

    for shape, dtype in shapes_and_dtypes:
        n = shape[-1]
        x_ref = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        x_mine = x_ref.detach().clone().requires_grad_(True)

        w_ref = torch.randn(n, device=device, dtype=dtype, requires_grad=True)
        b_ref = torch.randn(n, device=device, dtype=dtype, requires_grad=True)
        w_mine = w_ref.detach().clone().requires_grad_(True)
        b_mine = b_ref.detach().clone().requires_grad_(True)

        y_ref = torch.nn.functional.layer_norm(x_ref, (n,), w_ref, b_ref, eps=1e-5)
        y_mine = _LayerNormFn.apply(x_mine, w_mine, b_mine, 1e-5)

        atol, rtol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-5, 1e-5)
        torch.testing.assert_close(y_mine, y_ref, atol=atol, rtol=rtol)

        # одинаковый upstream grad для честного сравнения
        g = torch.randn_like(y_ref)
        y_ref.backward(g)
        y_mine.backward(g)

        # для градиентов допуски чуть мягче (особенно для bf16 — больше операций)
        atol_g, rtol_g = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-4, 1e-4)
        torch.testing.assert_close(x_mine.grad, x_ref.grad, atol=atol_g, rtol=rtol_g)
        torch.testing.assert_close(w_mine.grad, w_ref.grad, atol=atol_g, rtol=rtol_g)
        torch.testing.assert_close(b_mine.grad, b_ref.grad, atol=atol_g, rtol=rtol_g)

        print(f"ok  shape={shape} dtype={dtype}")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def _torch_eager(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (weight.shape[0],), weight, bias, eps=1e-5)


_torch_compiled = torch.compile(_torch_eager)


def _triton_fwd(x, weight, bias):
    return _LayerNormFn.apply(x, weight, bias, 1e-5)


_PROVIDERS = {
    "triton": _triton_fwd,
    "torch-eager": _torch_eager,
    "torch-compile": _torch_compiled,
}


def run_benchmark():
    @triton.testing.perf_report([
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[512, 1024, 2048, 4096, 8192],
            line_arg="provider",
            line_vals=list(_PROVIDERS.keys()),
            line_names=list(_PROVIDERS.keys()),
            styles=[("blue", "-"), ("red", "--"), ("green", "--")],
            ylabel="GB/s",
            plot_name="layernorm_fwd_bf16_M4096",
            args={"M": 4096, "mode": "fwd"},
        ),
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[512, 1024, 2048, 4096, 8192],
            line_arg="provider",
            line_vals=list(_PROVIDERS.keys()),
            line_names=list(_PROVIDERS.keys()),
            styles=[("blue", "-"), ("red", "--"), ("green", "--")],
            ylabel="GB/s",
            plot_name="layernorm_fwd_bwd_bf16_M4096",
            args={"M": 4096, "mode": "fwd+bwd"},
        ),
    ])
    def benchmark(N: int, M: int, mode: str, provider: str):
        dtype = torch.bfloat16
        device = "cuda"

        x = torch.randn((M, N), device=device, dtype=dtype, requires_grad=(mode == "fwd+bwd"))
        weight = torch.randn(N, device=device, dtype=dtype, requires_grad=(mode == "fwd+bwd"))
        bias = torch.randn(N, device=device, dtype=dtype, requires_grad=(mode == "fwd+bwd"))
        fn_provider = _PROVIDERS[provider]

        if mode == "fwd":
            def run():
                return fn_provider(x, weight, bias)
        else:
            dy = torch.randn_like(x)

            def run():
                # каждый запуск — независимый граф, чтобы backward не падал на retain_graph
                x_ = x.detach().requires_grad_(True)
                y = fn_provider(x_, weight, bias)
                y.backward(dy, retain_graph=False)

        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=[0.5, 0.2, 0.8])

        # bandwidth: fwd читает x и пишет y → 2*M*N elem; bwd дополнительно
        # читает dy, x, mean/rstd и пишет dx → ещё ~3*M*N. weight/bias малы.
        elem = M * N * x.element_size()
        total_bytes = 2 * elem if mode == "fwd" else 5 * elem

        def gbps(t: float) -> float:
            return (total_bytes * 1e-9) / (t * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(save_path=".", show_plots=False, print_data=True)


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    print("== correctness ==")
    _check_correctness()

    print("\n== benchmark ==")
    run_benchmark()
