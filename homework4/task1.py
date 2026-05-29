import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    # pin_memory + num_workers — асинхронная H2D-передача, оверлапится со step'ом
    dataloader = DataLoader(
        prepare_data(),
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).cuda().train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # храним loss как detached-тензор на GPU — .item() в конце эпохи, чтобы
    # не форсить cuda.synchronize() на каждом батче
    losses_gpu = []
    forward_times_ms = []
    backward_times_ms = []

    # первые батчи: JIT/cuBLAS init, autotune, аллокаторный warm-up — выбрасываем
    # из статистики
    WARMUP_BATCHES = 3
    # печатать loss раз в K батчей — синхронный print каждый шаг ломает overlap
    LOG_EVERY = 10

    start_fwd = torch.cuda.Event(enable_timing=True)
    end_fwd = torch.cuda.Event(enable_timing=True)
    start_bwd = torch.cuda.Event(enable_timing=True)
    end_bwd = torch.cuda.Event(enable_timing=True)

    epoch_start = time.time()

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to('cuda', non_blocking=True)
        target = target.to('cuda', non_blocking=True)
        # шум создаём сразу на GPU — без CPU-копии и без блокирующего H2D
        noise = torch.randn_like(data)
        data = data + noise

        # set_to_none=True — меньше memset'ов, дефолт на новых PyTorch
        optimizer.zero_grad(set_to_none=True)

        start_fwd.record()
        output = model(data)
        loss = criterion(output, target)
        end_fwd.record()

        start_bwd.record()
        loss.backward()
        end_bwd.record()

        optimizer.step()

        # .detach() — рвём autograd-граф, иначе активации forward'а держатся
        # до конца эпохи и дают OOM
        losses_gpu.append(loss.detach())

        if batch_idx >= WARMUP_BATCHES:
            # синхронизация только на завершающем event'е окна — измеряем
            # реальное время kernel'ов, а не launch overhead
            end_bwd.synchronize()
            forward_times_ms.append(start_fwd.elapsed_time(end_fwd))
            backward_times_ms.append(start_bwd.elapsed_time(end_bwd))

        if batch_idx % LOG_EVERY == 0:
            print(f"Batch {batch_idx} loss: {loss.item():.4f}")

    # один синк в конце — батчевый перевод GPU-скаляров в python float
    torch.cuda.synchronize()
    losses_history = [t.item() for t in losses_gpu]

    epoch_time = time.time() - epoch_start

    def _summary(name: str, xs: list[float]) -> str:
        return (
            f"{name}: avg={statistics.mean(xs):.3f}ms "
            f"median={statistics.median(xs):.3f}ms "
            f"p90={statistics.quantiles(xs, n=10)[-1]:.3f}ms"
        )

    print(
        f"Epoch finished in {epoch_time:.2f}s, "
        f"final loss={losses_history[-1]:.4f}\n"
        f"  {_summary('forward', forward_times_ms)}\n"
        f"  {_summary('backward', backward_times_ms)}"
    )


if __name__ == '__main__':
    train()
