import math
import torch
from torch.optim.optimizer import ParamsT
from collections.abc import Callable, Iterable

PI = math.pi


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        # User might pass in a callable closure to re-compute the loss before the optimizer step.
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                # A dictionary-like object that stores persistent buffers associated with the model's parameters.
                state = self.state[p]
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: tuple[float | torch.Tensor, float | torch.Tensor] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            eps=eps,
            betas=betas,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 0 == len(state):
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)  # first moment
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                g = p.grad
                # 相当于 state["exp_avg"].mul_(betas[0]).add_((1 - betas[0]) * g)
                state["exp_avg"].mul_(betas[0]).add_(g, alpha=1 - betas[0])
                state["exp_avg_sq"].mul_(betas[1]).add_(torch.square(g), alpha=(1 - betas[1]))
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                bias_correction1 = 1 - betas[0] ** state["step"]
                bias_correction2 = 1 - betas[1] ** state["step"]
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                p.data.mul_(1 - lr * group["weight_decay"])
                p.data.sub_(step_size * m / (torch.sqrt(v) + group["eps"]))


def cosine_learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    lr: float = 0.0
    if 0 <= it < warmup_iters:
        lr = it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        lr = min_learning_rate + 0.5 * (
            1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * PI)
        ) * (max_learning_rate - min_learning_rate)
    elif cosine_cycle_iters < it:
        lr = min_learning_rate
    else:
        raise ValueError(f"Invalid iteration: {it}")
    return lr


if __name__ == "__main__":
    from datetime import date

    print(date.today())
    for lr in list([1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]):
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        print(f"{'=' * 10} lr is {lr} {'=' * 10} ")
        opt = SGD([weights], lr=1)
        for t in range(10):
            opt.zero_grad()  # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean()  # Compute a scalar loss value.
            print(f"{loss.cpu().item():.5f}")
            loss.backward()  # Run backward pass, which computes gradients.
            opt.step()  # Run optimizer step.
