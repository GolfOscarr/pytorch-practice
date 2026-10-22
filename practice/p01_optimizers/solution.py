"""Reference solution for p01_optimizers. Try optimizers.py yourself before reading this."""

import math
from collections.abc import Callable
from typing import overload

import torch
from torch.optim.optimizer import Optimizer, ParamsT


class SGD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        # Validate eagerly: a bad value should fail here, not as NaNs 10k steps later.
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening = 0")

        # The base class turns `params` into self.param_groups and fills every group's
        # missing keys from `defaults`. That is what lets users pass per-group overrides.
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    # no_grad: the updates below are in-place ops on leaf tensors that require grad.
    # Autograd must not record them (and would raise if it tried).
    # The overloads mirror Optimizer.step's signature so type checkers accept the override.
    @overload
    def step(self, closure: None = None) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            # The closure runs forward + backward, so it needs grad mode back on.
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Read hyperparameters from the group on every step (never cache them in
            # __init__): LR schedulers work by editing group["lr"] between steps.
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue  # unused this step: no update, no state change

                d_p = p.grad
                if weight_decay != 0:
                    # L2 penalty: gradient of (wd / 2) * ||p||^2 is wd * p.
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    state = self.state[p]  # defaultdict: {} the first time we see p
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        # First step: torch seeds the buffer with the gradient itself
                        # (no dampening), not with zeros.
                        buf = d_p.clone()  # already under no_grad, so no detach needed
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    d_p = d_p.add(buf, alpha=momentum) if nesterov else buf

                p.add_(d_p, alpha=-lr)

        return loss


class Adam(Optimizer):
    # Subclasses (AdamW) flip this to switch the weight-decay style; all else is shared.
    _decoupled_weight_decay = False

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    # The overloads mirror Optimizer.step's signature so type checkers accept the override.
    @overload
    def step(self, closure: None = None) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    # Lazy init on first real gradient. Storing everything in self.state
                    # (keyed by the parameter) is what makes state_dict()/load_state_dict()
                    # save and restore it for free.
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # The step count is per parameter: a param that skipped some steps
                # (grad None) has seen fewer updates and needs its own bias correction.
                state["step"] += 1
                step = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                if weight_decay != 0:
                    if self._decoupled_weight_decay:
                        # AdamW: shrink the weights directly, outside the adaptive update.
                        p.mul_(1 - lr * weight_decay)
                    else:
                        # Adam: L2 folded into the gradient, so it also gets rescaled by
                        # 1/sqrt(v). That coupling is exactly what AdamW removes.
                        grad = grad.add(p, alpha=weight_decay)

                # m = b1*m + (1-b1)*g ;  v = b2*v + (1-b2)*g^2   (all in place)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # m and v start at 0, so early on they are biased toward 0.
                # Dividing by (1 - beta^t) undoes that bias.
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # p -= lr * m_hat / (sqrt(v_hat) + eps), arranged the way torch does it
                # (eps is added after the sqrt-correction) so results match closely.
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

        return loss


class AdamW(Adam):
    _decoupled_weight_decay = True

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,  # AdamW's default is nonzero, unlike Adam's
    ) -> None:
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
