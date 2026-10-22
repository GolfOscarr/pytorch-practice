"""Optimizers from scratch: SGD, Adam and AdamW as real `torch.optim.Optimizer` subclasses.

Build three optimizers that are drop-in replacements for torch.optim.SGD, torch.optim.Adam and
torch.optim.AdamW. The tests train identical model copies with yours and with torch's, step by
step, and require the parameters to match.

Production skills this practices:
- Subclassing a framework base class correctly: let `Optimizer.__init__` build `param_groups`
  from a `defaults` dict, so per-group overrides, `zero_grad()`, `state_dict()`,
  `load_state_dict()` and LR schedulers all work without you writing them.
- Input validation: reject bad hyperparameters with a clear ValueError at construction time.
- State management: keep per-parameter state in `self.state[p]` (a defaultdict(dict)), created
  lazily on the first step that parameter has a gradient.

Rules:
- No torch.optim.SGD / Adam / AdamW (or torch.optim._functional) inside your code. Subclassing
  `torch.optim.Optimizer` is the point.
- `step(closure=None)` must be decorated with `@torch.no_grad()`. If a closure is given, call it
  under `torch.enable_grad()` and return its result; otherwise return None.
- Skip parameters whose `.grad` is None: no update, no weight decay, no state change.
  Check `p.grad is None` BEFORE touching `self.state[p]`: it's a defaultdict, so even reading
  `self.state[p]` creates an (empty) entry that then shows up in `state_dict()`.
- Read hyperparameters from `group[...]` inside `step`, every step. Different groups may have
  different lr (etc.), and LR schedulers change `group["lr"]` between steps.

Update rules (g = p.grad, wd = weight_decay, t = this parameter's step count, starting at 1):
- SGD:   g = g + wd * p
         if momentum: buf = g on the first step, else buf = momentum * buf + (1 - dampening) * g
                      g = g + momentum * buf if nesterov else buf
         p = p - lr * g
- Adam:  g = g + wd * p                                (L2, like torch.optim.Adam)
         m = b1 * m + (1 - b1) * g ;  v = b2 * v + (1 - b2) * g^2    (m, v start at zeros)
         p = p - (lr / (1 - b1^t)) * m / (sqrt(v) / sqrt(1 - b2^t) + eps)
- AdamW: same as Adam, but instead of adding wd * p to g, first do p = p * (1 - lr * wd).

Hints:
- Validation (match torch): lr >= 0, momentum >= 0, weight_decay >= 0, eps >= 0,
  0 <= beta < 1 for both betas, and nesterov needs momentum > 0 and dampening == 0.
- Do the math with in-place ops (`mul_`, `add_`, `addcmul_`, `addcdiv_`) on the state tensors
  and on `p` itself. Never rebind `p` or replace `p.data`.
- Keep the step count per parameter, not per optimizer. A parameter that missed steps has seen
  fewer updates and needs its own bias correction.
- Write the Adam denominator exactly as above (eps added after dividing by sqrt(1 - b2^t)).
  Algebraically equivalent forms give slightly different numbers.
- AdamW can share almost all of Adam's code. Changing the base class is fine.

Run:
    uv run pytest practice/p01_optimizers
    uv run pytest practice/p01_optimizers -rA      # list each test
"""

from collections.abc import Callable
from typing import overload

import torch
from torch.optim.optimizer import Optimizer, ParamsT


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum, Nesterov and L2 weight decay."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        """Validate hyperparameters and register them as defaults.

        Args:
            params: iterable of tensors, or of dicts defining parameter groups.
            lr: learning rate.
            momentum: momentum factor (0 disables momentum).
            dampening: dampening for momentum.
            weight_decay: L2 penalty coefficient.
            nesterov: use Nesterov momentum.

        Raises:
            ValueError: if lr, momentum or weight_decay is negative, or nesterov is set
                without momentum > 0 and dampening == 0.
        """
        raise NotImplementedError("Your code here")

    # The overloads mirror Optimizer.step's signature so type checkers accept the override.
    @overload
    def step(self, closure: None = None) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform one optimization step over every parameter group.

        Update rule, for each parameter p with a gradient g = p.grad
        (lr, momentum, dampening, weight_decay, nesterov come from p's group):

            g ← g + weight_decay · p                          # L2 penalty (skip if 0)

            if momentum != 0:
                buf ← g                                       # first step: buf = clone of g
                buf ← momentum · buf + (1 − dampening) · g    # every later step
                g   ← g + momentum · buf   if nesterov
                g   ← buf                  otherwise

            p ← p − lr · g

        State per parameter: self.state[p]["momentum_buffer"] (only when momentum != 0).

        Args:
            closure: optional callable that re-evaluates the model and returns the loss.

        Returns:
            The closure's loss, or None if no closure was given.
        """
        raise NotImplementedError("Your code here")


class Adam(Optimizer):
    """Adam with bias correction and L2 weight decay (added to the gradient)."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """Validate hyperparameters and register them as defaults.

        Args:
            params: iterable of tensors, or of dicts defining parameter groups.
            lr: learning rate.
            betas: decay rates for the first and second moment estimates.
            eps: term added to the denominator for numerical stability.
            weight_decay: L2 penalty coefficient.

        Raises:
            ValueError: if lr, eps or weight_decay is negative, or a beta is outside [0, 1).
        """
        raise NotImplementedError("Your code here")

    # The overloads mirror Optimizer.step's signature so type checkers accept the override.
    @overload
    def step(self, closure: None = None) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform one optimization step over every parameter group.

        Update rule, for each parameter p with a gradient g = p.grad
        (lr, (β1, β2) = betas, eps, weight_decay come from p's group):

            t ← t + 1                                         # this parameter's step count
            g ← g + weight_decay · p                          # L2, added to the gradient

            m ← β1 · m + (1 − β1) · g                         # 1st moment (mean)
            v ← β2 · v + (1 − β2) · g²                        # 2nd moment (uncentered var)

            bc1 = 1 − β1ᵗ                                     # bias corrections: m and v
            bc2 = 1 − β2ᵗ                                     # start at 0 and are too small

            p ← p − (lr / bc1) · m / (√v / √bc2 + eps)

        State per parameter, created on its first step:
            self.state[p]["step"] = t, ["exp_avg"] = m, ["exp_avg_sq"] = v   (m, v start at 0)

        Args:
            closure: optional callable that re-evaluates the model and returns the loss.

        Returns:
            The closure's loss, or None if no closure was given.
        """
        raise NotImplementedError("Your code here")


class AdamW(Optimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter)."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        """Validate hyperparameters and register them as defaults.

        Args:
            params: iterable of tensors, or of dicts defining parameter groups.
            lr: learning rate.
            betas: decay rates for the first and second moment estimates.
            eps: term added to the denominator for numerical stability.
            weight_decay: decoupled weight decay coefficient (note the nonzero default).

        Raises:
            ValueError: if lr, eps or weight_decay is negative, or a beta is outside [0, 1).
        """
        raise NotImplementedError("Your code here")

    # The overloads mirror Optimizer.step's signature so type checkers accept the override.
    @overload
    def step(self, closure: None = None) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform one optimization step over every parameter group.

        Update rule, for each parameter p with a gradient g = p.grad
        (lr, (β1, β2) = betas, eps, weight_decay come from p's group):

            t ← t + 1
            p ← p · (1 − lr · weight_decay)                   # decoupled decay, applied to p
                                                              # directly; g is NOT changed
            m ← β1 · m + (1 − β1) · g
            v ← β2 · v + (1 − β2) · g²

            bc1 = 1 − β1ᵗ
            bc2 = 1 − β2ᵗ

            p ← p − (lr / bc1) · m / (√v / √bc2 + eps)

        The only difference from Adam: weight decay shrinks p instead of being added to g,
        so it isn't rescaled by the adaptive denominator.

        State per parameter: same as Adam ("step", "exp_avg", "exp_avg_sq").

        Args:
            closure: optional callable that re-evaluates the model and returns the loss.

        Returns:
            The closure's loss, or None if no closure was given.
        """
        raise NotImplementedError("Your code here")
