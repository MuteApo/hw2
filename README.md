# Homework 2

Public repository and stub/testing code for Homework 2 of 10-714.

## Dev Environment
- Windows 11
- Miniconda3 (with Python 3.10.6)

## Hints

Question 1
- Notice convention of `*args` and `**kwargs` during function call. `fan_in` & `fan_out` (`int`) will be packed/unpacked for `*shape` (`tuple`).

Question 2
- Tests will directly access `nn.Linear.bias`, initialize it even passed with `bias=False`.
- Slow down and check `.shape` when dealing with `nn.LogSumExp`. Avoid implicit broadcasting even if sometimes it works (because we have `numpy` backend as `array_api`). But in this case it will lead to unexpected behaviors.
- Do not apply equation $\mathrm{Var}(x)=\mathrm{E}(x^2)-\mathrm{E}^2(x)$ in `nn.LayerNorm1d`, it may cause floating point precision issues.
- `nn.BatchNorm1d` differs from train process and test process, on updating `running_mean` & `running_var` with real value (train) or just taking them for later calculation (test).
- `init.randb()` can be used in `nn.Dropout`.

Question 3
- When assigning updated value to `param`, use `.data` rather than `.detach()`. Because `optim.SGD.u` is a `dict` with reference to `param` as its key, just modify `param` in place.