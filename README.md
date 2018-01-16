# pytorch-tour
A deep learning tour with pytorch.

* https://www.youtube.com/watch?v=BZyKufPCKdI

Static vs dynamic graph frameworks. Static: Tf, caffe, keras, CNTK, dynamic:
pytorch, dynet, chainer, mxnet (gluon).

TF: Define placeholders, variables. Nothing is run yet. Framework construct the
graph. Derive gradients, loss function.

Execute iteration, executed in a VM.

* declarative, symbolic
* custom VM
* easy to serialize
* easy to optimize

----

PyTorch:

* Error location might be far away from the execution.
* Custom VM makes it harder to debug.
* Do not declare everything up-front.

----

Vs. Stacktrace points to line where it breaks.

Goal:

* minimize mental overhead
* leverage Python ecosystem
* allow free-form models
* keep things very fast

A pytorch.Tensor is just a np.ndarray.

```
import torch

x = torch.Tensor(5, 5) # Tensor creation.
y = x + x              # Arithmetic.
s = x[..., -1:]        # Slicing. Full basic + a large part of NumPy advanced indexing.
b = x + x[:, -1:]      # Slicing. Broadcasting.
z = x.sigmoid()        # Activation. A ton of math functions as methods (+ chaining)
q = torch.abs(x)       # Vectorized functions. Function syntax.
x.uniform_()           # In-place modifications.
i = x.long()           # Type cast.
print(x)               # Pretty print.
```

Native GPU support.

```
c = x.cuda() # Transfer tensor to GPU (0).

r = torch.matmul(c, c)
q = c ** 2 + r * c

d = c.cpu() # Transfer back.
print(q)
```

NumPy bridge, 0.5ms per call. Reallocating a few bytes. Data is not copied.

