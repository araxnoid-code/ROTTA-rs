# 📦 Version: `0.0.4`

### ✨ New Features
- `Dropout`
- `SGD + Momentum`
- `AdaGrad`
- `powf`
- `train` and `val` method for [module](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/section/4_Module.md)
- New method for creating `tensors`


### 🚀 Optimizations
- Optimized Basic Operations `add`, `sub`, `mul`, `div`, `matmul`

### 🛠️ Bug Fixes
- Fixed bug on `Sum Square Residual`
- Fixed a bug where tensors accumulated their gradients

# version 0.0.3
### New
- negative indexing for sum_axis, slice, indexing, reshape.
- transpose
- reshape
- to_shape
- slice
- permute
- requires_grad
- Mean Absolute Error
- Mean Square Error
- sign
- abs
- sum function

### Optimalization
- optimizing operations with scalars

### fix bug
- fixed a bug in basic math in arrayy
- fixed backward error in multiple tensor

# version 0.0.2
### New
- Softplus
- ln
- powi
- sigmoid
- mul operation for tensor
- sub operation for tensor
- rename 'reshape' method to 'to_shape' in Arrayy
- update the algorithm of cross entropy loss
- update the algorithm of indexing on Arrayy

### fix bug
- fix bug in tensor broadcasting
- fix bug in derivative of divided for tensor

# version 0.0.1
tensor
- powered by arrayy(look in the folder with the name arrayy)

optimazer
- SGD

loss function
- Sum Square Residual
- Cross Entropy Loss

activation function
- Relu
- Softmax

module
- linear function
- has 3 weight initialization methods(Random, Glorot(default), He)

How to change weight initialization:
```rust
mod rotta_rs;

fn main() {
    let mut model = Module::init();
    model.update_initialization(WeightInitialization::He);
}
```
- default seed is 42

How to change module seed:
```rust
mod rotta_rs;

fn main() {
    let mut model = Module::init();
    model.update_seed(43);
}
```
