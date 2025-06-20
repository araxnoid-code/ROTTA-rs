## version 0.0.1
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
