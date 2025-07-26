# 📦 Version: `0.0.5`

### ✨ New Features
- `arange` method for create tensor
```rust
    let tensor = Tensor::arange(0, 10, 2); // (start, stop, step)
    println!("{}", tensor) // [0.0, 2.0, 4.0, 6.0, 8.0]
```

- `concat method for combining tensors in vectors`
```rust
fn main() {
    let tensor_a = Tensor::new([[1.0, 2.0, 3.0, 4.0, 5.0]]);
    let tensor_b = Tensor::new([[6.0, 7.0, 8.0, 9.0, 10.0]]);
    let vector = vec![&tensor_a, &tensor_b];

    let tensor = concat(vector, 0);
    println!("{}", tensor);     // [
                                //  [1.0, 2.0, 3.0, 4.0, 5.0]
                                //  [6.0, 7.0, 8.0, 9.0, 10.0]
                                // ]    
}
```

- `new method for slicing`
```rust
fn main() {
    let tensor_a = Tensor::arange(0, 12, 1).reshape(vec![-1, 3]);
    println!("{}", tensor_a);
    // [
    //  [0.0, 1.0, 2.0]
    //  [3.0, 4.0, 5.0]
    //  [6.0, 7.0, 8.0]
    //  [9.0, 10.0, 11.0]
    // ]

    // before 0.0.5
    let slicing = tensor_a.slice(vec![ArrSlice(Some(0), Some(2)), ArrSlice(Some(1), None)]);
    println!("{}", slicing);
    // [
    //  [1.0, 2.0]
    //  [4.0, 5.0]
    // ]

    // 0.0.5
    let slicing = tensor_a.slice(vec![r(0..2), r(1..)]);
    println!("{}", slicing)
    // [
    //  [1.0, 2.0]
    //  [4.0, 5.0]
    // ]
}
```

- `new method for sum axis and sum axis keep dim`
```rust
fn main() {
    let tensor_a = Tensor::arange(0, 12, 1).reshape(vec![-1, 3]);
    println!("{}", tensor_a);
    // [
    //  [0.0, 1.0, 2.0]
    //  [3.0, 4.0, 5.0]
    //  [6.0, 7.0, 8.0]
    //  [9.0, 10.0, 11.0]
    // ]

    // before 0.0.5
    // let sum = tensor_a.sum_axis(0); // in version 0.0.5, it can no longer be done

    // 0.0.5
    let slicing = tensor_a.sum_axis(&[0]);
    println!("{}", slicing);
    // [18.0, 22.0, 26.0]

    let slicing = tensor_a.sum_axis_keep_dim(&[0, 1]);
    println!("{}", slicing);
    // [
    //  [66.0]
    // ]
}
```

- `mean` & `mean axis` & `mean axis keep dim`

see more details in guide.md [tensor section](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/section/1_tensor.md)

- `RMSProp`

see more details in guide.md [Optimazer section](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/section/5_Optimazer.md)

- `Adam`

see more details in guide.md [Optimazer section](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/section/5_Optimazer.md)

- `Layer Norm`

see more details in guide.md [Module section](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/section/4_Module.md)

- `Batch Norm`

see more details in guide.md [Module section](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/section/4_Module.md)

- `Dataset` & `DataHandler`

see more details in guide.md [Dataset and DataHandler section](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/section/7_Dataset_and_DataHandler.md)


### 🚀 Optimizations
- implemented SIMD for matmul

### 🛠️ Bug Fixes
- Broadcast error during scalar operation
- Broadcast error when [x] will be broadcast to [1, x]


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
