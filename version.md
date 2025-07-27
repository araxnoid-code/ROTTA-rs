## 📦 Version: `0.0.6`

### ✨ New Features
- `lstm function`
```rust
fn main() {
    let mut model = Module::init();
    let lstm = model.lstm_init(6);

    let tensor = Tensor::arange(0..6)
        .to_shape(vec![1, 6])
        .collect();
    println!("{}", tensor);
    // [
    //  [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    // ]

    let cell_hidden = lstm.forward(&tensor, None); //(&Tensor, Option<LSTMCellHidden>)
    println!("cell:\n{}", cell_hidden.cell);
    // cell:
    // [
    //  [-0.35452980609595336, -0.5292895572406782, 0.7244741335616286, -0.4735260362646916, -0.9515427099660342, -0.2374937433097324]
    // ]

    println!("hidden:\n{}", cell_hidden.hidden)
    // hidden:
    // [
    //  [-0.23028978337822023, -0.0435225356725021, 0.36147911006385536, -0.09738843166346652, -0.705630337819917, -0.051854840060327305]
    // ]
}
```

```rust
fn main() {
    LSTMCellHidden {
        cell: Tensor::new([[0.0]]),
        hidden: Tensor::new([[0.0]]),
    };
}
```

- `gru function`
```rust
fn main() {
    let mut model = Module::init();
    let lstm = model.gru_init(6);

    let tensor = Tensor::arange(0..6)
        .to_shape(vec![1, 6])
        .collect();
    println!("{}", tensor);
    // [
    //  [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    // ]


    let hidden = lstm.forward(&tensor, None); // (&Tensor, Option<Tensor>)
    println!("hidden:\n{}", hidden);
    // hidden:
    // [
    //  [0.2838769389632212, 0.30500443468104915, 0.03277033032455531, -0.4399917698902819, -0.9237180542375861, 0.3168019973639146]
    // ]
}
```

- `argmax method`
```rust
fn main() {
    let tensor = Tensor::rand(vec![3, 4]);
    println!("{}", tensor);
    // [
    //  [0.490653500989335, 0.9084708072918432, 0.5516854111601106, 0.38535295859939467]
    //  [0.5510768003422782, 0.04506790076914613, 0.36501508148354644, 0.48345186596013223]
    //  [0.05192318702409915, 0.03138051383420948, 0.5086337769325273, 0.021017655640771404]
    // ]

    let max = tensor.argmax(0);
    println!("{}", max);
    // [1.0, 0.0, 0.0, 1.0]

    let max = tensor.argmax(1);
    println!("{}", max)
    // [1.0, 0.0, 2.0]
}
```

- `argmin method`
```rust
fn main() {
    let tensor = Tensor::rand(vec![3, 4]);
    println!("{}", tensor);
    // [
    //  [0.8611219295171262, 0.9410276855253755, 0.9133598099213944, 0.22062707185602048]
    //  [0.47193516661684776, 0.7818072906711374, 0.8048492003479746, 0.9925399063075784]
    //  [0.3559389967244023, 0.3472829046036767, 0.7791381493184755, 0.8910867638091713]
    // ]

    let max = tensor.argmin(0);
    println!("{}", max);
    // [2.0, 2.0, 2.0, 0.0]

    let max = tensor.argmin(1);
    println!("{}", max)
    // [3.0, 0.0, 1.0]
}
```

- update `arange` method
```rust
fn main() {
    let tensor = Tensor::arange(0..12).collect();
    println!("{}", tensor);
    //[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]


    let tensor = Tensor::arange(0..24)
        .step(2)
        .to_shape(vec![3, 4])
        .map(|x| x * 2.0)
        .collect();
    println!("{}", tensor)
    // [
    //  [0.0, 4.0, 8.0, 12.0]
    //  [16.0, 20.0, 24.0, 28.0]
    //  [32.0, 36.0, 40.0, 44.0]
    // ]
}
```

- update `concat` method
```rust
fn main() {
    let tensor_a = Tensor::arange(0..3)
        .to_shape(vec![1, 3])
        .collect();
    let tensor_b = Tensor::arange(3..6)
        .to_shape(vec![1, 3])
        .collect();

    // before 0.0.6
    let tensors = vec![&tensor_a, &tensor_b];
    let concat = concat(tensors, 0);
    println!("{}", concat);
    // [
    //  [0.0, 1.0, 2.0]
    //  [3.0, 4.0, 5.0]
    // ]

    // after 0.0.6
    let concat = vec![&tensor_a, &tensor_b].concat_tensor(0);
    println!("{}", concat);
    // [
    //  [0.0, 1.0, 2.0]
    //  [3.0, 4.0, 5.0]
    // ]

    let concat = vec![tensor_a, tensor_b].concat_tensor(0);
    println!("{}", concat)
    // [
    //  [0.0, 1.0, 2.0]
    //  [3.0, 4.0, 5.0]
    // ]
}
```


- `flatten method`
```rust
fn main() {
    let tensor = Tensor::arange(0..12)
        .to_shape(vec![3, 4])
        .collect();
    println!("{}", tensor);
    // [
    //  [0.0, 1.0, 2.0, 3.0]
    //  [4.0, 5.0, 6.0, 7.0]
    //  [8.0, 9.0, 10.0, 11.0]
    // ]

    let flat = tensor.flatten();
    println!("{}", flat);
    // [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
}
```

- `sin, cos, tan method`
```rust
fn main() {
    let tensor = Tensor::arange(0..12)
        .to_shape(vec![3, 4])
        .collect();
    println!("{}", tensor);
    // [
    //  [0.0, 1.0, 2.0, 3.0]
    //  [4.0, 5.0, 6.0, 7.0]
    //  [8.0, 9.0, 10.0, 11.0]
    // ]


    let sin = tensor.sin();

    let cos = tensor.cos();

    let tan = tensor.tan();
}
```

- update `embedding` function
```rust
fn main() {
    let mut model = Module::init();
    let embedding = model.embedding_init(6, 4); //(vocab_num: usize, hidden: usize)
    println!("{}", embedding.parameter);
    // [
    //  [0.1818961923066713, 0.450275407672484, -0.0724835971434803, 0.12736052119734032]
    //  [-0.21140612085881738, -0.35004112970967505, -0.19195944040209034, 0.3038727671756267]
    //  [0.27124878080285697, -0.2614147356186607, 0.006866870340068942, 0.4018031720487738]
    //  [0.45699026891614114, 0.09297857032433399, 0.22879848650077528, -0.3208635907368216]
    //  [0.07539314660070717, -0.32823770909937444, -0.055947665079024045, -0.42360899591141266]
    //  [-0.34544166573751034, -0.17956923353240994, 0.1996879885620404, -0.42159451133374315]
    // ]

    // using the float data type because the usize data type that is generally used in indexing is not yet available
    let token = Tensor::new([0.0, 2.0, 5.0]);
    let embedded = embedding.forward(&token);
    println!("{}", embedded);
    // [
    //  [0.1818961923066713, 0.450275407672484, -0.0724835971434803, 0.12736052119734032]
    //  [0.27124878080285697, -0.2614147356186607, 0.006866870340068942, 0.4018031720487738]
    //  [-0.34544166573751034, -0.17956923353240994, 0.1996879885620404, -0.42159451133374315]
    // ]
}
```


### 🚀 Optimizations
- Optimizing Slicing Method

### 🔧 Change
- replace `StdRng` to `chacha8rng`

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
