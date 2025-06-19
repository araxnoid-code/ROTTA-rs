![image alt](https://github.com/araxnoid-code/ROTTA-rs/blob/main/assets/rotta-rs_logo_for_github.png?raw=true)

# ROTTA-rs
AI framework built on the rust programming language

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

## How To Make Tensor
the only data types possible on tensors are f64

There are 3 ways to create a tensor
```rust
mod rotta_rs;

fn main() {
    let tensor = Tensor::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]);

    //

    let tensor = Tensor::from_vector(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    //

    let arrayy = Arrayy::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]);

    let tensor = Tensor::from_arrayy(arrayy);
}
```

## Operations On Tensors
This version still has many shortcomings in the operations that can be performed on tensors, including:

- add
```rust
mod rotta_rs;

fn main() {
    let tensor_a = Tensor::new([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]);

    let tensor_b = Tensor::new([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]);

    let result = &tensor_a + &tensor_b;
    // or
    let result = add(&tensor_a, &tensor_b);
}
```

- devided
```rust
mod rotta_rs;

fn main() {
    let tensor_a = Tensor::new([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]);

    let tensor_b = Tensor::new([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]);

    let result = &tensor_a / &tensor_b;
    // or
    let result = divided(&tensor_a, &tensor_b);
}
```

- dot product
```rust
mod rotta_rs;

fn main() {
    let tensor_a = Tensor::new([1.0, 2.0, 3.0]);

    let tensor_b = Tensor::new([1.0, 2.0, 3.0]);

    let result = dot(&tensor_a, &tensor_b);
}
```

- matmul
```rust
mod rotta_rs;

fn main() {
    let tensor_a = Tensor::new([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]);

    let tensor_b = Tensor::new([
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
    ]);

    let result = matmul(&tensor_a, &tensor_b);
}
```

other operations
- exp
- sum axis

## How To Make AI Model
```rust
mod rotta_rs;

fn main() {
    let mut model = Module::init();
    let optimazer = Sgd::init(model.parameters(), 0.00001);
    let loss_fn = SSResidual::init();

    let linear = model.liniar_init(1, 1);
}
```

## Create Training
```rust
mod rotta_rs;

fn main() {
    let mut model = Module::init();
    let optimazer = Sgd::init(model.parameters(), 0.00001);
    let loss_fn = SSResidual::init();

    let linear = model.liniar_init(1, 1);
    let linear_2 = model.liniar_init(1, 1);

    let input = Tensor::new([[1.0], [2.0]]);
    let actual = Tensor::new([[1.0], [4.0]]);

    for epoch in 0..100 {
        let x = linear.forward(&input);
        let x = relu(&x);
        let output = linear_2.forward(&x);

        let loss = loss_fn.forward(&output, &actual);
        println!("epoch:{epoch} | loss => {loss}");

        optimazer.zero_grad();

        loss.backward();

        optimazer.optim();
    }
}
```

## Support Developer With Donations
- Trakteer

[https://trakteer.id/araxnoid/tip](https://trakteer.id/araxnoid/tip)

## follow me on social media to get the latest updates
- youtube

araxnoid

[click here to go directly to youtube](https://www.youtube.com/@araxnoid-v5o)

- tiktok

araxnoid

[click here to go directly to tiktok](https://www.tiktok.com/@araxnoid_code)

## contact
- Gmail

araxnoid0@gmail.com