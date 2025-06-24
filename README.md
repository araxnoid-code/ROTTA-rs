![image alt](https://github.com/araxnoid-code/ROTTA-rs/blob/main/assets/rotta-rs_logo_for_github.png?raw=true)

# ROTTA-rs
AI framework built on the rust programming language

## version 0.0.3
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

you can see other versions via this [link](https://github.com/araxnoid-code/ROTTA-rs/tags)

You can see what changes have occurred in the previous version at this [link](https://github.com/araxnoid-code/ROTTA-rs/blob/main/version.md)

## Install on your code
for now ROTTA-rs is still not available on crates.io, to use ROTTA-rs you can access the zip file on this link and extract it into your rust project.

[ROTTA-rs.zip](https://github.com/araxnoid-code/ROTTA-rs/blob/main/rotta_rs_module)

note: ROTTA-rs also uses external dependencies, don't forget to add them in Cargo.toml

|   dependencies    | version | features |
|   :-----------    | :------ | :---     |
|   `rand`          | 0.9.1   | _        |
|   `rand_distr`    | 0.5.1   | _        |
|   `uuid`          | 1.17.0  | `v4`     |

suggestion: for convenience you can extract it into the src folder along with the main.rs file then access the ROTTA-rs module using:
```rust
mod rotta_rs;
```


# general introduction
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

## Basic Operations On Tensors
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

- sub
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

    let result = &tensor_a - &tensor_b;
    // or
    let result = sub(&tensor_a, &tensor_b);
}
```

- mul
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

    let result = &tensor_a * &tensor_b;
    // or
    let result = mul(&tensor_a, &tensor_b);
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
- powi
- ln

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

## DEPEDENCIES
- [rand](https://crates.io/crates/rand)
- [rand_distr](https://crates.io/crates/rand_distr)
- [uuid](https://crates.io/crates/uuid)