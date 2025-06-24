<div align=center>

![image alt](https://github.com/araxnoid-code/ROTTA-rs/blob/main/assets/rotta-rs_logo_for_github.png?raw=true)


</div>

<div align=center>

# GUIDE
### version 0.0.3
</div align=center>


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

- divided
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

## Method and Function In Tensor
- index & index_replace
```rust
fn main() {
    // index
    let tensor = Tensor::new([
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0],
        ],[
         [7.0, 8.0, 9.0],
         [10.0, 11.0, 12.0]],
    ]);

    let a = tensor.index(vec![0, 0, 0]);
    println!("{}", a); // [1.0]

    let a = tensor.index(vec![0, 1]);
    println!("{}", a); // [4.0, 5.0, 6.0]

    let a = tensor.index(vec![-1, -1, -1]);
    println!("{}", a); // [12.0]

    // index_replace
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);
    tensor.set_requires_grad(false);

    tensor.index_replace(vec![1, 1], Tensor::new([1.0, 1.0, 1.0]));
    println!("{}", tensor)  // [
                            //  [
                            //   [1.0, 2.0, 3.0]
                            //   [4.0, 5.0, 6.0]
                            //  ]
                            //  [
                            //   [7.0, 8.0, 9.0]
                            //   [1.0, 1.0, 1.0]
                            //  ]
                            // ]


}
```

- to_shape
```rust
fn main() {
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);

    let a = tensor.to_shape(vec![6, 2]);
    println!("{a}") // [
                    //  [1.0, 2.0]
                    //  [3.0, 4.0]
                    //  [5.0, 6.0]
                    //  [7.0, 8.0]
                    //  [9.0, 10.0]
                    //  [11.0, 12.0]
                    // ]

}
```

- reshape
```rust
fn main() {
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);

    let a = tensor.reshape(vec![6, 2]);
    println!("{a}"); // [
                     //  [1.0, 2.0]
                     //  [3.0, 4.0]
                     //  [5.0, 6.0]
                     //  [7.0, 8.0]
                     //  [9.0, 10.0]
                     //  [11.0, 12.0]
                     // ]


    let a = tensor.reshape(vec![3, -1]);
    println!("{a}") // [
                    //  [1.0, 2.0, 3.0, 4.0]
                    //  [5.0, 6.0, 7.0, 8.0]
                    //  [9.0, 10.0, 11.0, 12.0]
                    // ]

}
```

- slice & slice_replace
```rust
fn main() {
    // slice
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);

    let a = tensor.slice(vec![
        ArrSlice(Some(0), Some(2)), 
        ArrSlice(Some(1), Some(2))
        ]);
    println!("{a}"); // [
                     //  [
                     //   [4.0, 5.0, 6.0]
                     //  ]
                     //  [
                     //   [10.0, 11.0, 12.0]
                     //  ]
                     // ]

    let a = tensor.slice(
        vec![
            ArrSlice(Some(0), Some(2)), 
            ArrSlice(Some(1), Some(2)), 
            ArrSlice(Some(1), Some(-1))]
        );
    println!("{a}"); // [
                     //  [
                     //   [5.0, 6.0]
                     //  ]
                     //  [
                     //   [11.0, 12.0]
                     //  ]
                     // ]
    
    // slice_replace
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);
    tensor.set_requires_grad(false);

    let replace_value = Tensor::new([
        [
            [1.0, 1.0, 1.0]
        ],
        [
            [1.0, 1.0, 1.0]]
        ]);
        
    tensor.slice_replace(
        vec![
            ArrSlice(Some(0), Some(2)), 
            ArrSlice(Some(1), Some(2))
            ],
        &replace_value
    );

    println!("{tensor}"); // [
                          //  [
                          //   [1.0, 2.0, 3.0]
                          //   [1.0, 1.0, 1.0]
                          //  ]
                          //  [
                          //   [7.0, 8.0, 9.0]
                          //   [1.0, 1.0, 1.0]
                          //  ]
                          // ]
    // note
    // ArrSlice(Some(0), Some(2))   => 0..2
    // ArrSlice(Some(0), None)      => 0..
    // ArrSlice(None, Some(2))      => ..2
    // ArrSlice(None, None)         => ..
}
```

- permute
```rust
fn main() {
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);

    let a = tensor.permute(vec![1, 0, 2]);
    println!("{}", a);  // [
                        //  [
                        //   [1.0, 2.0, 3.0]
                        //   [7.0, 8.0, 9.0]
                        //  ]
                        //  [
                        //   [4.0, 5.0, 6.0]
                        //   [10.0, 11.0, 12.0]
                        //  ]
                        // ]
}
```
- transpose
```rust
fn main() {
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);

    let a = tensor.transpose((1, 2));
    println!("{}", a);  // [
                        //  [
                        //   [1.0, 4.0]
                        //   [2.0, 5.0]
                        //   [3.0, 6.0]
                        //  ]
                        //  [
                        //   [7.0, 10.0]
                        //   [8.0, 11.0]
                        //   [9.0, 12.0]
                        //  ]
                        // ]


    let a = tensor.transpose((-3, -1));
    println!("{}", a);  // [
                        //  [
                        //   [1.0, 7.0]
                        //   [4.0, 10.0]
                        //  ]
                        //  [
                        //   [2.0, 8.0]
                        //   [5.0, 11.0]
                        //  ]
                        //  [
                        //   [3.0, 9.0]
                        //   [6.0, 12.0]
                        //  ]
                        // ]


    let a = tensor.t(); // same with tensor.transpose((-1, -2));
    println!("{}", a);  // [
                        //  [
                        //   [1.0, 4.0]
                        //   [2.0, 5.0]
                        //   [3.0, 6.0]
                        //  ]
                        //  [
                        //   [7.0, 10.0]
                        //   [8.0, 11.0]
                        //   [9.0, 12.0]
                        //  ]
                        // ]

}
```

- sum
```rust
fn main() {
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);

    let a = tensor.sum();
    println!("{}", a); // [78.0]
}
```

- sum_axis
```rust
fn main() {
    let tensor = Tensor::new([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
    ]);

    let a = tensor.sum_axis(1);
    println!("{}", a);  // [
                        //  [5.0, 7.0, 9.0]
                        //  [17.0, 19.0, 21.0]
                        // ]


    let a = tensor.sum_axis(-1);
    println!("{}", a);  // [
                        //  [6.0, 15.0]
                        //  [24.0, 33.0]
                        // ]

    let a = tensor.sum_axis_keep_dim(-1);
    println!("{}", a);  // [
                        //  [
                        //   [6.0]
                        //   [15.0]
                        //  ]
                        //  [
                        //   [24.0]
                        //   [33.0]
                        //  ]
                        // ]

}
```
- other functions
```rust
fn main() {
    let tensor = Tensor::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]);

    let absolut = tensor.abs();
    let power = tensor.powi(2);
    let natural_log = tensor.ln();
    let exp = tensor.exp();
    let sign = tensor.sign();
}
```

## Activation Function
```rust
fn main() {
    let tensor = Tensor::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]);

    let relu = relu(&tensor);
    let softplus = softplus(&tensor);
    let sigmoid = sigmoid(&tensor);
    let tanh = tanh(&tensor);
    let softmax = softmax(&tensor, -1);
}
```

## Loss Function
- sum square residual
```rust
fn main() {
    let sum_square_residual = SSResidual::init();

    let prediction = Tensor::new([[1.0, 2.0, 3.0]]);
    let actual = Tensor::new([[2.0, 3.0, 4.0]]);

    let loss = sum_square_residual.forward(&prediction, &actual);
}
```

- MSE
```rust
fn main() {
    let sum_square_residual = MSE::init();

    let prediction = Tensor::new([[1.0, 2.0, 3.0]]);
    let actual = Tensor::new([[2.0, 3.0, 4.0]]);

    let loss = sum_square_residual.forward(&prediction, &actual);
}
```

- MAE
```rust
fn main() {
    let sum_square_residual = MAE::init();

    let prediction = Tensor::new([[1.0, 2.0, 3.0]]);
    let actual = Tensor::new([[2.0, 3.0, 4.0]]);

    let loss = sum_square_residual.forward(&prediction, &actual);
}
```

- Cross Entropy Loss
```rust
fn main() {
    let sum_square_residual = CrossEntropyLoss::init();

    // [B, C]
    let prediction = Tensor::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]);
    let prob_prediction = softmax(&prediction, -1);

    // [B]
    let actual = Tensor::new([0.0, 1.0, 2.0]);

    let loss = sum_square_residual.forward(&prob_prediction, &actual);
    println!("{}", loss);
}
```

## Module
```rust
fn main() {
    let model = Module::init();
}
```

- seed(default=42)
```rust
fn main() {
    let mut model = Module::init();
    model.update_seed(43);
}
```

- weight initialization
```rust
pub enum WeightInitialization {
    Random,
    He, // Default
    Glorot,
}
```

```rust
fn main() {
    let mut model = Module::init();
    model.update_initialization(WeightInitialization::Glorot);
}
```

- function
```rust
fn main() {
    let mut model = Module::init();

    let linear = model.liniar_init(1, 4);
    let tensor = Tensor::new([[1.0]]);

    let x = linear.forward(&tensor);
    println!("{}", x);  // [
                        //  [-0.8411698816909696, -0.11660485427898945, 0.6455877424124907, 1.7693457512474318]
                        // ]
}
```

## Optimazer
- Sgd
```rust
fn main() {
    let mut model = Module::init();
    let optimazer = Sgd::init(model.parameters(), 0.00001);

    optimazer.zero_grad();
    optimazer.optim();
}
```

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