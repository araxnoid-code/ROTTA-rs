# Loss Function

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
    let mse = MSE::init();

    let prediction = Tensor::new([[1.0, 2.0, 3.0]]);
    let actual = Tensor::new([[2.0, 3.0, 4.0]]);

    let loss = mse.forward(&prediction, &actual);
}
```

- MAE
```rust
fn main() {
    let mae = MAE::init();

    let prediction = Tensor::new([[1.0, 2.0, 3.0]]);
    let actual = Tensor::new([[2.0, 3.0, 4.0]]);

    let loss = mae.forward(&prediction, &actual);
}
```

- Cross Entropy Loss
```rust
fn main() {
    let cel = CrossEntropyLoss::init();

    // [B, C]
    let prediction = Tensor::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]);
    let prob_prediction = softmax(&prediction, -1);

    // [B]
    let actual = Tensor::new([0.0, 1.0, 2.0]);

    let loss = cel.forward(&prob_prediction, &actual);
    println!("{}", loss);
}
```