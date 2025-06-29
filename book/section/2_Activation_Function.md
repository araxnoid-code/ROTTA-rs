# Activation Function
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