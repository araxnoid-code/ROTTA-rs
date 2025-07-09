# Module

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

- training & testing

``` rust
fn main() {
    let mut model = Module::init();

    // training phase
    model.train();

    // testing phase
    model.eval();
}
```

## function
- Linear
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

- Dropout
``` rust
fn main() {
    let mut model = Module::init();

    let linear = model.liniar_init(1, 4);
    let mut dropout = model.dropout_init(0.3);

    let tensor = Tensor::new([[1.0]]);

    let x = linear.forward(&tensor);
    let x = dropout.forward(&x);
    println!("{}", x);  // [
                        // [-0.0, -0.0, 0.9222682034464154, 2.527636787496331]
                        // ]
```