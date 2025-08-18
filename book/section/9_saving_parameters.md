# Saving Parameters

## Saving
`ROTTA-rs` stores parameters using JSON format.
```rust
fn main() {
    let mut model = Module::init();
    let linear_a = model.liniar_init(1, 8);
    let linear_b = model.liniar_init(8, 1);
    model.save("paramters.json");
}
```

## Load
Please remember, when you want to load parameters, make sure the model structure is the same.
```rust
fn main() {
    let mut model = Module::init();
    let linear_a = model.liniar_init(1, 8);
    let linear_b = model.liniar_init(8, 1);
    model.load_save("paramters.json");
}
```

## Use `struct`
If you want to save parameters, it would be better to save the model too. We can use `struct` to create a modular model that can be easily used with the parameters that have been saved.
```rust
struct MyModel {
    linear_a: Linear,
    linear_b: Linear,
}

impl MyModel {
    pub fn init(module: &mut Module) -> MyModel {
        Self {
            linear_a: module.liniar_init(1, 8),
            linear_b: module.liniar_init(8, 1),
        }
    }
}

fn main() {
    let mut model = Module::init();
    let my_model = MyModel::init(&mut model);
    model.save("parameters.json");
    // or
    model.load_save("parameters.json");
}
```