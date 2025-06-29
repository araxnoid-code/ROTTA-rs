# Optimazer

- Sgd
```rust
fn main() {
    let mut model = Module::init();
    let optimazer = Sgd::init(model.parameters(), 0.00001);
}
```

- Sgd + Momentum
```rust
fn main() {
    let mut model = Module::init();
    let optimazer = SgdMomen::init(model.parameters(), 0.00001);
}
```

- AdaGrad
```rust
fn main() {
    let mut model = Module::init();
    let optimazer = AdaGrad::init(model.parameters(), 0.00001);
}
```