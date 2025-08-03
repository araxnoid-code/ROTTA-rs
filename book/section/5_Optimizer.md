# Optimazer

- Sgd
```rust
fn main() {
    let mut model = Module::init();
    let optimazer = Sgd::init(model.parameters(), 0.001);
}
```

- Sgd + Momentum
```rust
fn main() {
    let mut model = Module::init();
    let optimazer = SgdMomen::init(model.parameters(), 0.001);
}
```

- AdaGrad
```rust
fn main() {
    let mut model = Module::init();
    let optimazer = AdaGrad::init(model.parameters(), 0.001);
}
```

- RMSProp
```rust
fn main() {
    let mut model = Module::init();
    let optimazer = RMSprop::init(model.parameters(), 0.001);
}
```

- Adam
```rust
fn main() {
    let mut model = Module::init();
    let optimazer = Adam::init(model.parameters(), 0.001);
}
```