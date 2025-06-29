<div align="center">

![ROTTA Logo](https://github.com/araxnoid-code/ROTTA-rs/blob/main/assets/rotta-rs_logo_for_github.png?raw=true)

# ROTTA-rs  
**An AI Framework In Rust 🦀**

*🛠️ still in development stage 🛠️*
</div>

---

## 📦 Version: `0.0.4`

### ✨ New Features
- `Dropout`
- `SGD + Momentum`
- `AdaGrad`
- `powf`
- New method for creating `tensors`


### 🚀 Optimizations
- Optimized Basic Operations `add`, `sub`, `mul`, `div`, `matmul`

### 🛠️ Bug Fixes
- Fixed bug on `Sum Square Residual`
- Fixed a bug where tensors accumulated their gradients

📌 Check all releases: [Tags](https://github.com/araxnoid-code/ROTTA-rs/tags)  
📜 Full changelog: [version.md](https://github.com/araxnoid-code/ROTTA-rs/blob/main/version.md)

---

## ⚙️ Installation

Currently, **ROTTA-rs** is not yet available on [crates.io](https://crates.io).  
To use it:

```toml
[dependencies]
rotta_rs = {git = "https://github.com/araxnoid-code/ROTTA-rs"}
```

## 🧠 Simple AI Model
```rust
pub mod rotta_rs;
use rotta_rs::*;

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

        let backward = loss.backward();

        optimazer.optim(backward);
    }
}
```


## 📚 GUIDE
📘 Start learning: 
[🧭 GUIDE.md](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/guide.md)


## ❤️ Support the Developer
If you find this project useful, you can support further development via:

[🔗 saweria](https://saweria.co/araxnoid)

## 🌐 Connect with Me
- youtube

araxnoid

[click here to go directly to youtube](https://www.youtube.com/@araxnoid-v5o)

- tiktok

araxnoid

[click here to go directly to tiktok](https://www.tiktok.com/@araxnoid_code)

## contact
- Gmail

araxnoid0@gmail.com

## 📥 Dependencies
- [rand](https://crates.io/crates/rand)
- [rand_distr](https://crates.io/crates/rand_distr)
- [uuid](https://crates.io/crates/uuid)
- [rayon](https://crates.io/crates/rayon)