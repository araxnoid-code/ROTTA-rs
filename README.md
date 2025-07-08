<div align="center">

![ROTTA Logo](https://github.com/araxnoid-code/ROTTA-rs/blob/main/assets/rotta-rs_logo_for_github.png?raw=true)

# ROTTA-rs  
**A Deep Learning Library In Rust 🦀**

*🛠️ still in development stage 🛠️*
</div>

---

## 📦 Version: `0.0.5`

### ✨ New Features
- `Layer Norm`
- `Batch Norm`
- `concat method for tensor`
- `RMSProp`
- `Adam`
- `new method for slicing`
- `mean`
- `mean axis` & `mean axis keep dim`
- `new method for sum and sum axis`
- `Dataset` & `DataHandler`


### 🚀 Optimizations
- implemented SIMD for matmul

### 🛠️ Bug Fixes
- Broacasting error while skalar operation

📌 Check all releases: [Tags](https://github.com/araxnoid-code/ROTTA-rs/tags)  
📜 Full changelog: [version.md](https://github.com/araxnoid-code/ROTTA-rs/blob/main/version.md)

---

## ⚙️ Installation

[ROTTA-rs](https://crates.io/crates/rotta_rs) can be installed directly through [crates.io](https://crates.io).
To use it:

```toml
[dependencies]
rotta_rs = "0.0.4"
```

## 🧠 Simple AI Model
```rust
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


## 👍️ Support the Developer
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
