<div align="center">

<img width="300px" src="https://github.com/araxnoid-code/ROTTA-rs/blob/main/assets/rotta_logo.png?raw=true">
<!-- ![ROTTA Logo](https://github.com/araxnoid-code/ROTTA-rs/blob/main/assets/rotta_logo.png?raw=true) -->

# ROTTA-rs  
**A Deep Learning Library In Rust 🦀**

[<img alt="Static Badge" src="https://img.shields.io/badge/discord-ROTTA_rs-%235a69fc">](https://discord.gg/cgB7jst7mS)
[![Current Crates.io Version](https://img.shields.io/crates/v/rotta_rs.svg)](https://crates.io/crates/rotta_rs)
[![license](https://shields.io/badge/license-Apache--2.0-blue)](https://github.com/araxnoid-code/ROTTA-rs/blob/main/LICENSE)




*🛠️ still in development stage 🛠️*
</div>

---

## ⚙️ Create Your AI Model On Rust 🦀
`ROTTA-rs` was developed as an open-source deep learning library with the primary goal of providing an `easy-to-use`, `lightweight`, and `flexible` tool

```rust
fn main() {
    let tensor = Tensor::new([[0.1, 0.2, 0.3]]);
    println!("{}", tensor);
}
```

## 📦 Version `0.0.6`
### see more details about what's new in: [0.0.6](https://github.com/araxnoid-code/ROTTA-rs/releases/tag/0.0.6)


🧑‍💻 see other versions: </br>
📌 Check all releases: [Tags](https://github.com/araxnoid-code/ROTTA-rs/tags)  
📜 Full changelog: [version.md](https://github.com/araxnoid-code/ROTTA-rs/blob/main/version.md)



## ⚙️ Installation

[ROTTA-rs](https://crates.io/crates/rotta_rs) can be installed directly through [crates.io](https://crates.io).
To use it:

Run the following Cargo command in your project directory 📁:
```toml
[dependencies]
rotta_rs = "0.0.6"
```
or

Run the following Cargo command in your project directory 💻️:
```sh
cargo add rotta_rs
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

## 🏫 Tutorial (currently only available in Indonesian)
Those who want to learn how to create AI can access: [📔 learn.md](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/learn.md)


## 📚 GUIDE
📘 Start Exploration: 
[🧭 GUIDE.md](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/guide.md)

## 🤖 Experimental Model
ROTTA-rs also has several AI models intended for testing and not for production.

See more details about: [🤖 Experimental Models](https://github.com/araxnoid-code/ROTTA-rs/blob/main/experimental_model)


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
- [matrixmultiply](https://crates.io/crates/matrixmultiply)
- [rand_chacha](https://crates.io/crates/rand_chacha)
