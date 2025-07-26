<div align="center">

![ROTTA Logo](https://github.com/araxnoid-code/ROTTA-rs/blob/main/assets/rotta-rs_logo_for_github.png?raw=true)

# ROTTA-rs  
**A Deep Learning Library In Rust 🦀**

[<img alt="Static Badge" src="https://img.shields.io/badge/discord-ROTTA_rs-%235a69fc">](https://discord.gg/cgB7jst7mS)
[![Current Crates.io Version](https://img.shields.io/crates/v/rotta_rs.svg)](https://crates.io/crates/rotta_rs)
[![license](https://shields.io/badge/license-Apache--2.0-blue)](https://github.com/araxnoid-code/ROTTA-rs/blob/main/LICENSE)




*🛠️ still in development stage 🛠️*
</div>

---

## 📦 Version: `0.0.6`

### ✨ New Features
- `lstm function`
```rust
fn main() {
    let mut model = Module::init();
    let lstm = model.lstm_init(6);

    let tensor = Tensor::arange(0..6)
        .to_shape(vec![1, 6])
        .collect();
    println!("{}", tensor);
    // [
    //  [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    // ]

    let cell_hidden = lstm.forward(&tensor, None); //(&Tensor, Option<LSTMCellHidden>)
    println!("cell:\n{}", cell_hidden.cell);
    // cell:
    // [
    //  [-0.35452980609595336, -0.5292895572406782, 0.7244741335616286, -0.4735260362646916, -0.9515427099660342, -0.2374937433097324]
    // ]

    println!("hidden:\n{}", cell_hidden.hidden)
    // hidden:
    // [
    //  [-0.23028978337822023, -0.0435225356725021, 0.36147911006385536, -0.09738843166346652, -0.705630337819917, -0.051854840060327305]
    // ]
}
```

```rust
fn main() {
    LSTMCellHidden {
        cell: Tensor::new([[0.0]]),
        hidden: Tensor::new([[0.0]]),
    };
}
```

- `gru function`
```rust
fn main() {
    let mut model = Module::init();
    let lstm = model.gru_init(6);

    let tensor = Tensor::arange(0..6)
        .to_shape(vec![1, 6])
        .collect();
    println!("{}", tensor);
    // [
    //  [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    // ]


    let hidden = lstm.forward(&tensor, None); // (&Tensor, Option<Tensor>)
    println!("hidden:\n{}", hidden);
    // hidden:
    // [
    //  [0.2838769389632212, 0.30500443468104915, 0.03277033032455531, -0.4399917698902819, -0.9237180542375861, 0.3168019973639146]
    // ]
}
```

- `argmax method`
```rust
fn main() {
    let tensor = Tensor::rand(vec![3, 4]);
    println!("{}", tensor);
    // [
    //  [0.490653500989335, 0.9084708072918432, 0.5516854111601106, 0.38535295859939467]
    //  [0.5510768003422782, 0.04506790076914613, 0.36501508148354644, 0.48345186596013223]
    //  [0.05192318702409915, 0.03138051383420948, 0.5086337769325273, 0.021017655640771404]
    // ]

    let max = tensor.argmax(0);
    println!("{}", max);
    // [1.0, 0.0, 0.0, 1.0]

    let max = tensor.argmax(1);
    println!("{}", max)
    // [1.0, 0.0, 2.0]
}
```

- `argmin method`
```rust
fn main() {
    let tensor = Tensor::rand(vec![3, 4]);
    println!("{}", tensor);
    // [
    //  [0.8611219295171262, 0.9410276855253755, 0.9133598099213944, 0.22062707185602048]
    //  [0.47193516661684776, 0.7818072906711374, 0.8048492003479746, 0.9925399063075784]
    //  [0.3559389967244023, 0.3472829046036767, 0.7791381493184755, 0.8910867638091713]
    // ]

    let max = tensor.argmin(0);
    println!("{}", max);
    // [2.0, 2.0, 2.0, 0.0]

    let max = tensor.argmin(1);
    println!("{}", max)
    // [3.0, 0.0, 1.0]
}
```

- update `arange` method
```rust
fn main() {
    let tensor = Tensor::arange(0..12).collect();
    println!("{}", tensor);
    //[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]


    let tensor = Tensor::arange(0..24)
        .step(2)
        .to_shape(vec![3, 4])
        .map(|x| x * 2.0)
        .collect();
    println!("{}", tensor)
    // [
    //  [0.0, 4.0, 8.0, 12.0]
    //  [16.0, 20.0, 24.0, 28.0]
    //  [32.0, 36.0, 40.0, 44.0]
    // ]
}
```

- update `concat` method
```rust
fn main() {
    let tensor_a = Tensor::arange(0..3)
        .to_shape(vec![1, 3])
        .collect();
    let tensor_b = Tensor::arange(3..6)
        .to_shape(vec![1, 3])
        .collect();

    // before 0.0.6
    let tensors = vec![&tensor_a, &tensor_b];
    let concat = concat(tensors, 0);
    println!("{}", concat);
    // [
    //  [0.0, 1.0, 2.0]
    //  [3.0, 4.0, 5.0]
    // ]

    // after 0.0.6
    let concat = vec![&tensor_a, &tensor_b].concat_tensor(0);
    println!("{}", concat);
    // [
    //  [0.0, 1.0, 2.0]
    //  [3.0, 4.0, 5.0]
    // ]

    let concat = vec![tensor_a, tensor_b].concat_tensor(0);
    println!("{}", concat)
    // [
    //  [0.0, 1.0, 2.0]
    //  [3.0, 4.0, 5.0]
    // ]
}
```


- `flatten method`
```rust
fn main() {
    let tensor = Tensor::arange(0..12)
        .to_shape(vec![3, 4])
        .collect();
    println!("{}", tensor);
    // [
    //  [0.0, 1.0, 2.0, 3.0]
    //  [4.0, 5.0, 6.0, 7.0]
    //  [8.0, 9.0, 10.0, 11.0]
    // ]

    let flat = tensor.flatten();
    println!("{}", flat);
    // [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
}
```

- `sin, cos, tan method`
```rust
fn main() {
    let tensor = Tensor::arange(0..12)
        .to_shape(vec![3, 4])
        .collect();
    println!("{}", tensor);
    // [
    //  [0.0, 1.0, 2.0, 3.0]
    //  [4.0, 5.0, 6.0, 7.0]
    //  [8.0, 9.0, 10.0, 11.0]
    // ]


    let sin = tensor.sin();

    let cos = tensor.cos();

    let tan = tensor.tan();
}
```

- update `embedding` function
```rust
fn main() {
    let mut model = Module::init();
    let embedding = model.embedding_init(6, 4); //(vocab_num: usize, hidden: usize)
    println!("{}", embedding.parameter);
    // [
    //  [0.1818961923066713, 0.450275407672484, -0.0724835971434803, 0.12736052119734032]
    //  [-0.21140612085881738, -0.35004112970967505, -0.19195944040209034, 0.3038727671756267]
    //  [0.27124878080285697, -0.2614147356186607, 0.006866870340068942, 0.4018031720487738]
    //  [0.45699026891614114, 0.09297857032433399, 0.22879848650077528, -0.3208635907368216]
    //  [0.07539314660070717, -0.32823770909937444, -0.055947665079024045, -0.42360899591141266]
    //  [-0.34544166573751034, -0.17956923353240994, 0.1996879885620404, -0.42159451133374315]
    // ]

    // using the float data type because the usize data type that is generally used in indexing is not yet available
    let token = Tensor::new([0.0, 2.0, 5.0]);
    let embedded = embedding.forward(&token);
    println!("{}", embedded);
    // [
    //  [0.1818961923066713, 0.450275407672484, -0.0724835971434803, 0.12736052119734032]
    //  [0.27124878080285697, -0.2614147356186607, 0.006866870340068942, 0.4018031720487738]
    //  [-0.34544166573751034, -0.17956923353240994, 0.1996879885620404, -0.42159451133374315]
    // ]
}
```


### 🚀 Optimizations
- Optimizing Slicing Method

### 🔧 Change
- replace `StdRng` to `chacha8rng`

📌 Check all releases: [Tags](https://github.com/araxnoid-code/ROTTA-rs/tags)  
📜 Full changelog: [version.md](https://github.com/araxnoid-code/ROTTA-rs/blob/main/version.md)

---

## ⚙️ Installation

[ROTTA-rs](https://crates.io/crates/rotta_rs) can be installed directly through [crates.io](https://crates.io).
To use it:

```toml
[dependencies]
rotta_rs = "0.0.6"
```
or

Run the following Cargo command in your project directory:
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
Those who want to learn how to create AI can access [📔 learn.md](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/learn.md) (currently only available in Indonesian)

[📔 learn.md](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/learn.md) 


## 📚 GUIDE
📘 Start learning: 
[🧭 GUIDE.md](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/guide.md)

## 🤖 Experimental Model
ROTTA-rs also has several AI models intended for testing and not for production.

See more details about: [Experimental Models](https://github.com/araxnoid-code/ROTTA-rs/blob/main/experimental_model)


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