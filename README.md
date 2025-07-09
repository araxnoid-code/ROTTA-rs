<div align="center">

![ROTTA Logo](https://github.com/araxnoid-code/ROTTA-rs/blob/main/assets/rotta-rs_logo_for_github.png?raw=true)

# ROTTA-rs  
**A Deep Learning Library In Rust 🦀**

*🛠️ still in development stage 🛠️*
</div>

---

## 📦 Version: `0.0.5`

### ✨ New Features
- `arange` method for create tensor
```rust
    let tensor = Tensor::arange(0, 10, 2);
    println!("{}", tensor) // [0.0, 2.0, 4.0, 6.0, 8.0]
```

- `concat method for combining tensors in vectors`
```rust
fn main() {
    let tensor_a = Tensor::new([[1.0, 2.0, 3.0, 4.0, 5.0]]);
    let tensor_b = Tensor::new([[6.0, 7.0, 8.0, 9.0, 10.0]]);
    let vector = vec![&tensor_a, &tensor_b];

    let tensor = concat(vector, 0);
    println!("{}", tensor);     // [
                                //  [1.0, 2.0, 3.0, 4.0, 5.0]
                                //  [6.0, 7.0, 8.0, 9.0, 10.0]
                                // ]    
}
```

- `new method for slicing`
```rust
fn main() {
    let tensor_a = Tensor::arange(0, 12, 1).reshape(vec![-1, 3]);
    println!("{}", tensor_a);
    // [
    //  [0.0, 1.0, 2.0]
    //  [3.0, 4.0, 5.0]
    //  [6.0, 7.0, 8.0]
    //  [9.0, 10.0, 11.0]
    // ]

    // before 0.0.5
    let slicing = tensor_a.slice(vec![ArrSlice(Some(0), Some(2)), ArrSlice(Some(1), None)]);
    println!("{}", slicing);
    // [
    //  [1.0, 2.0]
    //  [4.0, 5.0]
    // ]

    // 0.0.5
    let slicing = tensor_a.slice(vec![r(0..2), r(1..)]);
    println!("{}", slicing)
    // [
    //  [1.0, 2.0]
    //  [4.0, 5.0]
    // ]
}
```

- `new method for sum axis and sum axis keep dim`
```rust
fn main() {
    let tensor_a = Tensor::arange(0, 12, 1).reshape(vec![-1, 3]);
    println!("{}", tensor_a);
    // [
    //  [0.0, 1.0, 2.0]
    //  [3.0, 4.0, 5.0]
    //  [6.0, 7.0, 8.0]
    //  [9.0, 10.0, 11.0]
    // ]

    // before 0.0.5
    // let sum = tensor_a.sum_axis(0); // in version 0.0.5, it can no longer be done

    // 0.0.5
    let slicing = tensor_a.sum_axis(&[0]);
    println!("{}", slicing);
    // [18.0, 22.0, 26.0]

    let slicing = tensor_a.sum_axis_keep_dim(&[0, 1]);
    println!("{}", slicing);
    // [
    //  [66.0]
    // ]
}
```

- `mean` & `mean axis` & `mean axis keep dim`

see more details in guide.md [tensor section](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/section/1_tensor.md)

- `RMSProp`

see more details in guide.md [Optimazer section](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/section/5_Optimazer.md)

- `Adam`

see more details in guide.md [Optimazer section](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/section/5_Optimazer.md)

- `Layer Norm`

see more details in guide.md [Module section](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/section/4_Module.md)

- `Batch Norm`

see more details in guide.md [Module section](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/section/4_Module.md)

- `Dataset` & `DataHandler`

see more details in guide.md [Dataset and DataHandler section](https://github.com/araxnoid-code/ROTTA-rs/blob/main/book/section/7_Dataset_and_DataHandler.md)


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
