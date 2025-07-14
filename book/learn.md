# Learn

pada learn.md akan memberikan contoh pemakaian ROTTA-rs dalam bahasa indonesia.

hal yang harus diperhatikan:
- pastikan sudah menginstall RUST
- tambahkan di dalam file `Cargo.toml`
```toml
[dependencies]
rotta_rs = "0.0.5"
```

## tujuan
Kita akan membuat sebuah model AI untuk memprediksi data linear, kasus nyatanya seperti adalah model AI untuk memprediksi harga rumah berdasarkan jumlah kamar dan furnitur.
namun saat ini kita akan membuat model sederhana untuk memprediksi data linear biasa saja.

## dasar
Ayo awali dengan data sederhana
```rust
fn main() {
    let data = Tensor::arange(0, 20, 1).reshape(vec![1, -1]);
    println!("{}", data)
}
```
pada terminal akan menampilkan output:
```sh
[
 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
]
```
ini adalah tensor, bisa dibilang tensor adalah bentuk menyimpan data berupa angka dengan cara berdimensi dimensi, yang ada di atas adalah contoh dari tensor beridemensi 2.

ayo kita buat tensor satu lagi! lalu kita lakukan operasi perkalian secara element wise!
```rust
fn main() {
    let tensor = Tensor::arange(0, 20, 1).reshape(vec![1, -1]);
    let tensor = &tensor * 2.0;
    println!("{}", tensor)
}
```
output:
```sh
[
 [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0]
]
```
bisa kita lihat, pada versi 0.0.5 saat ini dan dibawahnya, perkalian menggunakan tensor harus menggunakan reference `&`. Mungkin saja di update berikutnya akan ditambahkan pengoperasian dengan cara langsung tanpa reference, tergantung mood saya ;)


## features & label
untuk lebih mudahnya, features bisa dianggap adalah input dan label bisa dianggap sebagai nilai yang dinginkan pada output dari model AI.

berdasarkan contoh yang diatas kita bisa membuat features dan label:
```rust
fn main() {
    let features = Tensor::arange(0, 20, 1).reshape(vec![1, -1]);
    let label = &Tensor::arange(0, 20, 1).reshape(vec![1, -1]) * 2.0;

    println!("features:\n{}", features);
    println!("label:\n{}", label)
}
```
output:
```sh
features:
[
 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
]

label:
[
 [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0]
]
```

data pada features dan label ini saat di masukkan ke kordinat kartesius yang mana features sebagai x dan label sebagai y kita bisa mendapatkan sebuah garis linear. karena dalam contoh ini kita akan membuat model untuk memprediksi data linear.

sebelum itu dalam deep learning ada 2 tahapan yang penting yaitu training phase dan testing phase. data pada training phase digunakan untuk melatih AI dan data pada testing phase untuk memvalidasi kinerja AI kita.

ayo kita bagi data features dan label kita menjadi data untuk training dan data untuk testing, dalam kasus ini kita akan split 80% untuk training dan 20% untuk testing
```rust
fn main() {
    let features = Tensor::arange(0, 20, 1).reshape(vec![1, -1]);
    let label = &Tensor::arange(0, 20, 1).reshape(vec![1, -1]) * 2.0;

    let training_features = features.slice(
        vec![r(..), r(..((features.len() as f64) * 0.8) as i32)]
    );
    println!("training_features:\n{}", training_features);

    let training_label = label.slice(vec![r(..), r(..((features.len() as f64) * 0.8) as i32)]);
    println!("training_label:\n{}", training_label);

    let testing_features = features.slice(vec![r(..), r(((features.len() as f64) * 0.8) as i32..)]);
    println!("testing_features:\n{}", testing_features);

    let testing_label = label.slice(vec![r(..), r(((features.len() as f64) * 0.8) as i32..)]);
    println!("testing_label:\n{}", testing_label);
}
```
output:
```sh
training_features:
[
 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
]

training_label:
[
 [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
]

testing_features:
[
 [16.0, 17.0, 18.0, 19.0]
]

testing_label:
[
 [32.0, 34.0, 36.0, 38.0]
]
```
kita akan membuat data diatas menjadi batch, agar mudah di training nantinya:
```rust
fn main() {
    let features = Tensor::arange(0, 20, 1).reshape(vec![1, -1]);
    let label = &Tensor::arange(0, 20, 1).reshape(vec![1, -1]) * 2.0;

    let training_features = features
        .slice(vec![r(..), r(..((features.len() as f64) * 0.8) as i32)])
        .reshape(vec![-1, 1]);
    println!("training_features:\n{}", training_features);

    let training_label = label
        .slice(vec![r(..), r(..((features.len() as f64) * 0.8) as i32)])
        .reshape(vec![-1, 1]);
    println!("training_label:\n{}", training_label);

    let testing_features = features
        .slice(vec![r(..), r(((features.len() as f64) * 0.8) as i32..)])
        .reshape(vec![-1, 1]);
    println!("testing_features:\n{}", testing_features);

    let testing_label = label
        .slice(vec![r(..), r(((features.len() as f64) * 0.8) as i32..)])
        .reshape(vec![-1, 1]);
    println!("testing_label:\n{}", testing_label);
}
```
output:
```sh
training_features:
[
 [0.0]
 [1.0]
 [2.0]
 [3.0]
 [4.0]
 [5.0]
 [6.0]
 [7.0]
 [8.0]
 [9.0]
 [10.0]
 [11.0]
 [12.0]
 [13.0]
 [14.0]
 [15.0]
]

training_label:
[
 [0.0]
 [2.0]
 [4.0]
 [6.0]
 [8.0]
 [10.0]
 [12.0]
 [14.0]
 [16.0]
 [18.0]
 [20.0]
 [22.0]
 [24.0]
 [26.0]
 [28.0]
 [30.0]
]

testing_features:
[
 [16.0]
 [17.0]
 [18.0]
 [19.0]
]

testing_label:
[
 [32.0]
 [34.0]
 [36.0]
 [38.0]
]
```

dari data di atas. kita akan membuat sebuah model AI yang dari data training akan dilatih sehingga dapat memprediksi data pada testing label.

## build An AI Model
ROTTA-rs disengaja oleh developer untuk dikembangkan dengan pendekatan mirip dengan pytorch(mengingat developer dulunya pengguna pytorch, dengan kata lain sudah kebiasaan hehehe)

```rust
fn main() {
    // initialisasi model deep learning
    let mut model = Module::init();
    let linear_1 = model.liniar_init(1, 8);
    let linear_2 = model.liniar_init(8, 1);

    // initialisasi optimazer
    let mut optimazer = Sgd::init(model.parameters(), 0.0001);

    // initialisasi loss function
    let loss_fn = MSE::init();
}
```
`penjelasan:`
- model deep learning perlu diperlukan untuk membangun neural network
- optimazer diperlukan untuk melatih model deep learning
- loss function diperlukan untuk menjadi acuan seberapa optimalnya model deep learning, Jika nilai loss nya besar model kita masih belum paham akan data, namun jika kecil menandakan model kita sudah paham dengan data yang diberikan

## Forward and Backward
dalam model AI Deep Learning, ada 2 tahapan dalam melatih AI, yaitu forward dan backward(bisa disebut backpropagation). singkatnya forward ialah proses AI akan memberikan output atas data yang telah di inputkan, backward adalah proses mendapatkan gradient untuk semua nilai yang dapat diupdate untuk diperbarui sehingga AI bisa lebih optimal lagi.


ayo kita gabungkan model Deep learning dengan data training dan data testing, serta kita latih AI kita menggunakan looping sebanyak 5 kali
```rust
fn main() {
    let features = Tensor::arange(0, 20, 1).reshape(vec![1, -1]);
    let label = &Tensor::arange(0, 20, 1).reshape(vec![1, -1]) * 2.0;

    let training_features = features
        .slice(vec![r(..), r(..((features.len() as f64) * 0.8) as i32)])
        .reshape(vec![-1, 1]);

    let training_label = label
        .slice(vec![r(..), r(..((features.len() as f64) * 0.8) as i32)])
        .reshape(vec![-1, 1]);

    let testing_features = features
        .slice(vec![r(..), r(((features.len() as f64) * 0.8) as i32..)])
        .reshape(vec![-1, 1]);

    let testing_label = label
        .slice(vec![r(..), r(((features.len() as f64) * 0.8) as i32..)])
        .reshape(vec![-1, 1]);

    // initialisasi model deep learning
    let mut model = Module::init();
    let linear_1 = model.liniar_init(1, 8);
    let linear_2 = model.liniar_init(8, 1);

    // initialisasi optimazer
    let optimazer = Sgd::init(model.parameters(), 0.0001);

    // initialisasi loss function
    let loss_fn = MSE::init();

    for epoch in 0..5 {
        println!("epoch : {} ========================================", epoch);
        // training
        model.train();
        let x = linear_1.forward(&training_features);
        let x = linear_2.forward(&x);
        let loss = loss_fn.forward(&x, &training_label);
        println!("training loss => {}", loss);

        optimazer.zero_grad();

        let backward = loss.backward();

        optimazer.optim(backward);

        // testing
        model.eval();
        let x = linear_1.forward(&testing_features);
        let x = linear_2.forward(&x);
        let loss = loss_fn.forward(&x, &testing_label);
        println!("testing loss => {}", loss);
    }
}
```
output:
```sh
epoch : 0 ========================================
training loss => [1729.5590425056262]

testing loss => [5499.424885043851]

epoch : 1 ========================================
training loss => [1366.846193093237]

testing loss => [4466.814181281116]

epoch : 2 ========================================
training loss => [1106.7852290839287]

testing loss => [3692.1134159826956]

epoch : 3 ========================================
training loss => [911.8460446887253]

testing loss => [3090.711529491906]

epoch : 4 ========================================
training loss => [760.6655626624074]

testing loss => [2611.2345855416906]
```
bisa dilihat loss semakin menurun, menandakan model Deep Learning kita belajar setiap iterasinya.

ayo kita looping 500 kali
```rust
fn main() {
    // ...

    for epoch in 0..500 {
        println!("epoch : {} ========================================", epoch);
        // training
        model.train();
        let x = linear_1.forward(&training_features);
        let x = linear_2.forward(&x);
        let loss = loss_fn.forward(&x, &training_label);
        println!("training loss => {}", loss);

        optimazer.zero_grad();

        let backward = loss.backward();

        optimazer.optim(backward);

        // testing
        model.eval();
        let x = linear_1.forward(&testing_features);
        let x = linear_2.forward(&x);
        let loss = loss_fn.forward(&x, &testing_label);
        println!("testing loss => {}", loss);
    }
}
```
output:
```sh
# ...
epoch : 495 ========================================
training loss => [0.27351168789466546]

testing loss => [0.5220625589097934]

epoch : 496 ========================================
training loss => [0.2727442261368042]

testing loss => [0.52059631352512]

epoch : 497 ========================================
training loss => [0.2719789212630659]

testing loss => [0.5191341937330147]

epoch : 498 ========================================
training loss => [0.2712157671881738]

testing loss => [0.5176761878785985]

epoch : 499 ========================================
training loss => [0.27045475784417017]

testing loss => [0.5162222843402036]
```
bisa dilihat loss pada training dan testing sudah sangatlah kecil, menandakan model AI kita dapat mempelajari data yang diberikan dengn baik

ayo kita lihat bagaimana output dari model kita pada testing data:
```rust
fn main(){
    // ...

    model.train();
    let x = linear_1.forward(&testing_features);
    let x = linear_2.forward(&x);
    println!("prediction:\n{}", x);
    println!("label:\n{}", testing_label)
}
```
ouput:
```sh
prediction:
[
 [31.43565071784062]
 [33.33840838486221]
 [35.2411660518838]
 [37.1439237189054]
]

label:
[
 [32.0]
 [34.0]
 [36.0]
 [38.0]
]
```
bisa kita lihat, prediksi model AI kita sudah mendekati data test, berarti model kita berhasil mempelajari data latihan dengan baik.

sekian, terimakasih telah membaca. maaf jikalau ada kesalahan.
