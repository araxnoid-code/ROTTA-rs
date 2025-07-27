use rotta_rs::Tensor;

fn main() {
    // println!("ROTTA-rs version 0.0.6")

    let tensor = Tensor::new([[0.1, 0.2, 0.3]]);
    println!("{}", tensor);
}
