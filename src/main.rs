use rotta_rs::Tensor;

fn main() {
    let tensor = Tensor::arange(0..20)
        .step(2)
        .map(|x| { x * 2.0 })
        .to_shape(vec![2, 5])
        .collect();
    println!("{}", tensor);
}
