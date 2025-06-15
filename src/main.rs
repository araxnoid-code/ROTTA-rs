use crate::rotta_rs::{ Arrayy, Tensor };

mod rotta_rs;

fn main() {
    let tensor_a = Tensor::new(Arrayy::from_vector(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));

    println!("{}", tensor_a);
}
