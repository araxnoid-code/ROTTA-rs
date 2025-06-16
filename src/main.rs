use crate::rotta_rs::{ add, Arrayy, Tensor };

mod rotta_rs;

fn main() {
    let tensor_a = Tensor::from_vector(vec![1, 3], vec![1.0, 2.0, 3.0]);
    let tensor_b = Tensor::from_vector(vec![3, 1], vec![1.0, 2.0, 3.0]);

    let adding = add(&tensor_a, &tensor_b);

    adding.backward();

    println!("{}", tensor_b.grad());

    println!("{}", adding.grad())
}
