use crate::rotta_rs::{ add, dot, Arrayy, Tensor };

mod rotta_rs;

fn main() {
    let tensor_a = Tensor::from_vector(vec![3], vec![1.0, 2.0, 3.0]);
    let tensor_b = Tensor::from_vector(vec![3], vec![1.0, 2.0, 3.0]);

    let doting = dot(&tensor_a, &tensor_b);

    doting.backward();
    println!("{}", tensor_a.grad());
}
