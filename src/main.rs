use crate::rotta_rs::{ add, dot, matmul, matmul_nd, transpose, Arrayy, Tensor };

mod rotta_rs;

fn main() {
    let x = Tensor::from_vector(vec![1], vec![2.0]);
    let w_1 = Tensor::from_vector(vec![1], vec![3.0]);
    let w_2 = Tensor::from_vector(vec![1], vec![4.0]);

    let h1 = dot(&x, &w_1);
    let h2 = dot(&h1, &w_2);
    let out = add(&h1, &h2);

    out.backward();

    println!("{}", w_1.grad());
}
