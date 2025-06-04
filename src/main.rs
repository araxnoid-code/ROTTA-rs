use ndarray::array;

use crate::rotta_rs::{ matmul, Tensor };

mod rotta_rs;

fn main() {
    let tensor_1 = Tensor::new(array![[2.0, 3.0, 4.0]]);
    let tensor_2 = Tensor::new(array![[1.0], [2.0], [3.0]]);

    let mul = matmul(&tensor_1, &tensor_2);
    println!("{}", mul);
}
