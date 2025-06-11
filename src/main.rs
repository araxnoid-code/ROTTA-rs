use ndarray::array;

use crate::rotta_rs::{ add, relu, softmax, CrossEntropyLoss, Module, SSResidual, Sgd, Tensor };

mod rotta_rs;

fn main() {
    let a = array![[1.0, 2.0, 3.0]];
    let tensor_a = Tensor::new(a);

    let b = array![[1.0, 2.0, 3.0]];
    let tensor_b = Tensor::new(b);

    let c = add(&tensor_a, &tensor_b);
    let exp = c.exp();
    exp.backward();
    println!("{}", c.grad());
}
