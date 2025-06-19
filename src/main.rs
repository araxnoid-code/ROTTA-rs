use std::process::Termination;

use crate::rotta_rs::{ divided, exp, softmax, sum_axis_keep_dim, CrossEntropyLoss };
#[allow(unused_imports)]
use crate::rotta_rs::{
    add,
    dot,
    matmul,
    matmul_nd,
    relu,
    sum_axis,
    transpose,
    Arrayy,
    Module,
    RecFlatten,
    SSResidual,
    Sgd,
    Tensor,
};

mod rotta_rs;

fn main() {
    let tensor = Tensor::new([[1.0], [2.0], [3.0]]);
    println!("{:?}", tensor.shape())
}
