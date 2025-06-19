use crate::rotta_rs::{
    add,
    divided,
    dot,
    matmul,
    relu,
    Arrayy,
    Module,
    SSResidual,
    Sgd,
    Tensor,
    WeightInitialization,
};

mod rotta_rs;

fn main() {
    let tensor_a = Tensor::new([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]);

    let tensor_b = Tensor::new([
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
    ]);

    let result = matmul(&tensor_a, &tensor_b);
}
