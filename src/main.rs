use crate::rotta_rs::{ add, dot, matmul, matmul_nd, transpose, Arrayy, SSResidual, Tensor };

mod rotta_rs;

fn main() {
    let pred = Tensor::from_vector(vec![1, 2], vec![1.0, 2.0]);
    let actual = Tensor::from_vector(vec![1, 2], vec![2.0, 4.0]);

    let loss_fn = SSResidual::init();

    let loss = loss_fn.forward(&pred, &actual);
    loss.backward();
}
