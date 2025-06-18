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
    let mut model = Module::init();
    let optimazer = Sgd::init(model.parameters(), 0.000001);
    let loss_fn = CrossEntropyLoss::init();

    let linear = model.liniar_init(1, 32);
    let linear_2 = model.liniar_init(32, 2);

    let input = Tensor::from_vector(vec![5, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let actual = Tensor::from_vector(
        vec![5, 2],
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    );

    for epoch in 0..10000 {
        let x = linear.forward(&input);
        let x = relu(&x);
        let out = linear_2.forward(&x);
        let pred = softmax(&out, 1);

        let loss = loss_fn.forward(&pred, &actual);
        println!("epochs:{} | loss => {}", epoch, loss);

        optimazer.zero_grad();

        loss.backward();

        optimazer.optim();
    }
}
