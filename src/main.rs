use std::{ sync::{ Arc, Mutex }, time::UNIX_EPOCH };

use crate::rotta_rs::{
    add_arr,
    add_arr_slice,
    broadcast_concat,
    broadcast_shape_slice,
    broadcasting,
    broadcasting_arr_slice,
    matmul_2d,
    matmul_2d_slice,
    matmul_nd,
    matmul_nd_slice,
    par_add_arr,
    relu,
    AdaGrad,
    Arrayy,
    Module,
    SSResidual,
    Sgd,
    SgdMomen,
    Tensor,
    MSE,
};

mod rotta_rs;

fn main() {
    // let mut model = Module::init();
    // let mut dropout = model.dropout_init(0.5);
    // model.eval();

    // let tensor = Tensor::from_vector(vec![2, 3, 2], vec![5.0;12]);

    // let x = dropout.forward(&tensor);

    // println!("{}", x);

    // println!("{x}");

    let mut model = Module::init();

    let mut optimazer = AdaGrad::init(model.parameters(), 0.0001);
    let loss_fn = MSE::init();

    let linear = model.liniar_init(1, 256);
    let mut drop = model.dropout_init(0.3);
    let linear_2 = model.liniar_init(256, 1);

    let input = Tensor::new([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0],
        [10.0],
    ]);
    let actual = Tensor::new([
        [10.0],
        [20.0],
        [30.0],
        [40.0],
        [50.0],
        [60.0],
        [70.0],
        [80.0],
        [90.0],
        [100.0],
    ]);

    // let tik = std::time::SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();

    model.train();
    for epoch in 0..1000 {
        let x = linear.forward(&input);
        // let x = drop.forward(&x);
        let output = linear_2.forward(&x);

        let loss = loss_fn.forward(&output, &actual);

        println!("prediction:\n{}", output);
        println!("actual:\n{}", actual);
        println!("epoch:{epoch} | loss => {loss}");

        optimazer.zero_grad();

        let backward = loss.backward();

        optimazer.optim(backward);
    }
    // let tok = std::time::SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();

    // println!("{}", tok - tik)
}
