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
    Arrayy,
    Module,
    SSResidual,
    Sgd,
    Tensor,
};

mod rotta_rs;

fn main() {
    // let data = 5012;
    // // println!("{}", data);
    // let x = Arrayy::ones(vec![data, data]);
    // let z = Arrayy::ones(vec![data]);

    // let mut avg = 0;
    // for _ in 0..25 {
    //     let tik = std::time::SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();

    //     // add_arr_slice((&x.value, &x.shape), (&z.value, &z.shape));

    //     let tok = std::time::SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();

    //     println!("{}ms", tok - tik);
    //     avg += tok - tik;
    // }
    // println!("avg multi {}ms", avg / 10)

    let mut model = Module::init();
    let optimazer = Sgd::init(model.parameters(), 0.00001);
    let loss_fn = SSResidual::init();

    let linear = model.liniar_init(1, 1024);
    let linear_2 = model.liniar_init(1024, 1);

    let input = Tensor::new([[1.0], [2.0]]);
    let actual = Tensor::new([[1.0], [4.0]]);

    let tik = std::time::SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();

    for epoch in 0..1000 {
        let x = linear.forward(&input);
        let x = relu(&x);
        let output = linear_2.forward(&x);

        let loss = loss_fn.forward(&output, &actual);
        println!("epoch:{epoch} | loss => {loss}");

        optimazer.zero_grad();

        let backward = loss.backward();

        optimazer.optim(backward);
    }
    let tok = std::time::SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();

    println!("{}", tok - tik)
}
