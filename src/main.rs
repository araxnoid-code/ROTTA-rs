use ndarray::array;

use crate::rotta_rs::{ relu, softmax, Module, SSResidual, Sgd, Tensor };

mod rotta_rs;

fn main() {
    // let mut model = Module::init();
    // let optimazer = Sgd::init(model.parameters(), 0.00001);
    // let loss_fn = SSResidual::init();

    // let linear = model.liniar_init(1, 1);

    // let input = Tensor::new(array![[1.0], [2.0], [3.0]]);
    // let actual = Tensor::new(array![[1.0], [2.0], [3.0]] * 10.0);

    // println!("{}", relu.grad())

    // for epoch in 0..100000000 {
    //     let pred = linear.forward(&input);

    //     let loss = loss_fn.forward(&pred, &actual);
    //     println!("=============");
    //     println!("prediction:\n{}", pred);
    //     println!("actual:\n{}", actual);
    //     println!("epoch:{epoch} | loss:{}", loss);
    //     println!("=============");
    //     std::process::Command::new("clear").status().unwrap().success();

    //     optimazer.zero_grad();

    //     loss.backward();

    //     optimazer.optim();
    // }

    let x = Tensor::new(array![[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]);
    softmax(&x);
}
