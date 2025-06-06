use ndarray::array;

use crate::rotta_rs::{ add, matmul, Module, SSResidual, Sgd, Tensor };

mod rotta_rs;

fn main() {
    let mut model = Module::init();
    let optimazer = Sgd::init(model.parameters(), 0.000001);
    let loss_fn = SSResidual::init();

    let linear = model.liniar_init(1, 1);

    let input = Tensor::new(array![[1.0], [2.0], [3.0]]);
    let actual = Tensor::new(array![[1.0], [2.0], [3.0]] * 10.0);

    for epoch in 0..100000000 {
        let pred = linear.forward(&input);

        let loss = loss_fn.forward(&pred, &actual);
        std::process::Command::new("clear").status().unwrap().success();
        println!("=============");
        println!("prediction:\n{}", pred);
        println!("actual:\n{}", actual);
        println!("epoch:{epoch} | loss:{}", loss);
        println!("=============");

        optimazer.zero_grad();

        loss.backward();

        optimazer.optim();
    }
}
