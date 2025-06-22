use rand_distr::num_traits::float::FloatCore;

use crate::rotta_rs::{ sum, sum_arr, Arrayy, Module, SSResidual, Sgd, Tensor, MAE, MSE };

mod rotta_rs;

fn main() {
    let mut model = Module::init();
    let optimazaer = Sgd::init(model.parameters(), 0.0001);
    let loss_fn = MSE::init();

    let input = Tensor::new([[1.0], [2.0], [3.0], [4.0], [5.0]]);
    let label = Tensor::new([[10.0], [12.0], [13.0], [14.0], [15.0]]);

    let linear_1 = model.liniar_init(1, 16);
    let linear_2 = model.liniar_init(16, 1);

    for epoch in 0..2500 {
        let x = linear_1.forward(&input);
        let logits = linear_2.forward(&x);

        let loss = loss_fn.forward(&logits, &label);
        println!("epoch:{epoch} | loss => {}", loss);

        optimazaer.zero_grad();

        loss.backward();

        optimazaer.optim();
    }
}
