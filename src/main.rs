use crate::rotta_rs::{
    add,
    dot,
    matmul,
    matmul_nd,
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
    let optimazer = Sgd::init(model.parameters(), 0.0001);
    let loss_fn = SSResidual::init();

    let linear = model.liniar_init(1, 16);
    let linear_2 = model.liniar_init(16, 1);

    let input = Tensor::from_vector(vec![3, 1], vec![1.0, 2.0, 3.0]);
    let actual = Tensor::from_vector(vec![3, 1], vec![2.0, 4.0, 6.0]);

    for epoch in 0..100000 {
        let x = linear.forward(&input);
        let pred = linear_2.forward(&x);

        let loss = loss_fn.forward(&pred, &actual);
        println!("epochs:{} | loss => {}\npred:\n{}", epoch, loss, pred);

        optimazer.zero_grad();

        loss.backward();

        optimazer.optim();
    }
}
