use crate::rotta_rs::{
    add,
    divided,
    dot,
    matmul,
    powi,
    relu,
    softmax,
    softplus,
    Arrayy,
    CrossEntropyLoss,
    Module,
    SSResidual,
    Sgd,
    Tensor,
    WeightInitialization,
};

mod rotta_rs;

fn main() {
    let mut model = Module::init();
    model.update_initialization(WeightInitialization::He);
    let optimazer = Sgd::init(model.parameters(), 0.00001);
    let loss_fn = CrossEntropyLoss::init();

    let linear = model.liniar_init(1, 64);
    let linear_2 = model.liniar_init(64, 2);

    let input = Tensor::from_vector(vec![2, 1], vec![1.0, 2.0]);
    let actual = Tensor::from_vector(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);

    for epoch in 0..10000 {
        let x = linear.forward(&input);
        let x = softplus(&x);
        let out = linear_2.forward(&x);
        let pred = softmax(&out, 1);

        let loss = loss_fn.forward(&pred, &actual);
        println!("epochs:{} | loss => {}", epoch, loss);

        optimazer.zero_grad();

        loss.backward();

        optimazer.optim();
    }
}
