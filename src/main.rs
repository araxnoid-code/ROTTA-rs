use crate::rotta_rs::{
    add,
    divided,
    dot,
    matmul,
    powi,
    relu,
    sigmoid,
    softmax,
    softplus,
    tanh,
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
    let pred = Tensor::new([[0.5, 0.25, 0.25]]);
    let actual = Tensor::new([1.0, 0.0, 0.0]);
    let actual_batch = Tensor::new([0.0]);

    let loss_fn = CrossEntropyLoss::init();

    let loss = loss_fn.forward(&pred, &actual);
    println!("{}", loss);

    loss_fn.test_forward(&pred, &actual_batch);

    // let mut model = Module::init();
    // model.update_initialization(WeightInitialization::He);
    // let optimazer = Sgd::init(model.parameters(), 0.0001);
    // let loss_fn = CrossEntropyLoss::init();

    // let linear = model.liniar_init(1, 32);
    // let linear_2 = model.liniar_init(32, 2);

    // let input = Tensor::from_vector(vec![2, 1], vec![1.0, 2.0]);
    // let actual = Tensor::from_vector(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);

    // for epoch in 0..10000 {
    //     let x = linear.forward(&input);
    //     let x = sigmoid(&x);
    //     let x = linear_2.forward(&x);
    //     let x = sigmoid(&x);
    //     let pred = softmax(&x, 1);

    //     let loss = loss_fn.forward(&pred, &actual);
    //     println!("epochs:{} | loss => {}", epoch, loss);

    //     optimazer.zero_grad();

    //     loss.backward();

    //     optimazer.optim();
    // }
}
