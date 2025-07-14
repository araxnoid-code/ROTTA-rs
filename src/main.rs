use std::time::SystemTime;

use rotta_rs::{ arrayy::Arrayy, *, arrayy::r };
use wide::f64x4;

fn main() {
    let features = Tensor::arange(0, 20, 1).reshape(vec![1, -1]);
    let label = &Tensor::arange(0, 20, 1).reshape(vec![1, -1]) * 2.0;

    let training_features = features
        .slice(vec![r(..), r(..((features.len() as f64) * 0.8) as i32)])
        .reshape(vec![-1, 1]);

    let training_label = label
        .slice(vec![r(..), r(..((features.len() as f64) * 0.8) as i32)])
        .reshape(vec![-1, 1]);

    let testing_features = features
        .slice(vec![r(..), r(((features.len() as f64) * 0.8) as i32..)])
        .reshape(vec![-1, 1]);

    let testing_label = label
        .slice(vec![r(..), r(((features.len() as f64) * 0.8) as i32..)])
        .reshape(vec![-1, 1]);

    // initialisasi model deep learning
    let mut model = Module::init();
    let linear_1 = model.liniar_init(1, 8);
    let linear_2 = model.liniar_init(8, 1);

    // initialisasi optimazer
    let optimazer = Sgd::init(model.parameters(), 0.0001);

    // initialisasi loss function
    let loss_fn = MSE::init();

    for epoch in 0..500 {
        // println!("epoch : {} ========================================", epoch);
        // training
        model.train();
        let x = linear_1.forward(&training_features);
        let x = linear_2.forward(&x);
        let loss = loss_fn.forward(&x, &training_label);
        // println!("training loss => {}", loss);

        optimazer.zero_grad();

        let backward = loss.backward();

        optimazer.optim(backward);

        // testing
        // model.eval();
        // let x = linear_1.forward(&testing_features);
        // let x = linear_2.forward(&x);
        // let loss = loss_fn.forward(&x, &testing_label);
        // println!("testing loss => {}", loss);
    }

    model.train();
    let x = linear_1.forward(&testing_features);
    let x = linear_2.forward(&x);
    println!("prediction:\n{}", x);
    println!("label:\n{}", testing_label)
}
