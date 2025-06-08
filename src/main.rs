use ndarray::array;

use crate::rotta_rs::{ relu, softmax, CrossEntropyLoss, Module, SSResidual, Sgd, Tensor };

mod rotta_rs;

fn main() {
    // let mut model = Module::init();
    // let optimazer = Sgd::init(model.parameters(), 0.00001);
    // let loss_fn = SSResidual::init();

    // let linear = model.liniar_init(1, 1);

    // let input = Tensor::new(array![[1.0], [2.0], [3.0]]);
    // let actual = Tensor::new(array![[1.0], [2.0], [3.0]] * 10.0);

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

    let mut model = Module::init();
    let optimazer = Sgd::init(model.parameters(), 0.001);
    let loss_fn = CrossEntropyLoss::init();

    let linear = model.liniar_init(1, 64);
    let linear_2 = model.liniar_init(64, 2);

    // let input = Tensor::new(array![[1.0]]);
    // let actual = Tensor::new(array![[1.0, 0.0]]);

    let inputs = vec![
        Tensor::new(array![[1.0]]),
        Tensor::new(array![[2.0]]),
        Tensor::new(array![[3.0]]),
        Tensor::new(array![[4.0]]),
        Tensor::new(array![[5.0]])
    ];

    let actuals = vec![
        Tensor::new(array![[1.0, 0.0]]),
        Tensor::new(array![[0.0, 1.0]]),
        Tensor::new(array![[1.0, 0.0]]),
        Tensor::new(array![[0.0, 1.0]]),
        Tensor::new(array![[1.0, 0.0]])
    ];

    for epoch in 0..10000 {
        // break;
        println!(".....................");
        let mut avg_loss = array![[0.0]];
        for i in 0..inputs.len() {
            let input = &inputs[i];
            let actual = &actuals[i];

            let x = linear.forward(&input);
            let x = relu(&x);
            let pred = linear_2.forward(&x);

            let prob = softmax(&pred);

            let loss = loss_fn.forward(&prob, &actual);
            avg_loss = avg_loss + loss.value();
            // println!("=============");
            // println!("input:\n{}", input);
            // println!("prediction:\n{}", prob);
            // println!("actual:\n{}", actual);
            // println!("epoch:{epoch} | loss:{}", loss);
            // println!("=============");

            optimazer.zero_grad();

            loss.backward();

            optimazer.optim();
        }
        println!("epoch {epoch} | loss => {:?}", avg_loss / 5.0);
        println!(".....................");
    }

    // test
    for i in 0..inputs.len() {
        let input = &inputs[i];
        let actual = &actuals[i];

        let x = linear.forward(&input);
        let x = relu(&x);
        let pred = linear_2.forward(&x);

        let prob = softmax(&pred);

        println!("=============");
        println!("input:\n{}", input);
        println!("prediction:\n{}", prob);
        println!("actual:\n{}", actual);
        println!("=============");
    }
}
