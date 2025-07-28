use std::{ sync::{ Arc, Mutex, RwLock }, thread, time::SystemTime };

use rotta_rs::{ r, Adam, BatchNorm, DataHandler, Dataset, Linear, Module, Tensor, MSE };

struct MyDataset {
    features: Vec<Tensor>,
    label: Vec<Tensor>,
}

impl Dataset for MyDataset {
    fn get(&self, idx: usize) -> (Tensor, Tensor) {
        (self.features[idx].clone(), self.label[idx].clone())
    }

    fn len(&self) -> usize {
        self.features.len()
    }
}

struct MyModel {
    norm: BatchNorm,
    linear_1: Linear,
    linear_2: Linear,
    model: Module,
}

impl MyModel {
    pub fn init(hidden: usize) -> Self {
        let mut model = Module::init();
        MyModel {
            norm: model.batch_norm_init(1, 2),
            linear_1: model.liniar_init(1, hidden),
            linear_2: model.liniar_init(hidden, 1),
            model,
        }
    }

    pub fn forward(&self) {
        let input = Tensor::new([0.0]);
    }
}

fn main() {
    let data = Tensor::arange(0..512)
        .collect()
        .reshape(vec![-1, 1]);
    data.set_requires_grad(false);

    let input = &data * 1.0;
    let label = &data * 2.0;

    let dataset = MyDataset {
        features: vec![
            input.slice(&[r(..128)]),
            input.slice(&[r(128..256)]),
            input.slice(&[r(256..384)]),
            input.slice(&[r(384..)])
        ],
        label: vec![
            label.slice(&[r(..128)]),
            label.slice(&[r(128..256)]),
            label.slice(&[r(256..384)]),
            label.slice(&[r(384..)])
        ],
    };
    let data_handler = DataHandler::init(dataset);

    // let hidden = 64;

    //

    //

    let mut model = Module::init();
    let linear_1 = model.liniar_init(1, 64);
    let linear_2 = model.liniar_init(64, 64);
    let linear_3 = model.liniar_init(64, 1);

    let loss_fn = MSE::init();
    let mut optimazer = Adam::init(model.parameters(), 0.01);

    // // arc

    let tick = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();

    for epoch in 0..100 {
        let linear_1 = linear_1.clone();
        let linear_2 = linear_2.clone();
        let linear_3 = linear_3.clone();
        let loss_fn = loss_fn.clone();

        let loss = data_handler.par_by_sample(move |(input, label)| {
            let x = linear_1.forward(input);
            let x = linear_2.forward(&x);
            let x = linear_3.forward(&x);

            let loss = loss_fn.forward(&x, label);
            loss
        });
        println!("epoch:{epoch} | loss => {}", loss);

        optimazer.zero_grad();

        let backward = loss.backward();

        optimazer.optim(backward);
    }

    let tock = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();

    println!("{}ms", tock - tick);

    // let tick = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();

    // for epoch in 0..100 {
    //     let x = linear_1.forward(&input);
    //     let x = linear_2.forward(&x);

    //     let loss = loss_fn.forward(&x, &label);
    //     println!("epoch:{epoch} | loss => {}", loss);

    //     optimazer.zero_grad();

    //     let backward = loss.backward();

    //     optimazer.optim(backward);
    // }

    // let tock = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();

    // println!("{}ms", tock - tick);
}
