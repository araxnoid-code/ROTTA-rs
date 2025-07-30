use std::{ sync::{ Arc, Mutex, RwLock }, thread, time::SystemTime };

use rayon::iter::{ IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator };
use rotta_rs::{ matmul, r, Adam, BatchNorm, DataHandler, Dataset, Linear, Module, Tensor, MSE };

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
    // let data = Tensor::arange(0..512)
    //     .collect()
    //     .reshape(vec![-1, 1]);
    // data.set_requires_grad(false);

    let raw = (0..512).map(|i| i as f64).collect::<Vec<f64>>();
    let data = Tensor::from_vector(vec![512], raw).reshape(vec![-1, 1]);

    let input = &data * 1.0;
    let label = &data * 2.0;

    //
    let input_list = vec![
        // input.slice(&[r(..128)]),
        // input.slice(&[r(128..256)]),
        // input.slice(&[r(256..384)]),
        // input.slice(&[r(384..)])

        // input.slice(&[r(..256)]),
        // input.slice(&[r(256..)])

        input.slice(&[r(..170)]),
        input.slice(&[r(170..340)]),
        input.slice(&[r(340..)])
    ];

    let output_list = vec![
        // label.slice(&[r(..128)]),
        // label.slice(&[r(128..256)]),
        // label.slice(&[r(256..384)]),
        // label.slice(&[r(384..)])

        //
        // label.slice(&[r(..256)]),
        // label.slice(&[r(256..)])

        label.slice(&[r(..170)]),
        label.slice(&[r(170..340)]),
        label.slice(&[r(340..)])
    ];
    //

    // let dataset = MyDataset {
    //     features: vec![input.slice(&[r(..256)]), input.slice(&[r(256..)])],
    //     label: vec![label.slice(&[r(..256)]), label.slice(&[r(256..)])],
    // };

    let dataset = MyDataset {
        features: vec![
            input.slice(&[r(..170)]),
            input.slice(&[r(170..340)]),
            input.slice(&[r(340..)])
        ],
        label: vec![
            label.slice(&[r(..170)]),
            label.slice(&[r(170..340)]),
            label.slice(&[r(340..)])
        ],
    };

    let data_handler = DataHandler::init(dataset);

    let hidden = 64;

    let mut model = Module::init();
    let linear_1 = model.liniar_init(1, 2048);
    let linear_2 = model.liniar_init(2048, 2048);
    let linear_3 = model.liniar_init(2048, 1);

    let loss_fn = MSE::init();
    let mut optimazer = Adam::init(model.parameters(), 0.001);

    // // arc

    let linear_1 = Arc::new(linear_1);
    let linear_2 = Arc::new(linear_2);
    let linear_3 = Arc::new(linear_3);
    let loss_fn = Arc::new(loss_fn);

    let tick = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();
    // for epoch in 0..1 {
    // optimazer.zero_grad();

    // let linear_1 = Arc::clone(&linear_1);
    // let linear_2 = Arc::clone(&linear_2);
    // let linear_3 = Arc::clone(&linear_3);
    // let loss_fn = Arc::clone(&loss_fn);

    // let loss = data_handler.par_by_sample(move |(input, label)| {
    //     input.set_requires_grad(false);
    //     label.set_requires_grad(false);

    //     let x = linear_1.forward(input);
    //     let x = linear_2.forward(&x);
    //     let x = linear_3.forward(&x);
    //     let loss = loss_fn.forward(&x, label);

    //     loss.backward();

    //     // loss
    // });
    // println!("epoch:{0} | loss => {}", loss);

    // // optimazer.optim();
    // // }

    // let tock = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();

    // println!("{}ms", tock - tick);

    // let tick = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();

    // input.set_requires_grad(false);
    // label.set_requires_grad(false);

    // for epoch in 0..1 {
    //     let x = linear_1.forward(&input);
    //     let x = linear_2.forward(&x);
    //     let x = linear_3.forward(&x);

    //     let loss = loss_fn.forward(&x, &label);
    //     println!("epoch:{epoch} | loss => {}", loss);

    //     // optimazer.zero_grad();

    //     loss.backward();

    //     // optimazer.optim();
    // }

    // let tock = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();

    // println!("{}ms", tock - tick);

    // 0000000000000000000000000000000000000000

    let tick = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();

    // ......................
    // let x = linear_1.forward(&input);
    // let x = linear_2.forward(&x);
    // let x = linear_3.forward(&x);

    // let loss = loss_fn.forward(&x, &label);
    // loss.backward();
    // ......................
    input_list
        .par_iter()
        .zip(output_list)
        .for_each(|(input, label)| {
            let x = linear_1.forward(&input);
            let x = linear_2.forward(&x);
            let x = linear_3.forward(&x);

            let loss = loss_fn.forward(&x, &label);
            loss.backward();
        });
    // ......................

    let tock = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();

    println!("{}ms", tock - tick);
}
