use std::{ sync::{ Arc, Mutex, RwLock }, thread, time::SystemTime };

use rayon::iter::{ IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator };
use rotta_rs::{
    matmul,
    r,
    Adam,
    BatchNorm,
    DataHandler,
    DataHandlerMultiThreadTrait,
    Dataset,
    Linear,
    Module,
    Tensor,
    MSE,
};

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

#[derive(Clone)]
struct MyModel {
    linear_1: Linear,
    linear_2: Linear,
    linear_3: Linear,

    //
    loss: MSE,
}

impl MyModel {
    pub fn init(hidden: usize, model: &mut Module) -> Self {
        MyModel {
            linear_1: model.liniar_init(1, hidden),
            linear_2: model.liniar_init(hidden, hidden),
            linear_3: model.liniar_init(hidden, 1),

            loss: MSE::init(),
        }
    }
}

impl DataHandlerMultiThreadTrait for MyModel {
    type Input = Tensor;
    type Output = Tensor;
    fn forward(&self, data: &Self::Input) -> Self::Output {
        let x = self.linear_1.forward(data);
        let x = x.t().t();
        let x = self.linear_2.forward(&x);
        let x = x.t().t();
        let x = self.linear_3.forward(&x);
        let x = x.t().t();

        x
    }
}

fn main() {
    let raw = (0..512).map(|i| i as f64).collect::<Vec<f64>>();
    let data = Tensor::from_vector(vec![512], raw).reshape(vec![-1, 1]);

    let input = &data * 1.0;
    let label = &data * 2.0;

    ////////////////////////////////////////////////////////////
    let dataset = MyDataset {
        features: vec![input.slice(&[r(..256)]), input.slice(&[r(256..)])],
        label: vec![label.slice(&[r(..256)]), label.slice(&[r(256..)])],
    };

    let data_handler = DataHandler::init(dataset);

    let mut model = Module::init();
    let mut optimazer = Adam::init(model.parameters(), 0.001);
    let mut my_model = MyModel::init(64, &mut model);

    let tick = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();
    for epoch in 0..100 {
        optimazer.zero_grad();
        let (loss, model) = data_handler.par_by_sample(my_model, |(input, label), model| {
            input.set_requires_grad(false);
            label.set_requires_grad(false);
            let x = model.forward(input);
            let loss = model.loss.forward(&x, label);

            loss.backward();
            loss
        });
        println!("epoch:{epoch} | loss => {}", loss);

        optimazer.optim();

        my_model = model;
    }
    let tock = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();

    println!("{}ms", tock - tick)

    ///////////////////
}
