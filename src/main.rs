use rotta_rs::{ Adam, DataHandler, Dataset, Linear, Module, ParDataHandler, Tensor, MSE };

struct MyDataset {
    data: Vec<Tensor>,
    label: Vec<Tensor>,
}

impl MyDataset {
    pub fn init(data: Vec<Tensor>, label: Vec<Tensor>) -> MyDataset {
        MyDataset {
            data,
            label,
        }
    }
}

impl Dataset for MyDataset {
    fn get(&self, idx: usize) -> (Tensor, Tensor) {
        (self.data[idx].clone(), self.label[idx].clone())
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Clone)]
struct MyModel {
    linear_1: Linear,
    linear_2: Linear,
    loss_fn: MSE,
}

impl MyModel {
    fn init(model: &mut Module) -> MyModel {
        MyModel {
            linear_1: model.liniar_init(1, 8),
            linear_2: model.liniar_init(8, 1),
            loss_fn: MSE::init(),
        }
    }
}

impl ParDataHandler for MyModel {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, data: &Self::Input) -> Self::Output {
        let x = self.linear_1.forward(data);
        let x = self.linear_2.forward(&x);
        x
    }
}

fn main() {
    let data_a = Tensor::new([[1.0]]);
    let label_a = Tensor::new([[2.0]]);

    let data_b = Tensor::new([[3.0]]);
    let label_b = Tensor::new([[4.0]]);

    let data_list = vec![data_a, data_b];
    let label_list = vec![label_a, label_b];
    let dataset = MyDataset::init(data_list, label_list);
    let mut datahandler = DataHandler::init(dataset);

    let mut model = Module::init();
    let mut my_model = MyModel::init(&mut model);
    let mut optimazer = Adam::init(model.parameters(), 0.001);

    // for i in 0..10 {
    //     for (input, label) in &mut datahandler {
    //         let x = my_model.forward(&input);

    //         let loss = my_model.loss_fn.forward(&x, &label);
    //         println!("{}", loss);

    //         optimazer.zero_grad();

    //         loss.backward();

    //         optimazer.optim();
    //     }
    // }

    for epoch in 0..10 {
        optimazer.zero_grad();
        let (loss, model) = datahandler.par_by_sample(my_model, 2, |(input, label), model| {
            let x = model.forward(input);

            let loss = model.loss_fn.forward(&x, &label);

            loss.backward();

            loss
        });
        // println!("epoch:{epoch} => {}", loss);

        optimazer.optim();

        my_model = model;
    }
}
