struct MyDataset {
    input: Vec<Tensor>,
    label: Vec<Tensor>,
}

impl MyDataset {
    pub fn init(input: Vec<Tensor>, label: Vec<Tensor>) -> MyDataset {
        for input in &input {
            input.set_requires_grad(false);
        }

        for label in &label {
            label.set_requires_grad(false);
        }

        MyDataset {
            input,
            label,
        }
    }
}

impl Dataset for MyDataset {
    fn get(&self, idx: usize) -> (Tensor, Tensor) {
        (self.input[idx].clone(), self.label[idx].clone())
    }

    fn len(&self) -> usize {
        self.input.len()
    }
}

use rotta_rs::{ arrayy::r, * };

fn main() {
    let input = Tensor::arange(0, 250, 1).reshape(vec![250, 1]);
    let label = (&Tensor::arange(0, 250, 1) * 10.0).reshape(vec![250, 1]);

    let dataset = MyDataset::init(
        vec![input.slice(&[r(..125)]), input.slice(&[r(125..)])],
        vec![label.slice(&[r(..125)]), label.slice(&[r(125..)])]
    );
    let mut datahandler = DataHandler::init(dataset);
    datahandler.shuffle();
    datahandler.batch(64);

    let mut model = Module::init();
    model.update_initialization(rotta_rs::WeightInitialization::He);

    let linear_a = model.liniar_init(1, 1024);
    let mut batch_norm_a = model.batch_norm_init(1024, 2);
    // let mut droput = model.dropout_init(0.3);
    let linear_b = model.liniar_init(1024, 1024);
    let mut batch_norm_b = model.batch_norm_init(1024, 2);
    let linear_c = model.liniar_init(1024, 1);

    let loss_fn = MAE::init();
    let mut optimazer = Adam::init(model.parameters(), 0.5);

    for epoch in 0..10 {
        let mut avg = 0.0;
        let mut iteration = 0.0;
        for (input, label) in &mut datahandler {
            let x = linear_a.forward(&input);
            let x = batch_norm_a.forward(&x);

            let x = linear_b.forward(&x);
            let x = batch_norm_b.forward(&x);

            let x = linear_c.forward(&x);

            let loss = loss_fn.forward(&x, &label);
            // println!("epoch:{epoch} | loss => {}", loss);
            avg += loss.value().value[0];

            optimazer.zero_grad();

            let backward = loss.backward();

            optimazer.optim(backward);
            iteration += 1.0;
        }
        println!("epoch:{epoch} | loss => {}", &avg / iteration);
    }
}
