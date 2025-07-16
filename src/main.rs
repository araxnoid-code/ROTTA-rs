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

use std::time::SystemTime;

use rotta_rs::{ arrayy::r, * };

fn main() {
    let input = Tensor::arange(0, 250, 1).reshape(vec![250, 1]);
    let label = (&Tensor::arange(0, 250, 1) * 10.0).reshape(vec![250, 1]);

    let mut model = Module::init();
    model.update_initialization(rotta_rs::WeightInitialization::He);

    let linear_a = model.liniar_init(1, 512);
    // let mut droput = model.dropout_init(0.3);
    let linear_b = model.liniar_init(512, 512);

    let linear_c = model.liniar_init(512, 1);

    let loss_fn = MAE::init();
    let mut optimazer = Adam::init(model.parameters(), 0.00001);

    let tick = std::time::SystemTime
        ::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis();

    for epoch in 0..500 {
        let x = linear_a.forward(&input);
        let x = relu(&x);

        let x = linear_b.forward(&x);
        let x = relu(&x);
        // let x = droput.forward(&x);

        let x = linear_c.forward(&x);

        let loss = loss_fn.forward(&x, &label);
        println!("epoch:{epoch} | loss => {}", loss);

        optimazer.zero_grad();

        let backward = loss.backward();

        optimazer.optim(backward);
    }

    let tock = std::time::SystemTime
        ::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    println!("{}ms", tock - tick)
}
