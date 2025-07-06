use std::{ ops::{ Range, RangeFrom, RangeFull }, time::SystemTime };

use rotta_rs::{
    arrayy::{ broadcasting, mean_arr, r, ArrSlice, Arrayy },
    concat,
    relu,
    sigmoid,
    softmax,
    Adam,
    CrossEntropyLoss,
    DataHandler,
    Dataset,
    Module,
    SSResidual,
    SgdMomen,
    Tensor,
};

struct MyDataset {
    input: Vec<Tensor>,
    label: Vec<Tensor>,
}

impl MyDataset {
    pub fn init(input: Vec<Tensor>, label: Vec<Tensor>) -> MyDataset {
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

fn main() {
    let input = Tensor::new([
        [1.0, 2.0],
        [4.0, 6.0],
        [8.0, 9.0],
        [10.0, 11.0],
        [25.0, 2.0],
        [3.0, 45.0],
        [10.0, 10.0],
        [10.0, 9.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [9.0, 0.0],
        [7.0, 12.0],
        [2.0, 4.0],
        [3.0, 4.0],
    ]);

    let label = Tensor::new([
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    ]);

    let mut model = Module::init();
    let mut optimazer = Adam::init(model.parameters(), 0.01);
    let loss_fn = CrossEntropyLoss::init();

    //
    let linear_a = model.liniar_init(2, 64);
    let mut batch_norm = model.layer_norm_init(&[64]);

    let linear_b = model.liniar_init(64, 2);
    //

    for epoch in 0..500 {
        let x = linear_a.forward(&input);
        let x = sigmoid(&x);
        // let x = relu(&x);
        let x = batch_norm.forward(&x);
        let x = linear_b.forward(&x);
        let pred = softmax(&x, -1);

        let loss = loss_fn.forward(&pred, &label);
        println!("train epoch:{epoch} | loss => {}", loss);

        optimazer.zero_grad();

        let backward = loss.backward();

        optimazer.optim(backward);
    }
}
