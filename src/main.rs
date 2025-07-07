use std::{ ops::{ Range, RangeFrom, RangeFull }, time::SystemTime };

use rotta_rs::{
    arrayy::{ broadcasting, matmul_2d_slice, mean_arr, r, ArrSlice, Arrayy },
    concat,
    matmul,
    relu,
    sigmoid,
    softmax,
    Adam,
    BatchNorm,
    CrossEntropyLoss,
    DataHandler,
    Dataset,
    Module,
    SSResidual,
    SgdMomen,
    Tensor,
    MAE,
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
    let input = Tensor::arange(&[250, 1]);
    let label = &Tensor::arange(&[250, 1]) * 10.0;

    let mut model = Module::init();
    model.update_initialization(rotta_rs::WeightInitialization::He);

    let linear_a = model.liniar_init(1, 1024);
    let mut batch_norm_a = model.batch_norm_init(1024, 2);
    // let mut droput = model.dropout_init(0.3);
    let linear_b = model.liniar_init(1024, 1024);
    let mut batch_norm_b = model.batch_norm_init(1024, 2);
    let linear_c = model.liniar_init(1024, 1);

    let loss_fn = MAE::init();
    let mut optimazer = Adam::init(model.parameters(), 0.001);

    println!("learning starting");
    let tick = std::time::SystemTime
        ::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    for epoch in 0..100 {
        let x = linear_a.forward(&input);
        let x = relu(&x);
        let x = batch_norm_a.forward(&x);

        let x = linear_b.forward(&x);
        let x = relu(&x);
        // let x = batch_norm_b.forward(&x);
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
