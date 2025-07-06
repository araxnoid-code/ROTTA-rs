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
    // text: "halo(0) saya(1) manusia(2) dari(3) bumi(4) yang(5) paling(6) gokil(7)"

    let input = Tensor::new([
        [0.0, 2.0],
        [1.0, 3.0],
        [4.0, 6.0],
        [5.0, 7.0],
    ]);

    let label = Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut model = Module::init();
    let mut optimazer = Adam::init(model.parameters(), 0.01);
    let loss_fn = CrossEntropyLoss::init();
    let embedding = model.embedding_init(8, 16);

    let linear = model.liniar_init(32, 8);

    let embedded = embedding.forward(&input);

    for epoch in 0..100 {
        let mut avg = Tensor::new([0.0]);
        for batch in 0..4 {
            let input = embedded.index(vec![batch]).reshape(vec![1, -1]);

            let logits = linear.forward(&input);
            let pred = softmax(&logits, -1);

            let loss = loss_fn.forward(&pred, &label);
            avg = &avg + &loss;

            optimazer.zero_grad();

            let backward = loss.backward();

            optimazer.optim(backward);
        }
        println!("epoch:{epoch} | loss avg:{}", &avg / 4.0);
    }
}
