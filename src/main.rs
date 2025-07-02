use std::{ ops::{ Range, RangeFrom, RangeFull }, time::SystemTime };

use rotta_rs::{
    arrayy::{ r, ArrSlice, Arrayy },
    concat,
    relu,
    Adam,
    DataHandler,
    Dataset,
    Module,
    SSResidual,
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
    // for epoch in 0..1000 {
    //     let mut avg = Tensor::new([0.0]);
    //     for (input, actual) in &mut datahandler {
    //         let x = linear.forward(&input);
    //         let x = relu(&x);
    //         let output = linear_2.forward(&x);

    //         let loss = loss_fn.forward(&output, &actual);
    //         avg = &avg + &loss;

    //         optimazer.zero_grad();

    //         let backward = loss.backward();

    //         optimazer.optim(backward);
    //     }
    //     let loss = &avg / (datahandler.len() as f64);
    //     println!("epoch:{epoch} | loss => {loss}");
    // }
}
