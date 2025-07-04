use std::{ ops::{ Range, RangeFrom, RangeFull }, time::SystemTime };

use rotta_rs::{
    arrayy::{ broadcasting, mean_arr, r, ArrSlice, Arrayy },
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
    let a = Tensor::arange(&[10, 2, 1, 3]);

    let mut axis = vec![];
    a.shape()
        .into_iter()
        .enumerate()
        .for_each(|(i, _)| {
            if i != 1 {
                axis.push(i);
            }
        });

    println!("{:?}", axis);
}
