use std::{ ops::{ Range, RangeFrom, RangeFull }, time::SystemTime };

use rotta_rs::{
    arrayy::{ mean_arr, r, ArrSlice, Arrayy },
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
    let tensor = Tensor::arange(vec![2, 2, 3]);

    println!("{}", tensor);

    // println!("{}", tensor.sum_axis(-1))
}
