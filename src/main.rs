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

use rotta_rs::{ arrayy::{ concat_arr, r, Arrayy }, * };

fn main() {}
