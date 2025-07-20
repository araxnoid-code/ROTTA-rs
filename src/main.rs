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

fn main() {
    let data_a = Tensor::rand(vec![1, 2]);

    let mut model = Module::init();

    let lstm = model.lstm_init(2);
    // println!("{}", lstm.w_o.grad());

    let out = lstm.forward(&data_a, None);
    out.hidden.backward();

    // println!("{}", lstm.b_o.grad())

    for params in model.parameters().lock().unwrap().iter() {
        let node = &params.lock().unwrap().grad;

        println!("{}", node);
    }

    // let data_a = Arrayy::new([
    //     [1.0, 2.0, 3.0],
    //     [4.0, 5.0, 6.0],
    // ]);
    // let data_b = Arrayy::new([
    //     [7.0, 8.0, 9.0],
    //     [10.0, 11.0, 12.0],
    // ]);

    // println!("{}", concat_arr(vec![data_a, data_b], -1))
}
