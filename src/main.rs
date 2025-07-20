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
    let token = Tensor::new([0.0, 1.0, 2.0, 3.0, 4.0]);

    let mut model = Module::init();
    let lstm = model.lstm_init(16);
    let embbeding = model.embedding_init(5, 16);
    let linear = model.liniar_init(16, 5);

    // embedding
    let embedded = embbeding.forward(&Tensor::new([0.0, 3.0, 4.0]));

    // lstm

    let mut cell_hidden: Option<LSTMCellHidden> = None;
    for i in 0..3 {
        let word = embedded.index(vec![i]).reshape(vec![1, -1]);

        let out = lstm.forward(&word, cell_hidden);
        cell_hidden = Some(out);
    }

    let hidden = cell_hidden.unwrap().hidden;
    let linear = linear.forward(&hidden);
    let prob = softmax(&linear, -1);
    println!("{}", lstm.w_f.grad());
    prob.backward();

    println!("{}", lstm.w_f.grad())
}
