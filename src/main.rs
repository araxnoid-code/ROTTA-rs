use rotta_rs::*;

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
    let input_1 = Tensor::new([[1.0, 2.0, 3.0, 4.0, 5.0]]);
    let label_1 = Tensor::new([[2.0, 3.0, 4.0, 5.0, 6.0]]);

    let input_2 = Tensor::new([[11.0, 12.0, 13.0, 14.0, 15.0]]);
    let label_2 = Tensor::new([[12.0, 13.0, 14.0, 15.0, 16.0]]);

    let dataset = MyDataset::init(vec![input_1, input_2], vec![label_1, label_2]);
    let mut datahandler = DataHandler::init(dataset);
    datahandler.batch(32); // set how many batches will be issued
    datahandler.shuffle(); // randomize the order of the dataset

    for (input, label) in &mut datahandler {
        // perform deep learning operations here
    }
}
