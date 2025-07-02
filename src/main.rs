use std::{ ops::{ Range, RangeFrom, RangeFull }, time::SystemTime };

use rotta_rs::{
    arrayy::{ r, slice_arr_optim, ArrSlice, Arrayy },
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
    // let arr = Arrayy::zeros(vec![256, 256]);
    // let tick = std::time::SystemTime
    //     ::now()
    //     .duration_since(SystemTime::UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    // // arr.slice(vec![ArrSlice(None, None), ArrSlice(None, None)]);

    // let tock = std::time::SystemTime
    //     ::now()
    //     .duration_since(SystemTime::UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    // println!("{}ms", tock - tick)

    // slice_arr_optim(
    //     &arr,
    //     vec![ArrSlice(Some(0), Some(2)), ArrSlice(Some(1), Some(2)), ArrSlice(Some(2), Some(3))]
    // );

    // let mut model = Module::init();
    // let mut optimazer = Adam::init(model.parameters(), 0.001);
    // let loss_fn = SSResidual::init();

    // let linear = model.liniar_init(1, 256);
    // let linear_2 = model.liniar_init(256, 1);

    // let dataset = MyDataset::init(
    //     vec![Tensor::new([[1.0], [2.0]]), Tensor::new([[3.0], [4.0]])],
    //     vec![Tensor::new([[10.0], [20.0]]), Tensor::new([[30.0], [40.0]])]
    // );
    // let mut datahandler = DataHandler::init(dataset);
    // datahandler.shuffle();

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
