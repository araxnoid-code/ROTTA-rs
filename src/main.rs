use rotta_rs::{ arrayy::{ concat_arr, ArrSlice, Arrayy }, * };

// struct MyDataset {
//     input: Vec<Tensor>,
//     label: Vec<Tensor>,
// }

// impl MyDataset {
//     pub fn init(input: Vec<Tensor>, label: Vec<Tensor>) -> MyDataset {
//         MyDataset {
//             input,
//             label,
//         }
//     }
// }

// impl Dataset for MyDataset {
//     fn get(&self, idx: usize) -> (Tensor, Tensor) {
//         (self.input[idx].clone(), self.label[idx].clone())
//     }

//     fn len(&self) -> usize {
//         self.input.len()
//     }
// }

fn main() {
    // println!("{}", a.grad());

    // let mut model = Module::init();
    // let mut optimazer = RMSprop::init(model.parameters(), 0.001);
    // let loss_fn = SSResidual::init();

    // let linear = model.liniar_init(1, 128);
    // let linear_2 = model.liniar_init(128, 1);
    // let mut dropout = model.dropout_init(0.01);

    // // let input = Tensor::new([[1.0], [2.0], [3.0], [4.0]]);
    // // let actual = Tensor::new([[10.0], [20.0], [30.0], [40.0]]);

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
    //         let x = dropout.forward(&x);
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
