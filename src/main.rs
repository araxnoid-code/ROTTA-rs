use rotta_rs::{ arrayy::{ r, ArrSlice }, concat, Tensor };

fn main() {
    let tensor_a = Tensor::arange(0, 12, 1).reshape(vec![-1, 3]);
    println!("{}", tensor_a);
    // [
    //  [0.0, 1.0, 2.0]
    //  [3.0, 4.0, 5.0]
    //  [6.0, 7.0, 8.0]
    //  [9.0, 10.0, 11.0]
    // ]

    // before 0.0.5
    // let sum = tensor_a.sum_axis(0); // in version 0.0.5, it can no longer be done

    // 0.0.5
    let slicing = tensor_a.sum_axis(&[0]);
    println!("{}", slicing);
    // [18.0, 22.0, 26.0]

    let slicing = tensor_a.sum_axis_keep_dim(&[0, 1]);
    println!("{}", slicing);
    // [
    //  [66.0]
    // ]
}
