use ndarray::array;

use crate::rotta_rs::{ add, matmul, Tensor };

mod rotta_rs;

fn main() {
    let x = Tensor::new(array![[2.0]]);
    let w_1 = Tensor::new(array![[3.0]]);
    let w_2 = Tensor::new(array![[4.0]]);

    let h1 = matmul(&x, &w_1);
    let h2 = matmul(&h1, &w_2);
    let z = add(&h2, &h1);

    z.backward();
    // println!("{}", h1);
    // println!("{}", h2);
    // println!("{}", z);

    println!("{}", x.node.lock().unwrap().grad)
}
