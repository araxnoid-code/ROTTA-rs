use ndarray::Axis;

use crate::rotta_rs::{ NdArray, NodeType, Tensor };

pub fn softmax(x: &Tensor) {
    let x_value = x.node.lock().unwrap().value.clone();

    let reshape = [x_value.dim().0, 1];
    let x_exp = x_value.exp();
    let output = &x_exp / &x_exp.sum_axis(Axis(1)).to_shape(reshape).unwrap(); // sum
    println!("{}", output)
}

pub fn d_softmax(x: &NodeType, grad: &NdArray) {}
