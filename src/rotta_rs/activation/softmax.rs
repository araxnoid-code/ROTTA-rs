use ndarray::{ Array2, Axis };

use crate::rotta_rs::{
    divided,
    exp,
    reshape,
    sum_axis,
    sum_axis_keep_dim,
    BackwardLabel,
    NdArray,
    NodeType,
    Tensor,
};

pub fn softmax(x: &Tensor, d: usize) -> Tensor {
    let exp = exp(x);
    let sum = sum_axis_keep_dim(&exp, d);

    let s = divided(&exp, &sum);
    s
}
