use crate::{
    rotta_rs_module::{ arrayy::{ to_shape_arr, Arrayy }, BackwardLabel, NodeType, Tensor },
    ShareTensor,
};

pub fn to_shape(x: &Tensor, to_shape: Vec<usize>) -> Tensor {
    let arr = to_shape_arr(&x.value.read().unwrap(), to_shape);
    let mut tensor = Tensor::from_arrayy(arr);
    tensor.update_parent(vec![x.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::ToShape(x.shared_tensor(), x.shape())));

    tensor
}

pub fn reshape(x: &Tensor, reshape: Vec<i32>) -> Tensor {
    let mut tensor = Tensor::from_arrayy(x.value.read().unwrap().reshape(reshape));
    tensor.update_parent(vec![x.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::ToShape(x.shared_tensor(), x.shape())));

    tensor
}

pub fn d_to_shape(x: &ShareTensor, to_shape: Vec<usize>, grad: &Arrayy) {
    *x.grad.write().unwrap() = grad.to_shape(to_shape);
}
