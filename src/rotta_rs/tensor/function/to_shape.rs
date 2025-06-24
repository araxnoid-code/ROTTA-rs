use crate::rotta_rs::{ to_shape_arr, Arrayy, BackwardLabel, NodeType, Tensor };

pub fn to_shape(x: &Tensor, to_shape: Vec<usize>) -> Tensor {
    let arr = to_shape_arr(&x.value(), to_shape);
    let tensor = Tensor::from_arrayy(arr);
    tensor.update_parent(vec![x.node.clone()]);
    tensor.update_label(Some(BackwardLabel::ToShape(x.node.clone(), x.shape())));

    tensor
}

pub fn d_to_shape(x: &NodeType, to_shape: Vec<usize>, grad: &Arrayy) {
    let mut x = x.lock().unwrap();

    x.grad = grad.to_shape(to_shape);
}
