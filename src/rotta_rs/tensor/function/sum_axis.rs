use crate::rotta_rs::{
    arrayy_external::sum_axis_arr,
    reshape,
    Arrayy,
    BackwardLabel,
    NodeType,
    Tensor,
};

pub fn sum_axis(x: &Tensor, d: usize) -> Tensor {
    let array = x.value();
    let tensor = Tensor::from_arrayy(sum_axis_arr(&array, d));
    tensor.update_parent(vec![x.node.clone()]);
    tensor.node.lock().unwrap().label = Some(BackwardLabel::SumAxis(x.node.clone(), d));

    tensor
}

pub fn d_sum_axis(x: &NodeType, d: usize, grad: &Arrayy) {
    let ones = Arrayy::ones(x.lock().unwrap().value.shape.clone());
    let mut new_shape = grad.shape.clone();
    new_shape.insert(d, 1);

    let d = ones * reshape(grad, new_shape);
    x.lock().unwrap().add_grad(d);
}
