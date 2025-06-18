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
    tensor.node.lock().unwrap().label = Some(BackwardLabel::SumAxis(x.node.clone(), d, false));

    tensor
}

pub fn d_sum_axis(x: &NodeType, d: usize, keep_dim: bool, grad: &Arrayy) {
    if !keep_dim {
        let ones = Arrayy::ones(x.lock().unwrap().value.shape.clone());
        let mut new_shape = grad.shape.clone();
        new_shape.insert(d, 1);

        let d = ones * reshape(grad, new_shape);
        x.lock().unwrap().add_grad(d);
    } else {
        let ones = Arrayy::ones(x.lock().unwrap().value.shape.clone());
        let d = ones * grad;
        x.lock().unwrap().add_grad(d);
    }
}

pub fn sum_axis_keep_dim(x: &Tensor, d: usize) -> Tensor {
    let array = x.value();

    let sum = sum_axis_arr(&array, d);
    let mut keep_dim = sum.shape.clone();
    keep_dim.insert(d, 1);

    let tensor = Tensor::from_vector(keep_dim, sum.value);
    tensor.update_parent(vec![x.node.clone()]);
    tensor.node.lock().unwrap().label = Some(BackwardLabel::SumAxis(x.node.clone(), d, true));

    tensor
}
