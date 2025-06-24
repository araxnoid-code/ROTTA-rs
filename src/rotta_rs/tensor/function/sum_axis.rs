use crate::rotta_rs::{
    arrayy::sum_axis_arr,
    to_shape_arr,
    Arrayy,
    BackwardLabel,
    NodeType,
    Tensor,
};

pub fn sum_axis(x: &Tensor, d: i32) -> Tensor {
    let array = x.value();
    let tensor = Tensor::from_arrayy(sum_axis_arr(&array, d));
    tensor.update_parent(vec![x.node.clone()]);
    tensor.node.lock().unwrap().label = Some(BackwardLabel::SumAxis(x.node.clone(), d, false));

    tensor
}

pub fn sum_axis_keep_dim(x: &Tensor, d: i32) -> Tensor {
    let array = x.value();

    let sum = sum_axis_arr(&array, d);
    let mut keep_dim = sum.shape.clone();
    keep_dim.insert(d as usize, 1);

    let tensor = Tensor::from_vector(keep_dim, sum.value);
    tensor.update_parent(vec![x.node.clone()]);
    tensor.node.lock().unwrap().label = Some(BackwardLabel::SumAxis(x.node.clone(), d, true));

    tensor
}

pub fn d_sum_axis(x: &NodeType, d: i32, keep_dim: bool, grad: &Arrayy) {
    let mut x = x.lock().unwrap();

    if x.requires_grad {
        if !keep_dim {
            let ones = Arrayy::ones(x.value.shape.clone());
            let mut new_shape = grad.shape.clone();
            new_shape.insert(d as usize, 1);

            let d = ones * to_shape_arr(grad, new_shape);
            x.add_grad(d);
        } else {
            let ones = Arrayy::ones(x.value.shape.clone());
            let d = ones * grad;
            x.add_grad(d);
        }
    }
}
