use crate::rotta_rs::{
    broadcasting,
    to_shape,
    sum_axis_arr,
    Arrayy,
    BackwardLabel,
    NodeType,
    Tensor,
};

// broadcasting_tensor
pub fn broadcasting_tensor_non_panic(tensor_arr: &Tensor, broadcast_shape: Vec<usize>) -> Tensor {
    let arr = broadcasting(&tensor_arr.value(), broadcast_shape).unwrap_or(tensor_arr.value());

    let tensor = Tensor::from_arrayy(arr);
    tensor.update_parent(vec![tensor_arr.node.clone()]);
    tensor.node.lock().unwrap().label = Some(
        BackwardLabel::Broadcasting(tensor_arr.node.clone(), tensor.value())
    );

    tensor
}

pub fn d_broadcasting_tensor(tensor_arr: &NodeType, broad_arr: Arrayy, grad: Arrayy) {
    let broadcasted_shape = &broad_arr.shape;

    let pre_shape = tensor_arr.lock().unwrap().value.shape.clone();
    let mut sum_list = vec![];

    let mut broad_rev = broadcasted_shape.clone();
    broad_rev.reverse();

    let mut pre_shape_rev = pre_shape.clone();
    pre_shape_rev.reverse();

    for d in 0..broad_rev.len() {
        if let Some(pre) = pre_shape_rev.get(d) {
            if &broad_rev[d] != pre {
                sum_list.push(broad_rev.len() - 1 - d);
            }
        } else {
            sum_list.push(broad_rev.len() - 1 - d);
        }
    }

    let mut sum = grad;

    for sum_d in sum_list {
        sum = sum_axis_arr(&sum, sum_d);
    }

    let d_arr = to_shape(&sum, pre_shape);
    tensor_arr.lock().unwrap().add_grad(d_arr);
}
