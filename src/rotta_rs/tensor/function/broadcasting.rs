use crate::rotta_rs::{
    broadcasting,
    reshape,
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

    broadcasted_shape
        .iter()
        .enumerate()
        .for_each(|(d, broad)| {
            if *broad != pre_shape[d] {
                sum_list.push(d);
            }
        });

    let mut sum = grad;

    for sum_d in sum_list {
        sum = sum_axis_arr(&sum, sum_d);
    }

    let d_arr = reshape(&sum, pre_shape);
    tensor_arr.lock().unwrap().add_grad(d_arr);
}
