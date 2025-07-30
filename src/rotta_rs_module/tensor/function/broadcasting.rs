use crate::{
    rotta_rs_module::{
        arrayy::{ broadcasting, sum_axis_arr, to_shape_arr, Arrayy },
        BackwardLabel,
        Tensor,
    },
    ShareTensor,
};

// broadcasting_tensor
pub fn broadcasting_tensor_non_panic(tensor_arr: &Tensor, broadcast_shape: Vec<usize>) -> Tensor {
    let arr = broadcasting(&tensor_arr.value.read().unwrap(), broadcast_shape);

    let mut tensor = Tensor::from_arrayy(arr);
    tensor.update_parent(vec![tensor_arr.shared_tensor()]);
    tensor.update_label(
        Some(BackwardLabel::Broadcasting(tensor_arr.shared_tensor(), tensor.value()))
    );

    tensor
}

pub fn d_broadcasting_tensor(tensor_arr: &ShareTensor, broad_arr: Arrayy, grad: &Arrayy) {
    // let mut _tensor_arr = tensor_arr.read().unwrap();

    if tensor_arr.requires_grad() {
        let broadcasted_shape = &broad_arr.shape;

        let pre_shape = tensor_arr.value.read().unwrap().shape.clone();
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

        let mut sum = grad.clone();

        for sum_d in sum_list {
            sum = sum_axis_arr(&sum, &[sum_d as i32]);
        }

        let d_arr = to_shape_arr(&sum, pre_shape);
        tensor_arr.add_grad(d_arr);
    }
}
