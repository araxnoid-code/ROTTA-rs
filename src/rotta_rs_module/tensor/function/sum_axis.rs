use crate::{
    arrayy::sum_axis_keep_dim_arr,
    rotta_rs_module::{
        arrayy::{ sum_axis_arr, to_shape_arr, Arrayy },
        BackwardLabel,
        NodeType,
        Tensor,
    },
};

pub fn sum_axis(x: &Tensor, d: &[i32]) -> Tensor {
    let array = x.value();
    let tensor = Tensor::from_arrayy(sum_axis_arr(&array, d));
    tensor.update_parent(vec![x.node.clone()]);
    tensor.node.write().unwrap().label = Some(
        BackwardLabel::SumAxis(
            x.node.clone(),
            {
                let mut vec = d.to_vec();
                vec.sort();
                vec
            },
            false
        )
    );

    tensor
}

pub fn sum_axis_keep_dim(x: &Tensor, d: &[i32]) -> Tensor {
    let array = x.value();

    let sum = sum_axis_keep_dim_arr(&array, d);

    let tensor = Tensor::from_arrayy(sum);
    tensor.update_parent(vec![x.node.clone()]);
    tensor.node.write().unwrap().label = Some(
        BackwardLabel::SumAxis(x.node.clone(), d.to_vec(), true)
    );

    tensor
}

pub fn d_sum_axis(x: &NodeType, d: &[i32], keep_dim: bool, grad: &Arrayy) {
    let _x = x.read().unwrap();

    if _x.requires_grad {
        if !keep_dim {
            let ones = Arrayy::ones(_x.value.shape.clone());
            let mut new_shape = grad.shape.clone();

            for d in d {
                new_shape.insert(*d as usize, 1);
            }

            let d = ones * to_shape_arr(grad, new_shape);
            x.write().unwrap().add_grad(d);
        } else {
            let ones = Arrayy::ones(_x.value.shape.clone());
            // println!("{}", grad);
            let d = ones * grad;
            x.write().unwrap().add_grad(d);
        }
    }
}
