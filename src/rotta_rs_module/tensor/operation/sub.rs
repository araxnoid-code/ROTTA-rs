use std::ops::Sub;

use crate::{
    rotta_rs_module::{
        arrayy::{ broadcast_concat, Arrayy, MultipleSum },
        broadcasting_tensor_non_panic,
        BackwardLabel,
        Tensor,
    },
    ShareTensor,
};

pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    let a_arr = &*a.value.read().unwrap();
    let b_arr = &*b.value.read().unwrap();

    if a_arr.shape.multiple_sum() == 1 || b_arr.shape.multiple_sum() == 1 {
        // skalar
        let output = a_arr - b_arr;

        let mut tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![a.shared_tensor(), b.shared_tensor()]);
        tensor.update_label(Some(BackwardLabel::Sub(a.shared_tensor(), b.shared_tensor())));

        tensor
    } else if a_arr.shape == b_arr.shape {
        // same shape
        let output = a_arr - b_arr;

        let mut tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![a.shared_tensor(), b.shared_tensor()]);
        tensor.update_label(Some(BackwardLabel::Sub(a.shared_tensor(), b.shared_tensor())));

        tensor
    } else {
        // broadcasting
        let broadcast_shape = broadcast_concat(&a.value.read().unwrap(), &b.value.read().unwrap());

        let broadcast_a = broadcasting_tensor_non_panic(a, broadcast_shape.clone());
        let broadcast_b = broadcasting_tensor_non_panic(b, broadcast_shape);

        let output = &*broadcast_a.value.read().unwrap() - &*broadcast_b.value.read().unwrap();
        let mut tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![broadcast_a.shared_tensor(), broadcast_b.shared_tensor()]);
        tensor.update_label(
            Some(BackwardLabel::Sub(broadcast_a.shared_tensor(), broadcast_b.shared_tensor()))
        );

        tensor
    }
}

pub fn d_sub(a: &ShareTensor, b: &ShareTensor, grad: &Arrayy) {
    // d/da = 1  * grad = grad
    if a.requires_grad() {
        let d_a = if a.value.read().unwrap().shape.multiple_sum() == 1 {
            Arrayy::from_vector(a.value.read().unwrap().shape.clone(), vec![grad.sum()])
        } else {
            grad.clone()
        };
        a.add_grad(d_a);
    }

    // db = -1 * grad = -grad
    if b.requires_grad() {
        let d_b = if b.value.read().unwrap().shape.multiple_sum() == 1 {
            Arrayy::from_vector(b.value.read().unwrap().shape.clone(), vec![grad.sum()])
        } else {
            grad.clone()
        };
        b.add_grad(-1.0 * d_b);
    }
}

// method

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        sub(self, rhs)
    }
}

impl Sub<f32> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f32) -> Self::Output {
        let rhs = Tensor::from_vector(vec![1], vec![rhs]);
        rhs.set_requires_grad(false);
        sub(self, &rhs)
    }
}

impl Sub<&Tensor> for f32 {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        let float = Tensor::from_vector(vec![1], vec![self]);
        float.set_requires_grad(false);
        sub(&float, rhs)
    }
}
