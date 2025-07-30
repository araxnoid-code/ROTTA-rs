use std::ops::Mul;

use crate::{
    rotta_rs_module::{
        arrayy::{ broadcast_concat, Arrayy, MultipleSum },
        broadcasting_tensor_non_panic,
        BackwardLabel,
        NodeType,
        Tensor,
    },
    ShareTensor,
};

// add
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_arr = &*a.value.read().unwrap();
    let b_arr = &*b.value.read().unwrap();

    if a_arr.shape.multiple_sum() == 1 || b_arr.shape.multiple_sum() == 1 {
        // skalar
        let output = a_arr * b_arr;

        let mut tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![a.shared_tensor(), b.shared_tensor()]);
        tensor.update_label(Some(BackwardLabel::Mul(a.shared_tensor(), b.shared_tensor())));

        tensor
    } else if a_arr.shape == b_arr.shape {
        // same shape
        let output = a_arr * b_arr;

        let mut tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![a.shared_tensor(), b.shared_tensor()]);
        tensor.update_label(Some(BackwardLabel::Mul(a.shared_tensor(), b.shared_tensor())));

        tensor
    } else {
        // broadcasting
        let broadcast_shape = broadcast_concat(&a.value.read().unwrap(), &b.value.read().unwrap());

        let broadcast_a = broadcasting_tensor_non_panic(a, broadcast_shape.clone());
        let broadcast_b = broadcasting_tensor_non_panic(b, broadcast_shape);

        let output = &*broadcast_a.value.read().unwrap() * &*broadcast_b.value.read().unwrap();
        let mut tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![broadcast_a.shared_tensor(), broadcast_b.shared_tensor()]);
        tensor.update_label(
            Some(BackwardLabel::Mul(broadcast_a.shared_tensor(), broadcast_b.shared_tensor()))
        );

        tensor
    }
}

pub fn d_mul(a: &ShareTensor, b: &ShareTensor, grad: &Arrayy) {
    // da = b * grad
    if a.requires_grad() {
        let da = if a.value.read().unwrap().shape.multiple_sum() == 1 {
            let da = &*b.value.read().unwrap() * grad;
            Arrayy::from_vector(a.value.read().unwrap().shape.clone(), vec![da.sum()])
        } else {
            let da = &*b.value.read().unwrap() * grad;
            da
        };
        a.add_grad(da);
    }

    // db = a * grad
    if b.requires_grad() {
        let db = if b.value.read().unwrap().shape.multiple_sum() == 1 {
            let db = &*a.value.read().unwrap() * grad;
            Arrayy::from_vector(b.value.read().unwrap().shape.clone(), vec![db.sum()])
        } else {
            let db = &*a.value.read().unwrap() * grad;
            db
        };

        b.add_grad(db);
    }
}

// method

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        mul(self, rhs)
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f64) -> Self::Output {
        let rhs = Tensor::from_vector(vec![1], vec![rhs]);
        rhs.set_requires_grad(false);
        mul(self, &rhs)
    }
}

impl Mul<&Tensor> for f64 {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        let float = Tensor::from_vector(vec![1], vec![self]);
        float.set_requires_grad(false);
        mul(&float, rhs)
    }
}
