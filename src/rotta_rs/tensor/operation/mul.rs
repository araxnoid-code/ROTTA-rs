use std::ops::Mul;

use crate::rotta_rs::{
    broadcast_concat,
    broadcasting_tensor_non_panic,
    Arrayy,
    BackwardLabel,
    MultipleSum,
    NodeType,
    Tensor,
};

// add
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_arr = a.value();
    let b_arr = b.value();

    if a_arr.shape.multiple_sum() == 1 || b_arr.shape.multiple_sum() == 1 {
        // skalar
        let output = a_arr * b_arr;

        let tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
        tensor.node.lock().as_mut().unwrap().label = Some(
            BackwardLabel::Mul(a.node.clone(), b.node.clone())
        );

        tensor
    } else if a_arr.shape == b_arr.shape {
        // same shape
        let output = a_arr * b_arr;

        let tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
        tensor.node.lock().as_mut().unwrap().label = Some(
            BackwardLabel::Mul(a.node.clone(), b.node.clone())
        );

        tensor
    } else {
        // broadcasting
        let broadcast_shape = broadcast_concat(&a.value(), &b.value());

        let broadcast_a = broadcasting_tensor_non_panic(a, broadcast_shape.clone());
        let broadcast_b = broadcasting_tensor_non_panic(b, broadcast_shape);

        let output = broadcast_a.value() * broadcast_b.value();
        let tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![broadcast_a.node.clone(), broadcast_b.node.clone()]);
        tensor.node.lock().as_mut().unwrap().label = Some(
            BackwardLabel::Mul(broadcast_a.node.clone(), broadcast_b.node.clone())
        );

        tensor
    }
}

pub fn d_mul(a: &NodeType, b: &NodeType, grad: &Arrayy) {
    let mut a = a.lock().unwrap();
    let mut b = b.lock().unwrap();

    // da = b * grad
    if a.requires_grad {
        let da = if a.value.shape.multiple_sum() == 1 {
            let da = &b.value * grad;
            Arrayy::from_vector(a.value.shape.clone(), vec![da.sum()])
        } else {
            let da = &b.value * grad;
            da
        };
        a.add_grad(da);
    }

    // db = a * grad
    if b.requires_grad {
        let db = if b.value.shape.multiple_sum() == 1 {
            let db = &a.value * grad;
            Arrayy::from_vector(b.value.shape.clone(), vec![db.sum()])
        } else {
            let db = &a.value * grad;
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
