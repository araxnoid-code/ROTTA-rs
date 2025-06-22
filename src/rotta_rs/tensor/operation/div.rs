use std::ops::Div;

use crate::rotta_rs::{
    broadcast_concat,
    broadcasting_tensor_non_panic,
    Arrayy,
    BackwardLabel,
    MultipleSum,
    NodeType,
    Tensor,
};

pub fn divided(a: &Tensor, b: &Tensor) -> Tensor {
    let a_arr = a.value();
    let b_arr = b.value();

    if a_arr.shape.multiple_sum() == 1 || b_arr.shape.multiple_sum() == 1 {
        // skalar
        let output = a_arr / b_arr;

        let tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
        tensor.node.lock().as_mut().unwrap().label = Some(
            BackwardLabel::Diveded(a.node.clone(), b.node.clone())
        );

        tensor
    } else if a_arr.shape == b_arr.shape {
        // same shape
        let output = a_arr / b_arr;

        let tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
        tensor.node.lock().as_mut().unwrap().label = Some(
            BackwardLabel::Diveded(a.node.clone(), b.node.clone())
        );

        tensor
    } else {
        // broadcasting
        let broadcast_shape = broadcast_concat(&a.value(), &b.value());

        let broadcast_a = broadcasting_tensor_non_panic(a, broadcast_shape.clone());
        let broadcast_b = broadcasting_tensor_non_panic(b, broadcast_shape);

        let output = broadcast_a.value() / broadcast_b.value();
        let tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![broadcast_a.node.clone(), broadcast_b.node.clone()]);
        tensor.node.lock().as_mut().unwrap().label = Some(
            BackwardLabel::Diveded(broadcast_a.node.clone(), broadcast_b.node.clone())
        );

        tensor
    }
}

pub fn d_divided(a: &NodeType, b: &NodeType, grad: &Arrayy) {
    let a_value = a.lock().unwrap().value.clone();
    let b_value = b.lock().unwrap().value.clone();

    // da = 1/b
    let da = if a.lock().unwrap().value.shape.multiple_sum() == 1 {
        let da = (Arrayy::from_vector(vec![1], vec![1.0]) / &b_value) * grad;
        Arrayy::from_vector(a.lock().unwrap().value.shape.clone(), vec![da.sum()])
    } else {
        let da = (Arrayy::from_vector(vec![1], vec![1.0]) / &b_value) * grad;
        da
    };
    a.lock().unwrap().add_grad(da);

    // db = -a/b^2
    let db = if b.lock().unwrap().value.shape.multiple_sum() == 1 {
        let db = -1.0 * (a_value / b_value.powi(2)) * grad;
        Arrayy::from_vector(b.lock().unwrap().value.shape.clone(), vec![db.sum()])
    } else {
        let db = -1.0 * (a_value / b_value.powi(2)) * grad;
        db
    };
    b.lock().unwrap().add_grad(db);
}

// method

impl Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Self::Output {
        divided(self, rhs)
    }
}

impl Div<f64> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: f64) -> Self::Output {
        let rhs = Tensor::from_vector(vec![1], vec![rhs]);
        divided(self, &rhs)
    }
}

impl Div<&Tensor> for f64 {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Self::Output {
        let float = Tensor::from_vector(vec![1], vec![self]);
        divided(&float, rhs)
    }
}
