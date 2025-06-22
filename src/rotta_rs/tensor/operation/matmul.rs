use crate::rotta_rs::{ Arrayy, BackwardLabel, NodeType, Tensor };

// matmul
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let output = a.value().matmul(&b.value());

    let tensor = Tensor::from_arrayy(output);

    tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
    tensor.node.lock().as_mut().unwrap().label = Some(
        BackwardLabel::Matmul(a.node.clone(), b.node.clone())
    );

    tensor
}

pub fn d_matmul(a: &NodeType, b: &NodeType, grad: &Arrayy) {
    // da = grad * b^t
    let d_a = grad.matmul(&b.lock().unwrap().value.clone().t());
    a.lock().as_mut().unwrap().add_grad(d_a);

    // // db = a * grad
    let d_b = a.lock().unwrap().value.clone().t().matmul(grad);
    b.lock().as_mut().unwrap().add_grad(d_b);
}
