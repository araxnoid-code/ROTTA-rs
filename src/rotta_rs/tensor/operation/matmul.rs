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
    let mut a = a.lock().unwrap();
    let mut b = b.lock().unwrap();

    // da = grad * b^t
    if a.requires_grad {
        let d_a = grad.matmul(&b.value.t());
        a.add_grad(d_a);
    }

    // // db = a * grad
    if b.requires_grad {
        let d_b = a.value.t().matmul(grad);
        b.add_grad(d_b);
    }
}
