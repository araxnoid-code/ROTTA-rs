use crate::rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor };

// matmul
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let tensor_a = a.node.lock().unwrap();
    let tensor_b = b.node.lock().unwrap();

    // let output = if tensor_a.multithread {
    // tensor_a.value.par_matmul(&tensor_b.value)
    // } else {
    let output = tensor_a.value.matmul(&tensor_b.value);
    // };

    let tensor = Tensor::from_arrayy(output);

    tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
    tensor.node.lock().as_mut().unwrap().label = Some(
        BackwardLabel::Matmul(a.node.clone(), b.node.clone())
    );

    tensor
}

pub fn d_matmul(a: &NodeType, b: &NodeType, grad: &Arrayy) {
    // let mut a = a.lock().unwrap();

    // let mut b = b.lock().unwrap();

    // da = grad * b^t
    if a.lock().unwrap().requires_grad {
        let d_a = grad.matmul(&b.lock().unwrap().value.t());
        a.lock().unwrap().add_grad(d_a);
    }

    // // db = a * grad
    if b.lock().unwrap().requires_grad {
        let d_b = a.lock().unwrap().value.t().matmul(grad);
        b.lock().unwrap().add_grad(d_b);
    }
}
