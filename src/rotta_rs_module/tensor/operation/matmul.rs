use crate::rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor };

// matmul
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let tensor_a = a.node.read().unwrap();
    let tensor_b = b.node.read().unwrap();

    // let output = if tensor_a.multithread {
    // tensor_a.value.par_matmul(&tensor_b.value)
    // } else {
    let output = tensor_a.value.matmul(&tensor_b.value);
    // };

    let tensor = Tensor::from_arrayy(output);

    tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
    tensor.node.write().unwrap().label = Some(
        BackwardLabel::Matmul(a.node.clone(), b.node.clone())
    );

    tensor
}

pub fn d_matmul(a: &NodeType, b: &NodeType, grad: &Arrayy) {
    let (d_a, d_b) = {
        let mut _a = a.read().unwrap();
        let mut _b = b.read().unwrap();

        // da = grad * b^t
        let d_a = if _a.requires_grad {
            let d_a = grad.matmul(&_b.value.t());
            Some(d_a)
        } else {
            None
        };

        // // db = a * grad
        let d_b = if _b.requires_grad {
            let d_b = _a.value.t().matmul(grad);
            Some(d_b)
        } else {
            None
        };

        (d_a, d_b)
    };

    if let Some(d_a) = d_a {
        a.write().unwrap().add_grad(d_a);
    }

    if let Some(d_b) = d_b {
        b.write().unwrap().add_grad(d_b);
    }
}
