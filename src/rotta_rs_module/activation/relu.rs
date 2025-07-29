use crate::rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor };

#[allow(dead_code)]
pub fn relu(x: &Tensor) -> Tensor {
    // f(x) = if x >= 0 x, if x < 0 0
    let value = x.value();

    let output = value.map(|x| {
        if *x >= 0.0 { *x } else { 0.0 }
    });

    let tensor = Tensor::from_arrayy(output);
    tensor.update_parent(vec![x.node.clone()]);
    tensor.node.write().unwrap().label = Some(BackwardLabel::Relu(x.node.clone()));

    tensor
}

#[allow(dead_code)]
pub fn d_relu(x: &NodeType, grad: &Arrayy) {
    let _x_lock = x.read().unwrap();

    // f(x) = if x >= 0 1, if x < 0 0
    if _x_lock.requires_grad {
        let d_x =
            _x_lock.value.map(|x| {
                if *x >= 0.0 { 1.0 } else { 0.0 }
            }) * grad;

        x.write().unwrap().add_grad(d_x);
    }
}
