use crate::rotta_rs::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor };

pub fn relu(x: &Tensor) -> Tensor {
    // f(x) = if x >= 0 x, if x < 0 0
    let value = x.value();
    let output = value.map(|x| {
        if *x >= 0.0 { *x } else { 0.0 }
    });

    let tensor = Tensor::from_arrayy(output);
    tensor.update_parent(vec![x.node.clone()]);
    tensor.node.lock().as_mut().unwrap().label = Some(BackwardLabel::Relu(x.node.clone()));

    tensor
}

pub fn d_relu(x: &NodeType, grad: &Arrayy) {
    let mut x_lock = x.lock().unwrap();

    // f(x) = if x >= 0 1, if x < 0 0
    if x_lock.requires_grad {
        let d_x =
            x_lock.value.map(|x| {
                if *x >= 0.0 { 1.0 } else { 0.0 }
            }) * grad;

        x_lock.add_grad(d_x);
    }
}
