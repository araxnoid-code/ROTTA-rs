use crate::rotta_rs::{ BackwardLabel, NdArray, Node, NodeType, Tensor };

pub fn relu(x: &Tensor) -> Tensor {
    // f(x) = if x >= 0 x, if x < 0 0
    let value = x.value();
    let output = value.map(|x| {
        if *x >= 0.0 { *x } else { 0.0 }
    });

    let tensor = Tensor::new(output);
    tensor.update_parent(vec![x.node.clone()]);
    tensor.node.lock().as_mut().unwrap().label = Some(BackwardLabel::Relu(x.node.clone()));

    tensor
}

pub fn d_relu(x: &NodeType, grad: &NdArray) {
    let mut x_lock = x.lock();

    // f(x) = if x >= 0 1, if x < 0 0
    let d_x =
        x_lock
            .as_ref()
            .unwrap()
            .value.map(|x| {
                if *x >= 0.0 { 1.0 } else { 0.0 }
            }) * grad;

    x_lock.as_mut().unwrap().add_grad(&d_x);
}
