use crate::{ rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, Tensor }, ShareTensor };

#[allow(dead_code)]
pub fn relu(x: &Tensor) -> Tensor {
    // f(x) = if x >= 0 x, if x < 0 0
    let value = x.value.read().unwrap();

    let output = value.map(|x| {
        if *x >= 0.0 { *x } else { 0.0 }
    });

    let mut tensor = Tensor::from_arrayy(output);
    tensor.update_parent(vec![x.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::Relu(x.shared_tensor())));

    tensor
}

#[allow(dead_code)]
pub fn d_relu(x: &ShareTensor, grad: &Arrayy) {
    // f(x) = if x >= 0 1, if x < 0 0
    if x.requires_grad() {
        let d_x =
            x.value
                .read()
                .unwrap()
                .map(|x| {
                    if *x >= 0.0 { 1.0 } else { 0.0 }
                }) * grad;

        x.add_grad(d_x);
    }
}
