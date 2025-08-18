use crate::{ rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, Tensor }, ShareTensor };

// dot
pub fn dot(a: &Tensor, b: &Tensor) -> Tensor {
    // a dot(b) = c
    let output = a.value.read().unwrap().dot(&b.value.read().unwrap());

    let mut tensor = Tensor::from_arrayy(output);
    tensor.update_parent(vec![a.shared_tensor(), b.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::Dot(a.shared_tensor(), b.shared_tensor())));

    tensor
}

pub fn d_dot(a: &ShareTensor, b: &ShareTensor, grad: &Arrayy) {
    // d/da = b * grad
    if a.requires_grad() {
        let d_a = &*b.value.read().unwrap() * grad;
        a.add_grad(d_a);
    }

    // db = a * grad
    if b.requires_grad() {
        let d_b = &*a.value.read().unwrap() * grad;
        b.add_grad(d_b);
    }
}
