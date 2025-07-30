use crate::{ rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, Tensor }, ShareTensor };

// matmul
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let output = a.value.read().unwrap().matmul(&b.value.read().unwrap());

    let mut tensor = Tensor::from_arrayy(output);

    tensor.update_parent(vec![a.shared_tensor(), b.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::Matmul(a.shared_tensor(), b.shared_tensor())));

    tensor
}

pub fn d_matmul(a: &ShareTensor, b: &ShareTensor, grad: &Arrayy) {
    if a.requires_grad() {
        let d_a = grad.matmul(&b.value.read().unwrap().t());
        a.add_grad(d_a);
    }

    if b.requires_grad() {
        let d_b = a.value.read().unwrap().t().matmul(grad);
        b.add_grad(d_b);
    }
}
