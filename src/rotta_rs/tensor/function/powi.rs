use crate::rotta_rs::{ Arrayy, BackwardLabel, NodeType, Tensor };

pub fn powi(x: &Tensor, n: i32) -> Tensor {
    let arrayy = x.value().powi(n);
    let tensor = Tensor::from_arrayy(arrayy);
    tensor.update_parent(vec![x.node.clone()]);
    tensor.update_label(Some(BackwardLabel::Powi(x.node.clone(), n)));

    tensor
}

pub fn d_powi(x: &NodeType, powi: i32, grad: &Arrayy) {
    // d/x = n * x^n-1
    let dx =
        (powi as f64) *
        &x
            .lock()
            .unwrap()
            .value.powi(powi - 1) *
        grad;

    x.lock().unwrap().add_grad(dx);
}
