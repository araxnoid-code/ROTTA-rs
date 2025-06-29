use crate::rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor };

pub fn powi(x: &Tensor, n: i32) -> Tensor {
    let arrayy = x.value().powi(n);
    let tensor = Tensor::from_arrayy(arrayy);
    tensor.update_parent(vec![x.node.clone()]);
    tensor.update_label(Some(BackwardLabel::Powi(x.node.clone(), n)));

    tensor
}

pub fn d_powi(x: &NodeType, powi: i32, grad: &Arrayy) {
    let mut x = x.lock().unwrap();

    // d/x = n * x^n-1
    if x.requires_grad {
        let dx = (powi as f64) * &x.value.powi(powi - 1) * grad;

        x.add_grad(dx);
    }
}
