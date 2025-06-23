use crate::rotta_rs::{ Arrayy, BackwardLabel, NodeType, Tensor };

pub fn permute(x: &Tensor, order: Vec<usize>) -> Tensor {
    let tensor = Tensor::from_arrayy(x.value().permute(&order));
    tensor.update_parent(vec![x.node.clone()]);
    tensor.update_label(Some(BackwardLabel::Permute(x.node.clone(), order)));

    tensor
}

pub fn d_permute(x: &NodeType, order: Vec<usize>, grad: &Arrayy) {
    let mut x = x.lock().unwrap();

    let mut new_order = order.clone();
    for (i, d) in order.iter().enumerate() {
        new_order[*d] = i;
    }

    let d_x = grad.permute(&new_order);
    x.add_grad(d_x);
}
