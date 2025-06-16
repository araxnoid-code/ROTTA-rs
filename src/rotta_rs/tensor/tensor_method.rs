use ndarray::Axis;

use crate::rotta_rs::{ Arrayy, NdArray, NodeType, Tensor };

impl Tensor {
    pub fn exp(&self) -> Tensor {
        let output = self.value().exp();

        let tensor = Tensor::new(output.clone());
        tensor.update_parent(vec![self.node.clone()]);
        tensor.node.lock().as_mut().unwrap().label = Some(
            super::BackwardLabel::Exp(self.node.clone(), output)
        );

        tensor
    }

    // pub fn sum_axis(&self, axis: Axis) {
    // let sum_axis = self.value().sum_axis(axis);

    // let tensor = Tensor::new(sum_axis);
    // tensor.update_parent(vec![self.node.clone()]);
    // tensor.node.lock().as_mut().unwrap().label = Some(
    //     super::BackwardLabel::Exp(self.node.clone(), output)
    // );
    // }
}

pub fn d_exp(a: &NodeType, exp_value: Arrayy, grad: Arrayy) {
    let d_a = exp_value * grad;
    a.lock().as_mut().unwrap().add_grad(d_a);
}
