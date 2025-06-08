use ndarray::array;

use crate::rotta_rs::{ BackwardLabel, NdArray, NodeType, Tensor };

pub struct CrossEntropyLoss {}

impl CrossEntropyLoss {
    pub fn init() -> CrossEntropyLoss {
        CrossEntropyLoss {}
    }

    pub fn forward(&self, prob_prediction: &Tensor, prob_actual: &Tensor) -> Tensor {
        let epsilon = 0.1;
        let loss =
            (prob_actual.value() * (1.0 / (prob_prediction.value() + epsilon)).ln()).sum() /
            (prob_prediction.value().shape()[0] as f64);

        let tensor = Tensor::new(array![[loss]]);
        tensor.update_parent(vec![prob_prediction.node.clone(), prob_actual.node.clone()]);
        tensor.node.lock().as_mut().unwrap().label = Some(
            BackwardLabel::CEL(prob_prediction.node.clone(), prob_actual.node.clone())
        );

        tensor
    }
}

pub fn d_cel(prob_prediction: &NodeType, prob_actual: &NodeType, grad: &NdArray) {
    // df/dprediction = -prob_actual/prob_prediction
    let epsilon = 0.1;
    let prob_pred = prob_prediction.lock().unwrap().value.clone();
    let prob_actual = prob_actual.lock().unwrap().value.clone();

    let d_pred = -(prob_actual / (prob_pred + epsilon)) * grad;
    prob_prediction.lock().as_mut().unwrap().add_grad(&d_pred);
}
