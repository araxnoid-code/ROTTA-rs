use crate::rotta_rs::{ Arrayy, BackwardLabel, Node, NodeType, Tensor };

pub struct SSResidual {}

impl SSResidual {
    pub fn init() -> SSResidual {
        SSResidual {}
    }

    pub fn forward(&self, prediction: &Tensor, actual: &Tensor) -> Tensor {
        // f = sum((actual - prediction)^2)
        let prediction_value = prediction.value();
        let actual_value = actual.value();

        let output = (actual_value - prediction_value).powi(2).sum();
        let tensor = Tensor::from_vector(vec![1], vec![output]);
        tensor.update_parent(vec![prediction.node.clone(), actual.node.clone()]);
        tensor.node.lock().unwrap().label = Some(
            BackwardLabel::SSResidual(prediction.node.clone(), actual.node.clone())
        );

        tensor
    }
}

pub fn d_ssresidual(prediction: &NodeType, actual: &NodeType, grad: &Arrayy) {
    let prediction_value = prediction.lock().unwrap().value.clone();
    let actual_value = actual.lock().unwrap().value.clone();

    // df/dprediction = -2(actual - prediction) * grad
    let d_predcition =
        Arrayy::from_vector(vec![1], vec![-2.0]) * (&actual_value - &prediction_value) * grad;
    prediction.lock().as_mut().unwrap().add_grad(d_predcition);

    // df/dactual = 2(actual - prediction) * grad
    let d_actual =
        Arrayy::from_vector(vec![1], vec![2.0]) * (actual_value - prediction_value) * grad;
    actual.lock().as_mut().unwrap().add_grad(d_actual);
}
