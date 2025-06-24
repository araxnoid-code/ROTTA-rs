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
    let prediction_value = prediction.lock().unwrap();
    let mut actual_value = actual.lock().unwrap();

    // df/dprediction = -2(actual - prediction) * grad
    if prediction_value.requires_grad {
        let d_predcition = -2.0 * (&actual_value.value - &prediction_value.value) * grad;
        prediction.lock().as_mut().unwrap().add_grad(d_predcition);
    }

    // df/dactual = 2(actual - prediction) * grad
    if actual_value.requires_grad {
        let d_actual = 2.0 * (&actual_value.value - &prediction_value.value) * grad;
        actual_value.add_grad(d_actual);
    }
}
