use crate::{ rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, Tensor }, ShareTensor };

pub struct SSResidual {}

impl SSResidual {
    pub fn init() -> SSResidual {
        SSResidual {}
    }

    pub fn forward(&self, prediction: &Tensor, actual: &Tensor) -> Tensor {
        // f = sum((actual - prediction)^2)
        let prediction_value = &*prediction.value.read().unwrap();
        let actual_value = &*actual.value.read().unwrap();

        let output = (actual_value - prediction_value).powi(2).sum();
        let mut tensor = Tensor::from_vector(vec![1], vec![output]);
        tensor.update_parent(vec![prediction.shared_tensor(), actual.shared_tensor()]);
        tensor.update_label(
            Some(BackwardLabel::SSResidual(prediction.shared_tensor(), actual.shared_tensor()))
        );
        tensor
    }
}

pub fn d_ssresidual(prediction: &ShareTensor, actual: &ShareTensor, grad: &Arrayy) {
    // let mut prediction_value = prediction.write().unwrap();
    // let mut actual_value = actual.write().unwrap();

    // df/dprediction = -2(actual - prediction) * grad
    if prediction.requires_grad() {
        let d_predcition =
            -2.0 * (&*actual.value.read().unwrap() - &*prediction.value.read().unwrap()) * grad;
        prediction.add_grad(d_predcition);
    }

    // df/dactual = 2(actual - prediction) * grad
    if actual.requires_grad() {
        let d_actual =
            2.0 * (&*actual.value.read().unwrap() - &*prediction.value.read().unwrap()) * grad;
        actual.add_grad(d_actual);
    }
}
