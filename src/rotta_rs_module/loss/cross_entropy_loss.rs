use crate::{ rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor }, ShareTensor };

pub struct CrossEntropyLoss {}

impl CrossEntropyLoss {
    pub fn init() -> CrossEntropyLoss {
        CrossEntropyLoss {}
    }

    pub fn forward(&self, prob_prediction: &Tensor, prob_actual: &Tensor) -> Tensor {
        let epsilon = 1e-9;
        let prediction_shape = prob_prediction.shape();
        let actual_shape = prob_actual.shape();
        if prediction_shape.len() != 2 || actual_shape.len() != 1 {
            panic!(
                "prediction probability and actual probability for this cross entropy loss must be 2D(Batch, Class) for prediction and 1D(Batch) for actual\n
                your prediction is {}D and actual is {}D",
                prediction_shape.len(),
                actual_shape.len()
            );
        }

        let pred_arr = &*prob_prediction.value.read().unwrap();
        let actual_arr = &*prob_actual.value.read().unwrap();

        // prob_actual * log(1/prob_prediction)
        let mut loss_batch = Arrayy::new([0.0]);
        for batch in 0..prediction_shape[0] {
            let actual_class = actual_arr.index(vec![batch as i32]);
            let loss = (
                1.0 /
                (pred_arr.index(vec![batch as i32, actual_class.value[0] as i32]) + epsilon)
            ).ln();

            loss_batch = loss_batch + loss;
        }

        let mut tensor = Tensor::from_arrayy(loss_batch);
        tensor.update_parent(vec![prob_prediction.shared_tensor(), prob_actual.shared_tensor()]);
        tensor.update_label(
            Some(BackwardLabel::CEL(prob_prediction.shared_tensor(), prob_actual.shared_tensor()))
        );

        tensor
    }
}

pub fn d_cel(prob_prediction: &ShareTensor, prob_actual: &ShareTensor, grad: &Arrayy) {
    let epsilon = 1e-9;
    // let mut _prob_pred = prob_prediction.write().unwrap();
    // let prob_actual = prob_actual.write().unwrap();

    if prob_prediction.requires_grad() {
        for batch in 0..prob_prediction.value.read().unwrap().shape[0] {
            let actual_class = prob_actual.value
                .read()
                .unwrap()
                .index(vec![batch as i32]);
            let pred_value = prob_prediction.value
                .read()
                .unwrap()
                .index(vec![batch as i32, actual_class.value[0] as i32]);

            let d = -1.0 * (1.0 / (pred_value + epsilon)) * grad.index(vec![0]);
            // let mut d_pred = &mut *prob_prediction.grad.write().unwrap();
            prob_prediction.grad
                .write()
                .unwrap()
                .index_mut(vec![batch as i32, actual_class.value[0] as i32], d);
        }

        // prob_prediction.add_grad(d_pred);
    }
}
