use crate::rotta_rs::{ Arrayy, BackwardLabel, NodeType, Tensor };

pub struct CrossEntropyLoss {}

impl CrossEntropyLoss {
    pub fn init() -> CrossEntropyLoss {
        CrossEntropyLoss {}
    }

    pub fn forward(&self, prob_prediction: &Tensor, prob_actual: &Tensor) -> Tensor {
        let epsilon = 1e-9;
        let loss =
            (prob_actual.value() * (1.0 / (prob_prediction.value() + epsilon)).ln()).sum() /
            (prob_prediction.value().shape[0] as f64);

        let tensor = Tensor::from_vector(vec![1], vec![loss]);
        tensor.update_parent(vec![prob_prediction.node.clone(), prob_actual.node.clone()]);
        tensor.node.lock().as_mut().unwrap().label = Some(
            BackwardLabel::CEL(prob_prediction.node.clone(), prob_actual.node.clone())
        );

        tensor
    }

    pub fn test_forward(&self, prediction: &Tensor, actual: &Tensor) {
        let prediction_shape = prediction.shape();
        let actual_shape = actual.shape();
        if prediction_shape.len() != 2 || actual_shape.len() != 1 {
            panic!(
                "prediction probability and actual probability for this cross entropy loss must be 2D(Batch, Class) for prediction and 1D(Batch) for actual\n
                your prediction is {}D and actual is {}D",
                prediction_shape.len(),
                actual_shape.len()
            );
        }

        let pred_arr = prediction.value();
        let actual_arr = actual.value();

        // prob_actual * log(1/prob_prediction)
        let mut loss_batch = Arrayy::new([0.0]);
        for batch in 0..prediction_shape[0] {
            let actual_class = actual_arr.index([batch].as_slice());
            let loss = (1.0 / pred_arr.index([batch, actual_class as usize].as_slice())).ln();

            loss_batch = loss_batch + loss;
        }

        println!("{}", loss_batch)
    }
}

pub fn d_cel(prob_prediction: &NodeType, prob_actual: &NodeType, grad: &Arrayy) {
    // df/dprediction = -prob_actual/prob_prediction
    let epsilon = 1e-9;
    let prob_pred = prob_prediction.lock().unwrap().value.clone();
    let prob_actual = prob_actual.lock().unwrap().value.clone();

    let d_pred = -1.0 * (prob_actual / (prob_pred + epsilon)) * grad;
    prob_prediction.lock().as_mut().unwrap().add_grad(d_pred);
}
