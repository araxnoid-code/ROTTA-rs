use crate::rotta_rs_module::{ arrayy::{ Arrayy }, BackwardLabel, NodeType, Tensor };

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

        let pred_arr = prob_prediction.value();
        let actual_arr = prob_actual.value();

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

        let tensor = Tensor::from_arrayy(loss_batch);
        tensor.update_parent(vec![prob_prediction.node.clone(), prob_actual.node.clone()]);
        tensor.node.write().unwrap().label = Some(
            BackwardLabel::CEL(prob_prediction.node.clone(), prob_actual.node.clone())
        );

        tensor
    }
}

pub fn d_cel(prob_prediction: &NodeType, prob_actual: &NodeType, grad: &Arrayy) {
    let epsilon = 1e-9;
    let _prob_pred = prob_prediction.read().unwrap();
    let prob_actual = prob_actual.read().unwrap();

    if _prob_pred.requires_grad {
        let mut d_pred = _prob_pred.grad.clone();
        for batch in 0.._prob_pred.value.shape[0] {
            let actual_class = prob_actual.value.index(vec![batch as i32]);
            let pred_value = _prob_pred.value.index(
                vec![batch as i32, actual_class.value[0] as i32]
            );

            let d = -1.0 * (1.0 / (pred_value + epsilon)) * grad.index(vec![0]);
            d_pred.index_mut(vec![batch as i32, actual_class.value[0] as i32], d);
        }

        prob_prediction.write().unwrap().add_grad(d_pred);
    }
}
