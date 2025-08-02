use std::sync::{ Arc, Mutex };

use crate::{ rotta_rs_module::{ arrayy::Arrayy, Backward }, ShareTensor };

pub struct Sgd {
    parameters: Arc<Mutex<Vec<ShareTensor>>>,
    lr: Arrayy,
    pub auto_zero_grad_execute: bool,
}

impl Sgd {
    pub fn init(parameters: Arc<Mutex<Vec<ShareTensor>>>, lr: f64) -> Sgd {
        let lr = Arrayy::from_vector(vec![1], vec![lr]);
        Sgd {
            parameters,
            lr,
            auto_zero_grad_execute: true,
        }
    }

    // zero
    pub fn zero_grad(&self) {
        for node_type in self.parameters.lock().unwrap().iter() {
            node_type.zero_grad();
        }
    }

    // optimazer
    pub fn optim(&self) {
        for node_type in self.parameters.lock().unwrap().iter() {
            let node_mutex = node_type;
            let new = &node_mutex.value() - &self.lr * &node_mutex.grad();
            node_type.update_value(new);
        }
    }
}
