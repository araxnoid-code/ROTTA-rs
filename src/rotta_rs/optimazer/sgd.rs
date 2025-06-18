use std::sync::{ Arc, Mutex };

use crate::rotta_rs::{ Arrayy, NodeType };

pub struct Sgd {
    parameters: Arc<Mutex<Vec<NodeType>>>,
    lr: Arrayy,
}

impl Sgd {
    pub fn init(parameters: Arc<Mutex<Vec<NodeType>>>, lr: f64) -> Sgd {
        let lr = Arrayy::from_vector(vec![1], vec![lr]);
        Sgd {
            parameters,
            lr,
        }
    }

    // zero
    pub fn zero_grad(&self) {
        for node_type in self.parameters.lock().unwrap().iter() {
            node_type.lock().as_mut().unwrap().zero_grad();
        }
    }

    // optimazer
    pub fn optim(&self) {
        for node_type in self.parameters.lock().unwrap().iter() {
            let mut node = node_type.lock();
            let node_mutex = node.as_mut().unwrap();
            let new = &node_mutex.value - &self.lr * &node_mutex.grad;
            node_mutex.update_value(new);
        }
    }
}
