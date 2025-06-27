use std::sync::{ Arc, Mutex };

use crate::rotta_rs::{ Arrayy, Backward, NodeType };

pub struct SgdMomen {
    parameters: Arc<Mutex<Vec<NodeType>>>,
    lr: Arrayy,
    v: Vec<Arrayy>,
    g: f64,
    pub auto_zero_grad_execute: bool,
}

impl SgdMomen {
    pub fn init(parameters: Arc<Mutex<Vec<NodeType>>>, lr: f64) -> SgdMomen {
        let lr = Arrayy::from_vector(vec![1], vec![lr]);
        SgdMomen {
            parameters,
            lr,
            v: vec![],
            g: 0.9,
            auto_zero_grad_execute: true,
        }
    }

    // zero
    pub fn zero_grad(&self) {
        for node_type in self.parameters.lock().unwrap().iter() {
            node_type.lock().as_mut().unwrap().zero_grad();
        }
    }

    // optimazer
    pub fn optim(&mut self, backward: Backward) {
        // v initialization
        if self.v.is_empty() {
            for node_type in self.parameters.lock().unwrap().iter() {
                let arr = Arrayy::arrayy_from_element(
                    node_type.lock().unwrap().value.shape.clone(),
                    0.0
                );

                self.v.push(arr);
            }
        }

        for (i, node_type) in self.parameters.lock().unwrap().iter().enumerate() {
            let mut node = node_type.lock().unwrap();

            // v = g * v + lr * grad(w)
            // w = w -  v
            let v = self.g * &self.v[i] + &self.lr * &node.grad;
            let new = &node.value - &v;
            node.update_value(new);

            // update v
            self.v[i] = v;
        }

        // auto_grad_zero
        if self.auto_zero_grad_execute {
            backward.zero_grad();
        }
    }

    // update
    pub fn update_hyperparameter(&mut self, g: f64) {
        self.g = g;
    }
}
