use std::sync::{ Arc, Mutex };

use crate::{ rotta_rs_module::{ arrayy::Arrayy }, ShareTensor };

pub struct SgdMomen {
    parameters: Arc<Mutex<Vec<ShareTensor>>>,
    lr: Arrayy,
    v: Vec<Arrayy>,
    g: f32,
    pub auto_zero_grad_execute: bool,
}

impl SgdMomen {
    pub fn init(parameters: Arc<Mutex<Vec<ShareTensor>>>, lr: f32) -> SgdMomen {
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
            node_type.zero_grad();
        }
    }

    // optimazer
    pub fn optim(&mut self) {
        for (i, node_type) in self.parameters.lock().unwrap().iter().enumerate() {
            let node = node_type;

            // v initialization
            if let None = self.v.get(i) {
                self.v.push(Arrayy::arrayy_from_element(node.shape(), 0.0));
            }

            // v = g * v + lr * grad(w)
            // w = w -  v
            let v = self.g * &self.v[i] + &self.lr * &*node.grad.read().unwrap();
            let new = &*node.value.read().unwrap() - &v;
            node_type.update_value(new);

            // update v
            self.v[i] = v;
        }
    }

    // update
    pub fn update_hyperparameter(&mut self, g: f32) {
        self.g = g;
    }
}
