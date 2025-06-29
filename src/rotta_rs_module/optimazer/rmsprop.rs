use std::sync::{ Arc, Mutex };

use crate::rotta_rs_module::{ arrayy::Arrayy, Backward, NodeType };

pub struct RMSprop {
    parameters: Arc<Mutex<Vec<NodeType>>>,
    lr: Arrayy,
    g: Vec<Arrayy>,
    hyperparameter: f64,
    pub auto_zero_grad_execute: bool,
}

impl RMSprop {
    pub fn init(parameters: Arc<Mutex<Vec<NodeType>>>, lr: f64) -> RMSprop {
        let lr = Arrayy::from_vector(vec![1], vec![lr]);
        RMSprop {
            parameters,
            lr,
            g: vec![],
            hyperparameter: 0.9,
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
        for (i, node_type) in self.parameters.lock().unwrap().iter().enumerate() {
            let mut node = node_type.lock().unwrap();
            if let None = self.g.get(i) {
                self.g.push(Arrayy::arrayy_from_element(node.value.shape.clone(), 0.0));
            }

            // g_n = g_n-1 * hyperparameter + (1 - hyperparameter) * grad(w_n)^2
            // w_n + 1 = w_n - (lr/((g_n)^0.5 + e)) * grad(w_n)

            let eps = 1e-8;
            let grad = &node.grad;
            let g_n = &self.g[i] * self.hyperparameter + (1.0 - self.hyperparameter) * grad.powi(2);
            let new = &node.value - (&self.lr / (g_n.powf(0.5) + eps)) * grad;
            node.update_value(new);

            self.g[i] = g_n;
        }

        // auto_grad_zero
        if self.auto_zero_grad_execute {
            backward.zero_grad();
        }
    }

    pub fn update_hyperparameter(&mut self, parameter: f64) {
        self.hyperparameter = parameter;
    }
}
