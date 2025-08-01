use std::sync::{ Arc, Mutex };

use crate::rotta_rs_module::{ arrayy::Arrayy, Backward, NodeType };

pub struct Adam {
    parameters: Arc<Mutex<Vec<NodeType>>>,
    pub lr: Arrayy,
    g: Vec<Arrayy>,
    m: Vec<Arrayy>,
    i: i32,
    pub eps: f64,
    pub hyperparameter_1: f64,
    pub hyperparameter_2: f64,
    pub auto_zero_grad_execute: bool,
}

impl Adam {
    pub fn init(parameters: Arc<Mutex<Vec<NodeType>>>, lr: f64) -> Adam {
        let lr = Arrayy::from_vector(vec![1], vec![lr]);
        Adam {
            parameters,
            lr,
            g: vec![],
            m: vec![],
            i: 1,
            eps: 1e-8,
            hyperparameter_1: 0.9,
            hyperparameter_2: 0.999,
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
                self.m.push(Arrayy::arrayy_from_element(node.value.shape.clone(), 0.0));
            }

            // g_n = parameter_1 * g_n-1 + (1 - parameter_1) * grad(w_n)^2
            // m_n = parameter_2 * g_n-1 + (1 - parameter_2) * grad(w_n)

            // bias correction
            // gh_n = g_n/(1 - parameter_1^i)
            // mh_n = m_n/(1 - parameter_2^i)

            // w_n+1 = w_n - 1/(gh_n^0.5 + e) * mh_n

            let eps = self.eps;
            let grad = &node.grad;
            let m_n = self.hyperparameter_1 * &self.m[i] + (1.0 - &self.hyperparameter_1) * grad;
            let g_n =
                self.hyperparameter_2 * &self.g[i] + (1.0 - &self.hyperparameter_2) * grad.powi(2);

            // bias correction
            let mh_n = &m_n / (1.0 - self.hyperparameter_1.powi(self.i));
            let gh_n = &g_n / (1.0 - self.hyperparameter_2.powi(self.i));

            let new = &node.value - (&self.lr / (gh_n.powf(0.5) + eps)) * mh_n;
            node.update_value(new);

            self.g[i] = g_n;
            self.m[i] = m_n;
        }
        self.i += 1;

        // auto_grad_zero
        if self.auto_zero_grad_execute {
            backward.zero_grad();
        }
    }

    pub fn update_hyperparameter(&mut self, parameter_1: f64, parameter_2: f64) {
        self.hyperparameter_1 = parameter_1;
        self.hyperparameter_2 = parameter_2;
    }
}
