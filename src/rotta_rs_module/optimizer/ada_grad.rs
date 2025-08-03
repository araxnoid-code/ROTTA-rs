use std::sync::{ Arc, Mutex };

use crate::{ rotta_rs_module::{ arrayy::Arrayy, Backward, NodeType }, ShareTensor };

pub struct AdaGrad {
    parameters: Arc<Mutex<Vec<ShareTensor>>>,
    pub lr: Arrayy,
    pub g: Vec<Arrayy>,
    pub eps: f64,
    pub auto_zero_grad_execute: bool,
}

impl AdaGrad {
    pub fn init(parameters: Arc<Mutex<Vec<ShareTensor>>>, lr: f64) -> AdaGrad {
        let lr = Arrayy::from_vector(vec![1], vec![lr]);
        AdaGrad {
            parameters,
            lr,
            g: vec![],
            eps: 1e-8,
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
            let _node = node_type;
            if let None = self.g.get(i) {
                self.g.push(Arrayy::arrayy_from_element(_node.value().shape.clone(), 0.0));
            }

            // ada grad
            // g_n = g_n - 1 + g_n^2
            // w_n + 1 = w_n - (lr/(g_n^0.5 + e).sqrt) * grad(w_n)

            let eps = self.eps;
            let grad = &_node.grad;
            let g_n = &self.g[i] + grad.read().unwrap().powi(2);
            let new = &_node.value() - (&self.lr / (g_n.powf(0.5) + eps)) * &*grad.read().unwrap();

            node_type.update_value(new);
            // update g
            self.g[i] = g_n;
        }
    }
}
