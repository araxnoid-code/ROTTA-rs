use std::{ collections::HashSet, sync::{ Arc, Mutex } };

use crate::rotta_rs_module::{
    d_abs,
    d_add,
    d_broadcasting_tensor,
    d_cel,
    d_divided,
    d_dot,
    d_exp,
    d_index,
    d_ln,
    d_matmul,
    d_mul,
    d_permute,
    d_powf,
    d_powi,
    d_relu,
    d_sign,
    d_slice,
    d_ssresidual,
    d_sub,
    d_sum,
    d_sum_axis,
    d_to_shape,
    BackwardLabel,
    NodeType,
    Tensor,
};

pub struct Backward {
    map: Arc<Mutex<Vec<NodeType>>>,
}

impl Backward {
    pub fn zero_grad(&self) {
        for node in self.map.lock().unwrap().iter() {
            let mut node = node.lock().unwrap();
            if node.auto_zero_grad {
                node.zero_grad();
            }
        }
    }
}

impl Tensor {
    pub fn backward(&self) -> Backward {
        let node = self.node.clone();
        node.lock().as_mut().unwrap().ones_grad();

        let mut graph: Vec<NodeType> = vec![];
        let mut visited: HashSet<u128> = HashSet::new();

        build(node, &mut graph, &mut visited);

        for idx in (0..graph.len()).rev() {
            let node_arc = graph[idx].clone();
            let node = node_arc.lock().unwrap();
            let grad = node.grad.clone();

            if let Some(label) = &node.label {
                match label {
                    // opearation
                    BackwardLabel::Dot(a, b) => d_dot(a, b, &grad),
                    BackwardLabel::Matmul(a, b) => d_matmul(a, b, &grad),
                    BackwardLabel::Add(a, b) => d_add(a, b, grad),
                    BackwardLabel::Diveded(a, b) => d_divided(a, b, &grad),
                    BackwardLabel::Mul(a, b) => d_mul(a, b, &grad),
                    BackwardLabel::Sub(a, b) => d_sub(a, b, &grad),

                    // mutation
                    BackwardLabel::Index(x, index) => d_index(x, index.clone(), &grad),
                    BackwardLabel::Broadcasting(tensor_arr, broad_arr) =>
                        d_broadcasting_tensor(tensor_arr, broad_arr.clone(), grad),
                    BackwardLabel::SumAxis(x, d, keep_dim) => d_sum_axis(x, *d, *keep_dim, &grad),
                    BackwardLabel::Sum(x) => d_sum(x, &grad),
                    BackwardLabel::Permute(x, order) => d_permute(x, order.clone(), &grad),
                    BackwardLabel::Slice(x, range) => d_slice(x, range.clone(), &grad),
                    BackwardLabel::ToShape(x, to_shape) => d_to_shape(x, to_shape.clone(), &grad),

                    // function
                    BackwardLabel::Exp(a, exp_value) => d_exp(a, exp_value, &grad),
                    BackwardLabel::Powi(x, powi) => d_powi(x, *powi, &grad),
                    BackwardLabel::Powf(x, powf) => d_powf(x, *powf, &grad),
                    BackwardLabel::Ln(x) => d_ln(x, &grad),
                    BackwardLabel::Abs(x) => d_abs(x, &grad),
                    BackwardLabel::Sign(x) => d_sign(x),

                    // activation
                    BackwardLabel::Relu(x) => d_relu(x, &grad),

                    // loss
                    BackwardLabel::SSResidual(prediction, actual) =>
                        d_ssresidual(prediction, actual, &grad),
                    BackwardLabel::CEL(prob_prediction, prob_actual) =>
                        d_cel(prob_prediction, prob_actual, &grad),

                    _ => (),
                }
            }
        }

        let backward = Backward {
            map: Arc::new(Mutex::new(graph)),
        };

        backward
    }
}

pub fn build(node_arc: NodeType, graph: &mut Vec<NodeType>, vitited: &mut HashSet<u128>) {
    let node = node_arc.lock().unwrap();
    if node.requires_grad {
        if let None = vitited.get(&node.id) {
            vitited.insert(node.id);

            let parents = node.parent.clone();

            for parent in parents {
                build(parent, graph, vitited);
            }

            graph.push(node_arc.clone());
        }
    }
}
