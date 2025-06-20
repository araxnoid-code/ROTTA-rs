use std::{ collections::HashSet };

use crate::rotta_rs::{
    d_add,
    d_broadcasting_tensor,
    d_cel,
    d_cel_test,
    d_divided,
    d_dot,
    d_exp,
    d_ln,
    d_matmul,
    d_mul,
    d_powi,
    d_relu,
    d_ssresidual,
    d_sub,
    d_sum_axis,
    BackwardLabel,
    NodeType,
    Tensor,
};

impl Tensor {
    pub fn backward(&self) {
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
                    BackwardLabel::Dot(a, b) => d_dot(a, b, grad),
                    BackwardLabel::Matmul(a, b) => d_matmul(a, b, grad),
                    BackwardLabel::Add(a, b) => d_add(a, b, grad),
                    BackwardLabel::Diveded(a, b) => d_divided(a, b, &grad),
                    BackwardLabel::Mul(a, b) => d_mul(a, b, &grad),
                    BackwardLabel::Sub(a, b) => d_sub(a, b, &grad),

                    // mutation
                    BackwardLabel::Broadcasting(tensor_arr, broad_arr) =>
                        d_broadcasting_tensor(tensor_arr, broad_arr.clone(), grad),
                    BackwardLabel::SumAxis(x, d, keep_dim) => d_sum_axis(x, *d, *keep_dim, &grad),

                    // function
                    BackwardLabel::Exp(a, exp_value) => d_exp(a, exp_value, &grad),
                    BackwardLabel::Powi(x, powi) => d_powi(x, *powi, &grad),
                    BackwardLabel::Ln(x) => d_ln(x, &grad),

                    // activation
                    BackwardLabel::Relu(x) => d_relu(x, &grad),

                    // loss
                    BackwardLabel::SSResidual(prediction, actual) =>
                        d_ssresidual(prediction, actual, &grad),
                    BackwardLabel::CEL(prob_prediction, prob_actual) =>
                        d_cel_test(prob_prediction, prob_actual, &grad),
                }
            }
        }
    }
}

pub fn build(node_arc: NodeType, graph: &mut Vec<NodeType>, vitited: &mut HashSet<u128>) {
    let node = node_arc.lock().unwrap();
    if let None = vitited.get(&node.id) {
        vitited.insert(node.id);

        let parents = node.parent.clone();

        for parent in parents {
            build(parent, graph, vitited);
        }

        graph.push(node_arc.clone());
    }
    // let parent = node.lock().unwrap().parent.clone();
}
