use std::{ collections::HashSet, sync::{ Arc, Mutex, RwLock } };

use crate::{ d_concat, rotta_rs_module::*, sin_cos_tan::{ d_cos, d_sin, d_tan } };

pub struct Backward {
    pub map: Arc<Mutex<Vec<NodeType>>>,
}

impl Backward {
    pub fn zero_grad(&self) {
        for node in self.map.lock().unwrap().iter() {
            let mut node = node.write().unwrap();
            if node.auto_zero_grad {
                node.zero_grad();
            }
        }
    }
}

impl Tensor {
    pub fn backward(&self) {
        self.ones_grad();

        let mut graph: Vec<ShareTensor> = vec![];
        let mut visited: HashSet<u128> = HashSet::new();
        let shared_tensor = Arc::new(RwLock::new(self.clone()));
        build(shared_tensor, &mut graph, &mut visited);

        for idx in (0..graph.len()).rev() {
            let node_arc = graph[idx].clone();
            let node = node_arc.read().unwrap();
            let grad = node.grad.clone();

            if let Some(label) = &node.label {
                match label {
                    _ => (),
                    // opearation
                    // BackwardLabel::Dot(a, b) => d_dot(a, b, &grad),
                    // BackwardLabel::Matmul(a, b) => d_matmul(a, b, &grad),
                    // BackwardLabel::Add(a, b) => d_add(a, b, grad),
                    // BackwardLabel::Diveded(a, b) => d_divided(a, b, &grad),
                    // BackwardLabel::Mul(a, b) => d_mul(a, b, &grad),
                    // BackwardLabel::Sub(a, b) => d_sub(a, b, &grad),

                    // mutation
                    // BackwardLabel::Index(x, index) => d_index(x, index.clone(), &grad),
                    // BackwardLabel::Broadcasting(tensor_arr, broad_arr) =>
                    // d_broadcasting_tensor(tensor_arr, broad_arr.clone(), grad),
                    // BackwardLabel::SumAxis(x, d, keep_dim) => d_sum_axis(x, d, *keep_dim, &grad),
                    // BackwardLabel::Sum(x) => d_sum(x, &grad),
                    // BackwardLabel::Permute(x, order) => d_permute(x, order.clone(), &grad),
                    // BackwardLabel::Slice(x, range) => d_slice(x, range.clone(), &grad),
                    // BackwardLabel::ToShape(x, to_shape) => d_to_shape(x, to_shape.clone(), &grad),
                    // BackwardLabel::Concat(nodes, dim) => d_concat(nodes.clone(), *dim, &grad),

                    // function
                    // BackwardLabel::Exp(a, exp_value) => d_exp(a, exp_value, &grad),
                    // BackwardLabel::Powi(x, powi) => d_powi(x, *powi, &grad),
                    // BackwardLabel::Powf(x, powf) => d_powf(x, *powf, &grad),
                    // BackwardLabel::Ln(x) => d_ln(x, &grad),
                    // BackwardLabel::Abs(x) => d_abs(x, &grad),
                    // BackwardLabel::Sign(x) => d_sign(x),
                    // BackwardLabel::Sin(x) => d_sin(x, &grad),
                    // BackwardLabel::Cos(x) => d_cos(x, &grad),
                    // BackwardLabel::Tan(x) => d_tan(x, &grad),

                    // activation
                    // BackwardLabel::Relu(x) => d_relu(x, &grad),

                    // loss
                    // BackwardLabel::SSResidual(prediction, actual) =>
                    //     d_ssresidual(prediction, actual, &grad),
                    // BackwardLabel::CEL(prob_prediction, prob_actual) =>
                    //     d_cel(prob_prediction, prob_actual, &grad),
                }
            }
        }

        // let backward = Backward {
        //     map: Arc::new(Mutex::new(graph)),
        // };

        // backward
    }
}

pub fn build(tensor: ShareTensor, graph: &mut Vec<ShareTensor>, vitited: &mut HashSet<u128>) {
    let node = tensor.read().unwrap();
    if node.requires_grad() {
        if let None = vitited.get(&node.id) {
            vitited.insert(node.id);

            let parents = &node.parent;

            for parent in parents {
                build(parent.clone(), graph, vitited);
            }

            graph.push(tensor.clone());
        }
    }
}
