use std::{ clone, collections::HashSet };

use crate::rotta_rs::{ d_add, d_matmul, d_ssresidual, BackwardLabel, NodeType, Tensor };

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
                    BackwardLabel::Matmul(a, b) => d_matmul(a, b, &grad),
                    BackwardLabel::Add(a, b) => d_add(a, b, &grad),

                    // loss
                    BackwardLabel::SSResidual(prediction, actual) =>
                        d_ssresidual(prediction, actual, &grad),
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
