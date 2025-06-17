use crate::rotta_rs::{
    add,
    dot,
    matmul,
    matmul_nd,
    transpose,
    Arrayy,
    Module,
    RecFlatten,
    SSResidual,
    Tensor,
};

mod rotta_rs;

fn main() {
    let mut model = Module::init();

    let a = Tensor::from_vector(vec![2, 1], vec![2.0, 3.0]);

    let linear = model.liniar_init(1, 16);
    let linear_2 = model.liniar_init(16, 1);

    let x = linear.forward(&a);
    let x = linear_2.forward(&x);
    // println!("{:?}", x.node.lock().unwrap().value.shape)
}
