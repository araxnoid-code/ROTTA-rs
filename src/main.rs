use rotta_rs::{ matmul, Tensor };

fn main() {
    let tensor_a = Tensor::new([[1.0, 2.0]]);
    println!("{}", tensor_a.able_update_grad());
    tensor_a.set_able_update_grad(false);
}
