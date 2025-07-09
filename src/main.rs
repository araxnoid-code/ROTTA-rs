use rotta_rs::*;

fn main() {
    let mut model = Module::init();
    let mut layer_norm = model.layer_norm_init(&[3]);

    let input = Tensor::rand(vec![1, 3]);
    println!("{}", input);

    let x = layer_norm.forward(&input);
    println!("{}", x)
}
