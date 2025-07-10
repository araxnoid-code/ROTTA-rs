use rotta_rs::{ arrayy::{ broadcasting_arr_test, Arrayy }, * };

fn main() {
    // shape [N, C]
    let input = Tensor::rand(vec![4, 3]);
    println!("{}", input);

    let mut model = Module::init();
    // model.layer_norm_init(channel features, dimension of input)
    let mut layer_norm = model.batch_norm_init(3, 2);

    let x = layer_norm.forward(&input);
    println!("{}", x);

    // shape [N, C, H, W]
    let input = Tensor::rand(vec![4, 5, 5, 3]);
    println!("{}", input);

    let mut model = Module::init();
    // model.layer_norm_init(channel features, input dimension)
    let mut layer_norm = model.batch_norm_init(5, 4);

    let x = layer_norm.forward(&input);
    println!("{}", x)
}
