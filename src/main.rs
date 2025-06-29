use rotta_rs::*;

fn main() {
    let mut model = Module::init();
    let mut optimazer = RMSprop::init(model.parameters(), 0.001);
    let loss_fn = SSResidual::init();

    let linear = model.liniar_init(1, 128);
    let linear_2 = model.liniar_init(128, 1);

    let input = Tensor::new([[1.0], [2.0], [3.0], [4.0]]);
    let actual = Tensor::new([[10.0], [20.0], [30.0], [40.0]]);

    for epoch in 0..1000 {
        let x = linear.forward(&input);
        let x = relu(&x);
        let output = linear_2.forward(&x);

        let loss = loss_fn.forward(&output, &actual);
        println!("epoch:{epoch} | loss => {loss}");

        optimazer.zero_grad();

        let backward = loss.backward();

        optimazer.optim(backward);
    }
}
