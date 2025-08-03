use rotta_rs::{ Adam, Module, SSResidual, VectorToTensor, MAE, MSE };

fn main() {
    let input = (0..20).collect::<Vec<i32>>().to_tensor().reshape(vec![-1, 1]);
    let label = &input * 2.0;
    input.set_requires_grad(false);
    label.set_requires_grad(false);

    let mut model = Module::init();
    let linear_1 = model.liniar_init(1, 8);
    let linear_2 = model.liniar_init(8, 1);

    let loss_fn = SSResidual::init();
    let mut optimizer = Adam::init(model.parameters(), 0.01);

    for epoch in 0..100 {
        let x = linear_1.forward(&input);
        let x = linear_2.forward(&x);

        let loss = loss_fn.forward(&x, &label);
        println!("epoch:{epoch} | loss => {loss}");
        optimizer.zero_grad();
        loss.backward();

        optimizer.optim();
    }
}
