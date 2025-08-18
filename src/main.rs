use rotta_rs::{ Linear, Module };

struct MyModel {
    linear_a: Linear,
    linear_b: Linear,
}

impl MyModel {
    pub fn init(module: &mut Module) -> MyModel {
        Self {
            linear_a: module.liniar_init(1, 8),
            linear_b: module.liniar_init(8, 1),
        }
    }
}

fn main() {
    let mut model = Module::init();
    let my_model = MyModel::init(&mut model);
    model.save("parameters.json");
    // or
    model.load_save("parameters.json");
}
