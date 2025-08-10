# par datahandler
par data handler adalah kependekan dari paralel datahandler adalah fitur pada `DataHandler` yang memungkinkan setiap sample pada `Dataset` dapat dieksekusi secara `paralel`.

## pertama, set up Dataset dan DataHandler
```rust
use rotta_rs::{ DataHandler, Dataset, Tensor };

struct MyDataset {
    data: Vec<Tensor>,
    label: Vec<Tensor>,
}

impl MyDataset {
    pub fn init(data: Vec<Tensor>, label: Vec<Tensor>) -> MyDataset {
        MyDataset {
            data,
            label,
        }
    }
}

impl Dataset for MyDataset {
    fn get(&self, idx: usize) -> (Tensor, Tensor) {
        (self.data[idx].clone(), self.label[idx].clone())
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

fn main() {
    let data_a = Tensor::new([[1.0]]);
    let label_a = Tensor::new([[2.0]]);

    let data_b = Tensor::new([[3.0]]);
    let label_b = Tensor::new([[4.0]]);

    let data_list = vec![data_a, data_b];
    let label_list = vec![label_a, label_b];
    let dataset = MyDataset::init(data_list, label_list);
    let datahandler = DataHandler::init(dataset);
}
```

## make your model AI
```rust
impl MyModel {
    fn init(model: &mut Module) -> MyModel {
        MyModel {
            linear_1: model.liniar_init(1, 8),
            linear_2: model.liniar_init(8, 1),
        }
    }
}
```

lalu kita gabungkan dengan code kita sebelumnya
```rust
use rotta_rs::{ DataHandler, Dataset, Linear, Module, Tensor };

// 
struct MyDataset {
    data: Vec<Tensor>,
    label: Vec<Tensor>,
}

impl MyDataset {
    pub fn init(data: Vec<Tensor>, label: Vec<Tensor>) -> MyDataset {
        MyDataset {
            data,
            label,
        }
    }
}

impl Dataset for MyDataset {
    fn get(&self, idx: usize) -> (Tensor, Tensor) {
        (self.data[idx].clone(), self.label[idx].clone())
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Clone)]
struct MyModel {
    linear_1: Linear,
    linear_2: Linear,
    loss_fn: MSE,
}
// 

// 
impl MyModel {
    fn init(model: &mut Module) -> MyModel {
        MyModel {
            linear_1: model.liniar_init(1, 8),
            linear_2: model.liniar_init(8, 1),
        }
    }
}
// 

fn main() {
    let data_a = Tensor::new([[1.0]]);
    let label_a = Tensor::new([[2.0]]);

    let data_b = Tensor::new([[3.0]]);
    let label_b = Tensor::new([[4.0]]);

    let data_list = vec![data_a, data_b];
    let label_list = vec![label_a, label_b];
    let dataset = MyDataset::init(data_list, label_list);
    let datahandler = DataHandler::init(dataset);

    let mut model = Module::init();
    let my_model = MyModel::init(&mut model);
}

```

untuk bisa menggunakan `par datahandler`, maka kita harus menggunakan trait yang bernama `ParDataHandler`
```rust
impl MyModel {
    fn init(model: &mut Module) -> MyModel {
        MyModel {
            linear_1: model.liniar_init(1, 8),
            linear_2: model.liniar_init(8, 1),
        }
    }
}

impl ParDataHandler for MyModel {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, data: &Self::Input) -> Self::Output {
        let x = self.linear_1.forward(data);
        let x = self.linear_2.forward(&x);
        x
    }
}
```

lalu kita gunakan method `par_by_sample` pada DataHandler
```rust
// ...
let (loss, model) = datahandler.par_by_sample(my_model, 2, |(input, label), model| {
        let x = model.forward(input);

        let loss = model.loss_fn.forward(&x, &label);
        loss.backward();

        loss
    });
```

lalu kita bisa menambahkan training loop dan optimazer