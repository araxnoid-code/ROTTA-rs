# Module

```rust
fn main() {
    let model = Module::init();
}
```

- seed(default=42)
```rust
fn main() {
    let mut model = Module::init();
    model.update_seed(43);
}
```

- weight initialization
```rust
pub enum WeightInitialization {
    Random,
    He, // Default
    Glorot,
}
```

```rust
fn main() {
    let mut model = Module::init();
    model.update_initialization(WeightInitialization::Glorot);
}
```

- training & testing

``` rust
fn main() {
    let mut model = Module::init();

    // training phase
    model.train();

    // testing phase
    model.eval();
}
```

## function
- Linear
```rust
fn main() {
    let mut model = Module::init();

    let linear = model.liniar_init(1, 4);
    let tensor = Tensor::new([[1.0]]);

    let x = linear.forward(&tensor);
    println!("{}", x);  // [
                        //  [-0.8411698816909696, -0.11660485427898945, 0.6455877424124907, 1.7693457512474318]
                        // ]
}
```

- Dropout
``` rust
fn main() {
    let mut model = Module::init();

    let linear = model.liniar_init(1, 4);
    let mut dropout = model.dropout_init(0.3);

    let tensor = Tensor::new([[1.0]]);

    let x = linear.forward(&tensor);
    let x = dropout.forward(&x);
    println!("{}", x);  // [
                        // [-0.0, -0.0, 0.9222682034464154, 2.527636787496331]
                        // ]
```

- Batch Norm
```rust
fn main() {
    // shape [N, C]
    let input = Tensor::rand(vec![4, 3]);
    println!("{}", input);
    // [
    //  [0.6065541671620274, 0.3080776981920479, 0.6442049369623437]
    //  [0.3286115121198031, 0.8860886054036982, 0.4349016344093295]
    //  [0.4885809530499826, 0.5448906992464032, 0.22447906464152978]
    //  [0.28556563490467135, 0.4566659131112475, 0.7941596514251009]
    // ]


    let mut model = Module::init();
    // model.layer_norm_init(channel features, dimension of input)
    let mut layer_norm = model.batch_norm_init(3, 2);

    let x = layer_norm.forward(&input);
    println!("{}", x);
    // [
    //  [1.3982903314039918, -1.1347153120913502, 0.5567832702312544]
    //  [-0.7701690975800358, 1.5884301048478846, -0.4162310471312238]
    //  [0.47788418334459615, -0.01903353092271295, -1.3944486413252506]
    //  [-1.1060054171685523, -0.434681261833821, 1.2538964182252195]
    // ]

    // shape [N, C, H, W]
    let input = Tensor::rand(vec![4, 5, 5, 3]);
    println!("{}", input);
    // [
    //  [
    //   [
    //    [0.9191833609785265, 0.11467614044934482, 0.3651331432873097]
    //    [0.6648474314778041, 0.04482044260274243, 0.7386766027231303]
    //    [0.18314422145098108, 0.10546228843491523, 0.36767086816454]
    //    [0.05367972150858624, 0.4694582951670303, 0.10467588087428847]
    //    [0.8875706674875007, 0.7636830631119149, 0.4293851691610734]
    //   ]
    //   ...
    //  ]
    // ]

    let mut model = Module::init();
    // model.layer_norm_init(channel features, input dimension)
    let mut layer_norm = model.batch_norm_init(5, 4);

    let x = layer_norm.forward(&input);
    println!("{}", x)
    // [
    //  [
    //   [
    //    [1.6947388119259272, -1.013241142230079, -0.170200177828192]
    //    [0.8386413386719765, -1.2483761722522448, 1.0871511228760107]
    //    [-0.7827768454273689, -1.0442550671756168, -0.1616581686064696]
    //    [-1.2185557442024855, 0.1809594050761565, -1.0469021234799385]
    //    [1.5883301457435823, 1.1713231374076922, 0.04607283208415827]
    //   ]
    //   ...
    //  ]
    // ]
}
```

- Layer Norm
```rust
fn main() {
    // shape [N, C]
    let input = Tensor::rand(vec![4, 3]);
    println!("{}", input);
    // [
    //  [0.27915410797778306, 0.687663313413999, 0.07613583039765615]
    //  [0.7021005531023465, 0.9162028784314272, 0.025782995740667558]
    //  [0.061904664145641775, 0.8716712494130606, 0.6913401190715669]
    //  [0.850197252067536, 0.532785768919453, 0.8443179891588507]
    // ]


    let mut model = Module::init();
    // model.layer_norm_init([C])
    let mut layer_norm = model.layer_norm_init(&[3]);

    let x = layer_norm.forward(&input);
    println!("{}", x);
    // [
    //  [-0.2693444156052575, 1.3369991007076736, -1.0676546851024162]
    //  [0.4060001436632176, 0.9701890895357663, -1.376189233198984]
    //  [-1.382041040730688, 0.9507738587706503, 0.43126718196003766]
    //  [0.7268411085257209, -1.414027911860354, 0.6871868033346348]
    // ]

    // shape [N, C, H, W]
    let input = Tensor::rand(vec![4, 3, 5, 3]);
    println!("{}", input);
    // [
    //  [
    //   [
    //    [0.10809007732401366, 0.7239885456083965, 0.7865357050103905]
    //    [0.5188113038428019, 0.02123492865878185, 0.7642409577112972]
    //    [0.24701245204105693, 0.1983279039599467, 0.1462213439416118]
    //    [0.002647079178351275, 0.45989042875353336, 0.5287350101045764]
    //    [0.27693985343666416, 0.16741881390091773, 0.8149889018886429]
    //   ]
    //   ...
    //  ]
    // ]



    let mut model = Module::init();
    // model.layer_norm_init([C, H, W])
    let mut layer_norm = model.layer_norm_init(&[3, 5, 3]);

    let x = layer_norm.forward(&input);
    println!("{}", x)
    // [
    //  [
    //   [
    //    [-1.1929170089128815, 0.7692268391534215, 0.9684910529068765]
    //    [0.11556834928089157, -1.4696221874483777, 0.8974639217971595]
    //    [-0.7503348351511742, -0.905435215217602, -1.0714375215208956]
    //    [-1.52883979616951, -0.07214316249008668, 0.14718352644766916]
    //    [-0.6549914148296812, -1.0039061224072066, 1.0591379177513331]
    //   ]
    //   ...
    //  ]
    // ]

    // enable & disable learnable
    layer_norm.enable_learnable(); // default
    layer_norm.disable_learnable();
}
```

- lstm
```rust
fn main() {
    let mut model = Module::init();
    let lstm = model.lstm_init(6);

    let tensor = Tensor::arange(0..6)
        .to_shape(vec![1, 6])
        .collect();
    println!("{}", tensor);
    // [
    //  [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    // ]

    let cell_hidden = lstm.forward(&tensor, None); //(&Tensor, Option<LSTMCellHidden>)
    println!("cell:\n{}", cell_hidden.cell);
    // cell:
    // [
    //  [-0.35452980609595336, -0.5292895572406782, 0.7244741335616286, -0.4735260362646916, -0.9515427099660342, -0.2374937433097324]
    // ]

    println!("hidden:\n{}", cell_hidden.hidden)
    // hidden:
    // [
    //  [-0.23028978337822023, -0.0435225356725021, 0.36147911006385536, -0.09738843166346652, -0.705630337819917, -0.051854840060327305]
    // ]
}
```

```rust
fn main() {
    LSTMCellHidden {
        cell: Tensor::new([[0.0]]),
        hidden: Tensor::new([[0.0]]),
    };
}
```

- gru
```rust
fn main() {
    let mut model = Module::init();
    let lstm = model.gru_init(6);

    let tensor = Tensor::arange(0..6)
        .to_shape(vec![1, 6])
        .collect();
    println!("{}", tensor);
    // [
    //  [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    // ]


    let hidden = lstm.forward(&tensor, None); // (&Tensor, Option<Tensor>)
    println!("hidden:\n{}", hidden);
    // hidden:
    // [
    //  [0.2838769389632212, 0.30500443468104915, 0.03277033032455531, -0.4399917698902819, -0.9237180542375861, 0.3168019973639146]
    // ]
}
```