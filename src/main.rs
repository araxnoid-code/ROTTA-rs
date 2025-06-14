use ndarray::array;
use plotters::{
    chart::ChartBuilder,
    prelude::{ BitMapBackend, IntoDrawingArea },
    style::{ IntoFont, WHITE },
    prelude::*,
};

use crate::rotta_rs::{ add, relu, softmax, CrossEntropyLoss, Module, SSResidual, Sgd, Tensor };

mod rotta_rs;

fn main() {
    let input_train = Tensor::new(
        array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
    );
    let label_train = Tensor::new(
        array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]] * 2.0
    );

    let input_test = Tensor::new(array![[11.0], [12.0], [13.0], [14.0], [15.0]]);
    let label_test = Tensor::new(array![[11.0], [12.0], [13.0], [14.0], [15.0]] * 2.0);

    let mut model = Module::init();
    let optimazer = Sgd::init(model.parameters(), 0.0001);
    let loss_fn = SSResidual::init();

    //
    let linear = model.liniar_init(1, 1);
    //

    for epoch in 0..100 {
        let prediction = linear.forward(&input_train);
        prediction.backward();
        println!("{}", label_train);
        println!("{}", prediction);
        let loss = loss_fn.forward(&prediction, &label_train);
        println!("epoch:{} => loss:{}", epoch, loss);
        optimazer.zero_grad();
        loss.backward();
        optimazer.optim();
    }

    // testing
    let prediction = linear.forward(&input_test);

    plt(&input_train, &label_train, &input_test, &label_test, Some(prediction));
}

fn plt(
    input_train: &Tensor,
    label_train: &Tensor,
    input_test: &Tensor,
    label_test: &Tensor,
    prediction: Option<Tensor>
) {
    let root = BitMapBackend::new("testing_data.png", (728, 728)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    root.margin(10, 10, 10, 10);

    let mut chart = ChartBuilder::on(&root);

    let chart = chart.caption("data", ("", 10).into_font());
    chart.y_label_area_size(25);
    chart.x_label_area_size(25);

    let mut chart = chart.build_cartesian_2d(0f32..50f32, 0f32..50f32).unwrap();
    chart.configure_mesh().draw().unwrap();

    let input_train_vector = input_train.node.lock().unwrap().value.flatten().to_vec();
    let label_train_vector = label_train.node.lock().unwrap().value.flatten().to_vec();

    let concat_train_data = input_train_vector
        .iter()
        .enumerate()
        .map(|(i, v)| { (*v as f32, label_train_vector[i] as f32) })
        .collect::<Vec<(f32, f32)>>();

    let input_test_vector = input_test.node.lock().unwrap().value.flatten().to_vec();
    let label_test_vector = label_test.node.lock().unwrap().value.flatten().to_vec();
    let concat_testing_data = input_test_vector
        .iter()
        .enumerate()
        .map(|(i, v)| { (*v as f32, label_test_vector[i] as f32) })
        .collect::<Vec<(f32, f32)>>();

    chart
        .draw_series(
            PointSeries::of_element(
                concat_train_data,
                5,
                &BLUE,
                &(|a, b, c| {
                    return EmptyElement::at(a) + Circle::new((0, 0), b, c.filled());
                })
            )
        )
        .unwrap();

    chart
        .draw_series(
            PointSeries::of_element(
                concat_testing_data,
                5,
                &RED,
                &(|a, b, c| {
                    return EmptyElement::at(a) + Circle::new((0, 0), b, c.filled());
                })
            )
        )
        .unwrap();

    if let Some(tensor) = prediction {
        let prediction_vector = tensor.node.lock().unwrap().value.flatten().to_vec();

        let predcition_concat = prediction_vector
            .iter()
            .enumerate()
            .map(|(i, v)| (input_test_vector[i] as f32, *v as f32))
            .collect::<Vec<(f32, f32)>>();

        chart
            .draw_series(
                PointSeries::of_element(
                    predcition_concat,
                    5,
                    &GREEN,
                    &(|a, b, c| {
                        return EmptyElement::at(a) + Circle::new((0, 0), b, c.filled());
                    })
                )
            )
            .unwrap();
    }
}
