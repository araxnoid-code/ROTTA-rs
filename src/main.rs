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
    // let a = array![[1.0, 2.0, 3.0]];
    // let tensor_a = Tensor::new(a);

    // let b = array![[1.0, 2.0, 3.0]];
    // let tensor_b = Tensor::new(b);

    // let c = add(&tensor_a, &tensor_b);
    // let exp = c.exp();
    // exp.backward();
    // println!("{}", c.grad());

    let root = BitMapBackend::new("testing.png", (640, 500)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let root = root.margin(10, 10, 10, 10);
    let mut chart = ChartBuilder::on(&root);

    let mut chart = chart
        .caption("testing", ("sans-serif", 40).into_font())
        .x_label_area_size(20)
        .y_label_area_size(40)
        .build_cartesian_2d(0f32..10f32, 0f32..10f32)
        .unwrap();

    chart
        .configure_mesh()
        // We can customize the maximum number of labels allowed for each axis
        // .x_labels(10)
        // .y_labels(10)
        // We can also change the format of the label text
        .y_label_formatter(&(|x| format!("{:.3}", x)))
        .draw()
        .unwrap();

    chart
        .draw_series(
            PointSeries::of_element(
                vec![(0.0, 5.0), (1.0, 5.0), (3.0, 10.0)],
                5,
                &RED,
                &(|c, s, st| {
                    return EmptyElement::at(c) + Circle::new((0, 0), s, st.filled());
                })
            )
        )
        .unwrap();

    // chart.dra
}
