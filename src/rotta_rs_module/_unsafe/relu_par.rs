use std::ops::Range;
use rayon::prelude::*;

use crate::{ arrayy::Arrayy, BackwardLabel, Tensor };

#[allow(dead_code)]
pub unsafe fn relu_par(x: &Tensor) -> Tensor {
    let arr = x.value();
    let value = &arr.value;
    let looping = std::thread::available_parallelism().unwrap().get();

    let mut len = value.len() as f64;
    let mut cpus = looping as f64;
    let mut start = 0;

    let thread_work = (0..cpus as usize)
        .into_iter()
        .map(|_| {
            let length = (len / cpus).ceil();
            len -= length;
            cpus -= 1.0;

            let stop = start + (length as usize);
            let range = start..stop;
            start = stop;
            range
        })
        .collect::<Vec<Range<usize>>>();

    let data = thread_work
        .into_par_iter()
        .enumerate()
        .map(|(_, range)| {
            // println!("{}..{}", range.start, range.end);
            let length = range.end - range.start;
            let mut output = Vec::with_capacity(length);
            for x in &value[range] {
                output.push(if *x >= 0.0 { *x } else { 0.0 });
            }

            output
        })
        .collect::<Vec<Vec<f64>>>()
        .concat();

    let output = Arrayy {
        value: data,
        shape: arr.shape.clone(),
    };

    let tensor = Tensor::from_arrayy(output);
    tensor.update_parent(vec![x.node.clone()]);
    tensor.node.lock().as_mut().unwrap().label = Some(BackwardLabel::Relu(x.node.clone()));

    tensor
}
