use crate::rotta_rs::*;

pub fn transpose(arr: &Arrayy, d: (i32, i32)) -> Arrayy {
    let shape = arr.shape.clone();
    let dimension_t = [d.0, d.1]
        .iter()
        .map(|d| {
            if d >= &(shape.len() as i32) {
                panic!(
                    "range out of array, array is {} dimension but will dimension index is {}",
                    arr.shape.len(),
                    d
                );
            }

            if d < &0 {
                let d = (shape.len() as i32) + d;
                if d < 0 {
                    panic!(
                        "range out of array, array is {} dimension but will dimension index is {}",
                        arr.shape.len(),
                        d
                    );
                }
                d
            } else {
                *d
            }
        })
        .collect::<Vec<i32>>();

    let mut order = shape
        .iter()
        .enumerate()
        .map(|(idx, _)| { idx })
        .collect::<Vec<usize>>();

    // change
    order[dimension_t[0] as usize] = dimension_t[1] as usize;
    order[dimension_t[1] as usize] = dimension_t[0] as usize;

    permute(order, arr)
}
