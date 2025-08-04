use crate::rotta_rs_module::arrayy::*;

pub fn add_arr_slice(
    arr_a: (&[f32], &[usize]),
    arr_b: (&[f32], &[usize])
) -> (Vec<f32>, Vec<usize>) {
    let (arr_a, shape_a) = arr_a;
    let (arr_b, shape_b) = arr_b;

    if shape_a == shape_b {
        // same shape

        let vector = arr_a
            .iter()
            .zip(arr_b.iter())
            .map(|(a, b)| { a + b })
            .collect::<Vec<f32>>();

        // let vector = arr_a
        //     .iter()
        //     .enumerate()
        //     .map(|(i, a)| {
        //         // add
        //         a + arr_b[i]
        //     })
        //     .collect::<Vec<f32>>();
        (vector, shape_a.to_vec())
    } else if shape_a.multiple_sum() == 1 || shape_b.multiple_sum() == 1 {
        // skalar

        let broadcasting_shape = broadcast_shape_slice(shape_a, shape_b).unwrap();

        if shape_a.multiple_sum() == 1 {
            let skalar = arr_a[0];

            let vector = arr_b
                .iter()
                .map(|v| skalar + *v)
                .collect::<Vec<f32>>();

            (vector, broadcasting_shape)
        } else {
            let skalar = arr_b[0];

            let vector = arr_a
                .iter()
                .map(|v| *v + skalar)
                .collect::<Vec<f32>>();

            (vector, broadcasting_shape)
        }
    } else {
        // broadcasting
        let broadcast_shape = broadcast_shape_slice(shape_a, shape_b).unwrap();

        let (arr_a, _) = broadcasting_arr_slice((arr_a, shape_a), &broadcast_shape);
        let (arr_b, _) = broadcasting_arr_slice((arr_b, shape_b), &broadcast_shape);

        let vector = arr_a
            .iter()
            .enumerate()
            .map(|(i, a)| {
                // add
                a + arr_b[i]
            })
            .collect::<Vec<f32>>();
        (vector, broadcast_shape)
    }
}
