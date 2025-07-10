use crate::rotta_rs_module::{ arrayy::{ broadcast_shape_slice, * } };

pub fn broadcast_concat(arr_a: &Arrayy, arr_b: &Arrayy) -> Vec<usize> {
    broadcast_shape_slice(&arr_a.shape, &arr_b.shape).unwrap()
}

pub fn broadcasting(arr: &Arrayy, broadcast_shape: Vec<usize>) -> Arrayy {
    let (vector, shape) = broadcasting_arr_slice((&arr.value, &arr.shape), &broadcast_shape);

    Arrayy::from_vector(shape, vector)
}

pub fn broadcasting_arr_test(arr: &Arrayy, broadcast_shape: Vec<usize>) {
    let value = &arr.value;
    let shape = if broadcast_shape.len() > arr.shape.len() {
        let distance = broadcast_shape.len() - arr.shape.len();
        let mut n_shape = vec![1;distance];
        n_shape.extend_from_slice(&arr.shape);
        n_shape
    } else if broadcast_shape.len() == arr.shape.len() {
        arr.shape.clone()
    } else {
        panic!("error, shape {:?} cant be broadcasting to {:?}", arr.shape, broadcast_shape)
    };

    let mut broadcasting_dimension = vec![];
    shape
        .iter()
        .enumerate()
        .for_each(|(idx, d)| {
            if d == &1 {
                // if broadcast_shape[idx] != 1 {
                broadcasting_dimension.push(idx);
                // }
            }
        });

    let mut output = vec![];
    for d_target in &broadcasting_dimension {
        let length = (&broadcast_shape[broadcasting_dimension[0] + 1..]).multiple_sum();
        let loping_count = (&broadcast_shape[..broadcasting_dimension[0]]).multiple_sum();

        for i in 0..loping_count {
            let start = i * length;
            let stop = start + length;

            let slice = &value[start..stop];

            for _ in 0..broadcast_shape[*d_target] {
                output.extend_from_slice(slice);
            }
        }
    }
}

// pub fn matmul_broadcasting(
//     arr: &Arrayy,
//     broadcast_shape: Vec<usize>
// ) -> Result<Arrayy, &'static str> {
//     // can broadcasting ?
//     let arr_shape = arr.shape.clone();
//     let mut b_shape = broadcast_shape.clone();

//     let extend = b_shape.len() - arr_shape.len();
//     let extended_arr_shape = &vec![vec![1;extend], arr_shape.to_vec()].concat()[..];
//     let mut broadcasting_target = vec![];

//     for i in 0..extended_arr_shape.len() - 2 {
//         if extended_arr_shape[i] == 1 || extended_arr_shape[i] == b_shape[i] {
//             if extended_arr_shape[i] != b_shape[i] {
//                 broadcasting_target.push(i);
//             }
//         } else {
//             return Err("can't broadcasting");
//         }
//     }

//     if broadcasting_target.len() <= 0 {
//         let mut arr = (*arr).clone();
//         arr.shape = extended_arr_shape.to_vec();

//         return Ok(arr);
//     }

//     let mut vector = arr.value.clone();
//     let mut out_vec = Vec::new();

//     for target in broadcasting_target {
//         let mut output = vec![];
//         let times = broadcast_shape[target];
//         let length = (&extended_arr_shape[target..]).multiple_sum();
//         let count = vector.len() / length;

//         for i in 0..count {
//             let start = i * length;
//             let stop = start + length;
//             let slice = &vector[start..stop];

//             for _ in 0..times {
//                 output.push(slice);
//             }
//         }

//         let concat = output.concat();
//         if out_vec.len() > 0 {
//             out_vec[0] = concat.clone();
//         } else {
//             out_vec.push(concat.clone());
//         }
//         vector = concat;
//     }

//     let len = b_shape.len();
//     b_shape[len - 1] = *arr_shape.last().unwrap();
//     b_shape[len - 2] = arr_shape[len - 2];

//     let output: Arrayy = Arrayy::from_vector(b_shape, out_vec[0].clone());
//     Ok(output)
// }
