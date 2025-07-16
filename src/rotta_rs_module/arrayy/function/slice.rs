use std::{ ops::Range, collections::VecDeque };

use crate::rotta_rs_module::arrayy::*;

#[derive(Debug, Clone)]
pub struct ArrSlice(pub Option<i32>, pub Option<i32>);

#[derive(Debug, Clone)]
struct ArrRange(pub Option<usize>, pub Option<usize>);

pub fn slice_arr_unsafe(arr: &Arrayy, slice: &[ArrSlice]) -> Arrayy {
    let arr_shape = &arr.shape;
    let ranges = slice
        .iter()
        .enumerate()
        .map(|(idx, arr_slice)| {
            let start = if arr_slice.0.unwrap_or(0) >= 0 {
                arr_slice.0.unwrap_or(0)
            } else {
                (arr_shape[idx] as i32) + arr_slice.0.unwrap()
            };

            let stop = if arr_slice.1.unwrap_or(arr_shape[idx] as i32) >= 0 {
                arr_slice.1.unwrap_or(arr_shape[idx] as i32)
            } else {
                (arr_shape[idx] as i32) + arr_slice.1.unwrap() + 1
            };

            start as usize..stop as usize
        })
        .collect::<Vec<Range<usize>>>();

    // let mut save = vec![(&arr.value[..], 0)];
    let mut save_q = VecDeque::new();
    save_q.push_front((&arr.value[..], 0));

    let mut output = vec![];
    let mut shape = arr_shape.clone();

    while save_q.len() > 0 {
        let saved_value = save_q.pop_back().unwrap();
        let arr = saved_value.0;
        let d = saved_value.1;
        let length = (&arr_shape[d + 1..]).multiple_sum();
        let range = &ranges[d];

        for i in range.start..range.end {
            let start = i * length;
            let stop = start + length;
            shape[d] = range.end - range.start;

            let slice = &arr[start..stop];
            // println!("{}", d);

            if d == ranges.len() - 1 {
                output.extend_from_slice(slice);
            } else {
                save_q.push_front((slice, d + 1));
            }
        }
    }

    let arr = Arrayy::from_vector(shape, output);
    arr
}

pub fn slice_arr(arr: &Arrayy, slice: Vec<ArrSlice>) -> Arrayy {
    let mut index = Vec::new();
    let mut d = 0;
    let mut shape = arr.shape.clone();

    let slice = slice
        .into_iter()
        .enumerate()
        .map(|(idx, slice)| {
            let mut start = slice.0.unwrap_or(0);
            if start < 0 {
                start = (shape[idx] as i32) + start;

                if start < 0 {
                    panic!(
                        "slice out of shape, array shape is {:?} but will slice by {:?}",
                        shape,
                        slice
                    );
                }
            }

            let mut stop = slice.1.unwrap_or(shape[idx] as i32);
            if stop < 0 {
                stop = (shape[idx] as i32) + stop + 1;

                if stop < 0 {
                    panic!(
                        "slice out of shape, array shape is {:?} but will slice by {:?}",
                        shape,
                        slice
                    );
                }
            }

            ArrRange(Some(start as usize), Some(stop as usize))
        })
        .collect::<Vec<ArrRange>>();

    let mut result = vec![];
    loop {
        if d == slice.len() {
            let i = d - 1;
            //
            let mut n = true;
            for (i, range) in slice.iter().enumerate() {
                let s = range.0.unwrap_or(0) as usize;
                let e = range.1.unwrap_or(arr.shape[i]);
                if shape[i] != e - s {
                    shape[i] = e - s;
                }

                if index[i] >= s && index[i] < e {
                } else {
                    n = false;
                    break;
                }
            }

            if n {
                let slice = slice_indexs(&arr.value, &arr.shape, &index);
                result.extend_from_slice(slice);
            }

            index[i] += 1;
            if index[i] > arr.shape[i] - 1 {
                index.pop();

                if d == 1 {
                    break;
                }
                d -= 2;

                // break;
            }
        } else {
            if let None = index.get(d) {
                index.push(0);

                d += 1;
            } else {
                index[d] += 1;

                if index[d] > arr.shape[d] - 1 {
                    if d == 0 {
                        break;
                    }

                    index.pop();
                    d -= 1;
                } else {
                    d += 1;
                }
            }
        }
    }

    Arrayy::from_vector(shape, result)
}

pub fn slice_replace_arr_unsafe(arr: &mut Arrayy, slice: &[ArrSlice], replace: &Arrayy) {
    let arr_shape = &arr.shape;
    let ranges = slice
        .iter()
        .enumerate()
        .map(|(idx, arr_slice)| {
            let start = if arr_slice.0.unwrap_or(0) >= 0 {
                arr_slice.0.unwrap_or(0)
            } else {
                (arr_shape[idx] as i32) + arr_slice.0.unwrap()
            };

            let stop = if arr_slice.1.unwrap_or(arr_shape[idx] as i32) >= 0 {
                arr_slice.1.unwrap_or(arr_shape[idx] as i32)
            } else {
                (arr_shape[idx] as i32) + arr_slice.1.unwrap() + 1
            };

            start as usize..stop as usize
        })
        .collect::<Vec<Range<usize>>>();

    let mut save_q = VecDeque::new();
    save_q.push_front((ranges[0].start, ranges[0].end, 0));

    let mut index = 0;
    while save_q.len() > 0 {
        let (start, end, d) = save_q.pop_back().unwrap();
        let length = (&arr_shape[d + 1..]).multiple_sum();

        if d == 0 {
            for i in start..end {
                // i * length..(i * length) + length
                let start = i * length;
                let end = start + length;
                if d == ranges.len() - 1 {
                    for v in &mut arr.value[start..end] {
                        *v = replace.value[index];
                        index += 1;
                    }
                } else {
                    save_q.push_front((start, end, d + 1));
                }
            }
        } else {
            for i in ranges[d].start..ranges[d].end {
                let start = i * length + start;
                let end = start + length;
                if d == ranges.len() - 1 {
                    for v in &mut arr.value[start..end] {
                        *v = replace.value[index];
                        index += 1;
                    }
                } else {
                    save_q.push_front((start, end, d + 1));
                }
            }
        }
    }
}

//
pub fn slice_replace_arr(arr: &mut Arrayy, slice: Vec<ArrSlice>, replace: &Arrayy) {
    let mut index = Vec::new();
    let mut d = 0;
    let shape = &arr.shape;
    let replace_vector = &replace.value;
    let replace_length = (&shape[slice.len()..]).multiple_sum();

    let slice = slice
        .into_iter()
        .enumerate()
        .map(|(idx, slice)| {
            let mut start = slice.0.unwrap_or(0);
            if start < 0 {
                start = (shape[idx] as i32) + start;

                if start < 0 {
                    panic!(
                        "slice out of shape, array shape is {:?} but will slice by {:?}",
                        shape,
                        slice
                    );
                }
            }

            let mut stop = slice.1.unwrap_or(shape[idx] as i32);
            if stop < 0 {
                stop = (shape[idx] as i32) + stop + 1;

                if stop < 0 {
                    panic!(
                        "slice out of shape, array shape is {:?} but will slice by {:?}",
                        shape,
                        slice
                    );
                }
            }

            ArrRange(Some(start as usize), Some(stop as usize))
        })
        .collect::<Vec<ArrRange>>();

    // let mut result = vec![];
    let mut i_r = 0;
    loop {
        if d == slice.len() {
            let i = d - 1;
            //
            let mut n = true;
            for (i, range) in slice.iter().enumerate() {
                let s = range.0.unwrap_or(0) as usize;
                let e = range.1.unwrap_or(arr.shape[i]);

                if index[i] >= s && index[i] < e {
                } else {
                    n = false;
                    break;
                }
            }

            if n {
                let start = i_r * replace_length;
                let stop = start + replace_length;
                let replace = &replace_vector[start..stop];
                let slice = slice_indexs_mut(&mut arr.value, &arr.shape, &index);

                for (idx, v) in slice.iter_mut().enumerate() {
                    *v = replace[idx];
                }
                i_r += 1;
            }

            index[i] += 1;
            if index[i] > arr.shape[i] - 1 {
                index.pop();

                if d == 1 {
                    break;
                }
                d -= 2;

                // break;
            }
        } else {
            if let None = index.get(d) {
                index.push(0);

                d += 1;
            } else {
                index[d] += 1;

                if index[d] > arr.shape[d] - 1 {
                    if d == 0 {
                        break;
                    }

                    index.pop();
                    d -= 1;
                } else {
                    d += 1;
                }
            }
        }
    }
}
//

// pub fn slice_replace_arr(arr: &mut Arrayy, slice: Vec<ArrSlice>, replace: &Arrayy) {
//     let shape = arr.shape.clone();
//     let slice = slice
//         .iter()
//         .enumerate()
//         .map(|(idx, slice)| {
//             let mut start = slice.0.unwrap_or(0);
//             if start < 0 {
//                 start = (shape[idx] as i32) + start;

//                 if start < 0 {
//                     panic!(
//                         "slice out of shape, array shape is {:?} but will slice by {:?}",
//                         shape,
//                         slice
//                     );
//                 }
//             }

//             let mut stop = slice.1.unwrap_or(shape[idx] as i32);
//             if stop < 0 {
//                 stop = (shape[idx] as i32) + stop + 1;

//                 if stop < 0 {
//                     panic!(
//                         "slice out of shape, array shape is {:?} but will slice by {:?}",
//                         shape,
//                         slice
//                     );
//                 }
//             }

//             ArrRange(Some(start as usize), Some(stop as usize))
//         })
//         .collect::<Vec<ArrRange>>();

//     let mut current_d = 0;
//     let mut index: Vec<usize> = vec![];
//     // let mut vector_idx = vec![];
//     let mut replace_id = 0;
//     loop {
//         if current_d >= shape.len() - 1 {
//             // kolom
//             if let None = index.get(current_d) {
//                 index.push(0);
//             } else {
//                 // operation do here
//                 let range = slice.get(current_d).unwrap_or(&ArrRange(None, None));
//                 let start = range.0.unwrap_or(0);
//                 let stop = range.1.unwrap_or(shape[current_d]);

//                 if index[current_d] >= start && index[current_d] < stop {
//                     arr.index_mut(
//                         index
//                             .clone()
//                             .into_iter()
//                             .map(|v| v as i32)
//                             .collect::<Vec<i32>>(),
//                         Arrayy::from_vector(vec![1], vec![replace.value[replace_id]])
//                     );
//                     replace_id += 1;
//                 }

//                 if index[current_d] < *shape.last().unwrap() - 1 {
//                     index[current_d] += 1;
//                 } else {
//                     index.pop();

//                     if current_d == 0 {
//                         break;
//                     } else {
//                         current_d -= 1;
//                     }
//                 }
//             }
//         } else {
//             if let None = index.get(current_d) {
//                 let range = slice.get(current_d).unwrap_or(&ArrRange(None, None));
//                 let start_range = range.0.unwrap_or(0);
//                 index.push(start_range);
//                 current_d += 1;
//             } else {
//                 let range = slice.get(current_d).unwrap_or(&ArrRange(None, None));
//                 if
//                     index[current_d] < shape[current_d] - 1 &&
//                     index[current_d] < range.1.unwrap_or(shape[current_d]) - 1
//                 {
//                     index[current_d] += 1;
//                     current_d += 1;
//                 } else {
//                     index.pop();

//                     if current_d == 0 {
//                         break;
//                     } else {
//                         current_d -= 1;
//                     }
//                 }
//             }
//         }
//     }

//     // for idx_replace
// }
