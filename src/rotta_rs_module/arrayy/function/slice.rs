use std::{ ops::Range, collections::VecDeque };

use crate::rotta_rs_module::arrayy::*;

#[derive(Debug, Clone)]
pub struct ArrSlice(pub Option<i32>, pub Option<i32>);

#[derive(Debug, Clone)]
struct ArrRange(pub Option<usize>, pub Option<usize>);

pub fn slice_arr(arr: &Arrayy, slice: &[ArrSlice]) -> Arrayy {
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

pub fn slice_replace_arr(arr: &mut Arrayy, slice: &[ArrSlice], replace: &Arrayy) {
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
