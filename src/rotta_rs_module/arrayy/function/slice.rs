use crate::rotta_rs_module::arrayy::*;

#[derive(Debug, Clone)]
pub struct ArrSlice(pub Option<i32>, pub Option<i32>);

#[derive(Debug, Clone)]
struct ArrRange(pub Option<usize>, pub Option<usize>);

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

pub fn slice_replace_arr(arr: &mut Arrayy, slice: Vec<ArrSlice>, replace: &Arrayy) {
    let shape = arr.shape.clone();
    let slice = slice
        .iter()
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

    let mut current_d = 0;
    let mut index: Vec<usize> = vec![];
    // let mut vector_idx = vec![];
    let mut replace_id = 0;
    loop {
        if current_d >= shape.len() - 1 {
            // kolom
            if let None = index.get(current_d) {
                index.push(0);
            } else {
                // operation do here
                let range = slice.get(current_d).unwrap_or(&ArrRange(None, None));
                let start = range.0.unwrap_or(0);
                let stop = range.1.unwrap_or(shape[current_d]);

                if index[current_d] >= start && index[current_d] < stop {
                    arr.index_mut(
                        index
                            .clone()
                            .into_iter()
                            .map(|v| v as i32)
                            .collect::<Vec<i32>>(),
                        Arrayy::from_vector(vec![1], vec![replace.value[replace_id]])
                    );
                    replace_id += 1;
                }

                if index[current_d] < *shape.last().unwrap() - 1 {
                    index[current_d] += 1;
                } else {
                    index.pop();

                    if current_d == 0 {
                        break;
                    } else {
                        current_d -= 1;
                    }
                }
            }
        } else {
            if let None = index.get(current_d) {
                let range = slice.get(current_d).unwrap_or(&ArrRange(None, None));
                let start_range = range.0.unwrap_or(0);
                index.push(start_range);
                current_d += 1;
            } else {
                let range = slice.get(current_d).unwrap_or(&ArrRange(None, None));
                if
                    index[current_d] < shape[current_d] - 1 &&
                    index[current_d] < range.1.unwrap_or(shape[current_d]) - 1
                {
                    index[current_d] += 1;
                    current_d += 1;
                } else {
                    index.pop();

                    if current_d == 0 {
                        break;
                    } else {
                        current_d -= 1;
                    }
                }
            }
        }
    }

    // for idx_replace
}
