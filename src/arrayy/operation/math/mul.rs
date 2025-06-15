use crate::{ broadcasting, broadcast_concat, Arrayy };

// function
pub fn mul(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    let arr_a_s = arr_a.shape.clone();
    let arr_b_s = arr_b.shape.clone();
    if arr_a_s == arr_b_s {
        //
        let vector = arr_a.value
            .iter()
            .enumerate()
            .map(|(i, v)| { v * arr_b.value[i] })
            .collect::<Vec<f64>>();

        let array = Arrayy::from_vector(arr_a_s, vector);
        return array;
    } else {
        // broadcasting
        let b_shape = broadcast_concat(arr_a, arr_b);

        let mut a = if let Ok(arr) = broadcasting(arr_a, b_shape.clone()) {
            arr
        } else {
            (*arr_a).clone()
        };
        let b = if let Ok(arr) = broadcasting(arr_b, b_shape) { arr } else { (*arr_a).clone() };

        for i in 0..a.value.len() {
            a.value[i] *= b.value[i];
        }
        a
    }
}
