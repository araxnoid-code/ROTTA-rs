# Tensor And Arrayy
`ROTTA-rs` has `tensor` and `array` (called Arrayy). By design, tensors don't store data directly; instead, they store `Arrayy`. In other words, data is stored in `arrayy` format, and `tensor` store the data already stored in `arrayy` format, along with important properties for forward and backward passes. Additionally, most operations performed on `tensor` are actually executed by `Arrayy`, and the output is automatically converted to a tensor.

## Why This Design?
I know that using `tensor` directly to store data would be more structurally simple and efficient. This is all because in the early stages of ROTTA-rs development, they still used an external Array library, and tensor would store array.

After `Arrayy` was developed and ready for use, the developers decided not to change the structure of the `tensor` and only changed the data type of the property within the tensor from the external Array library to `Arrayy`. This was all to speed up development rather than having to reconstruct the `tensor` from scratch, given that `tensor` were already very complex at the time.