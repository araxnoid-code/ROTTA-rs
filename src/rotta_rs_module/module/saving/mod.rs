use std::{ fs, sync::{ Arc, RwLock } };

use crate::{ arrayy::Arrayy, Module, Tensor };
use serde::{ Serialize, Deserialize };

#[derive(Debug, Serialize, Deserialize)]
pub struct SavingParameters {
    pub params: Vec<Arrayy>,
}

impl Module {
    pub fn save(&self, path: &str) {
        let mut params = vec![];
        for shared_tensor in self.parameters.lock().unwrap().iter() {
            params.push(shared_tensor.value());
        }

        let parameter = SavingParameters {
            params,
        };

        let json = serde_json
            ::to_string(&parameter)
            .expect("SAVING ERROR: Error For Translate parameters to tensor");
        drop(parameter);

        fs::write(path, json).expect("SAVING ERROR: Error for writing parameters to path");
    }

    pub fn load_save(&self, path: &str) {
        let string = fs
            ::read_to_string(path)
            .expect("LOAD_SAVE ERROR: error while read the save parameters");

        let params = serde_json
            ::from_str::<SavingParameters>(&string)
            .expect("LOAD_SAVE ERROR: Error while translate save paramters");

        for (i, arr) in params.params.into_iter().enumerate() {
            self.parameters.lock().unwrap()[i].update_value(arr);
        }
    }
}
