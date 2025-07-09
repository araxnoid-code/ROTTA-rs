use std::ops::{ Range, RangeFrom, RangeFull, RangeTo };

use crate::arrayy::ArrSlice;

pub trait ArrRangeTrait {
    fn get_range(&self) -> (Option<i32>, Option<i32>);
}

impl ArrRangeTrait for RangeFrom<i32> {
    fn get_range(&self) -> (Option<i32>, Option<i32>) {
        (Some(self.start), None)
    }
}

impl ArrRangeTrait for RangeTo<i32> {
    fn get_range(&self) -> (Option<i32>, Option<i32>) {
        (None, Some(self.end))
    }
}

impl ArrRangeTrait for Range<i32> {
    fn get_range(&self) -> (Option<i32>, Option<i32>) {
        (Some(self.start), Some(self.end))
    }
}

impl ArrRangeTrait for RangeFull {
    fn get_range(&self) -> (Option<i32>, Option<i32>) {
        (None, None)
    }
}

pub fn r<T: ArrRangeTrait>(range: T) -> ArrSlice {
    let range = range.get_range();
    ArrSlice(range.0, range.1)
}
