mod tokenizing;
use burn::{
    backend::{Autodiff, Wgpu},
    tensor::{Int, Tensor},
};
use tokenizing::*;

fn main() {
    type MyBackend = Autodiff<Wgpu>;
    // setup token
    let mut token = Tokenizing::default();
    let data_list = token.get_data("data.txt");

    // setup tensor
    let mut dataset = vec![];
    for (ask, ans) in data_list {
        let ask_index = token.sentence2index(ask.as_str());
        let ask_tensor: Tensor<MyBackend, 1, Int> = Tensor::from(&ask_index[..]);
        let ask_tensor = Tensor::cat(vec![ask_tensor, Tensor::from([1])], 0);

        let ans_index = token.sentence2index(ans.as_str());
        let ans_tensor: Tensor<MyBackend, 1, Int> = Tensor::from(&ans_index[..]);
        let ans_tensor = Tensor::cat(vec![ans_tensor, Tensor::from([1])], 0);

        dataset.push((ask_tensor, ans_tensor));
    }
}
