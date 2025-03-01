mod tokenizing;
use burn::{backend::Wgpu, tensor::Tensor};
use tokenizing::*;

fn main() {
    type MyBackend = Wgpu::default();
    // setup token
    let mut token = Tokenizing::default();
    let data_list = token.get_data("data.txt");

    // setup tensor
    for (ask, ans) in data_list {
        let ask_index = token.sentence2index(ask.as_str());
        let ask_tensor: Tensor<MyBackend, 2> = Tensor::from(ask_index);
        println!("{ask_index:?}");
    }
}
