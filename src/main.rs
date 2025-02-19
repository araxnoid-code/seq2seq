mod model;
mod tokenizing;
use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    tensor::Tensor,
};
use model::*;
use std::{collections::HashMap, fs::File, io::Read};
use tokenizing::*;

fn main() {
    type MyBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();
    // split the conversiation
    let mut raw_input = String::new();
    File::open("./data.txt")
        .unwrap()
        .read_to_string(&mut raw_input)
        .unwrap();

    let raw_input = raw_input.replace("?", "").replace(".", "").replace(",", "");
    let mut raw_inputs = raw_input.split("\n").collect::<Vec<&str>>();
    raw_inputs.pop();

    // split ask and answer
    let raw_inputs = raw_inputs
        .into_iter()
        .map(|value| {
            let ask_ans = value.split(" | ").collect::<Vec<&str>>();
            (
                ask_ans.get(0).unwrap().to_string(),
                ask_ans.get(1).unwrap().to_string(),
            )
        })
        .collect::<Vec<(String, String)>>();

    //set up tokenizing
    let mut token = Tokenizing::default();
    for (ask, ans) in &raw_inputs {
        token.set_up_tokenizing_sentence(ask);
        token.set_up_tokenizing_sentence(ans);
    }

    // tokenizing
    let mut input_tokens = vec![];
    let mut output_tokens = vec![];
    for (ask, ans) in &raw_inputs {
        let ask = token.sentence_to_index(ask);
        input_tokens.push(ask);

        let ans = token.sentence_to_index(ans);
        output_tokens.push(ans);
    }

    // tensoring data
    let mut input_tensor = Vec::new();
    for input in input_tokens {
        let mut list = [1.0; 20];
        for (idx, word) in input.iter().enumerate() {
            list[idx] = word.clone();
        }
        let tensor: Tensor<MyBackend, 2> = Tensor::from([list]);
        input_tensor.push(tensor);
    }
    let input_tensor = Tensor::cat(input_tensor, 0);

    let mut output_tensor = Vec::new();
    for output in output_tokens {
        let mut list = [1.0; 20];
        for (idx, word) in output.iter().enumerate() {
            list[idx] = word.clone();
        }
        let tensor: Tensor<MyBackend, 2> = Tensor::from([list]);
        output_tensor.push(tensor);
    }
    let output_tensor = Tensor::cat(output_tensor, 0);
}
