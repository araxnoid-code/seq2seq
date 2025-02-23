mod model;
mod tokenizing;
use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{Int, Tensor},
};
use model::*;
use std::{fs::File, io::Read};
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
        let mut list = [1; 20];
        for (idx, word) in input.iter().enumerate() {
            list[idx] = word.clone() as i32;
        }
        let tensor: Tensor<MyBackend, 2, Int> = Tensor::from([list]);
        input_tensor.push(tensor);
    }
    let input_tensor_copy = input_tensor.clone();
    let _input_tensor = Tensor::cat(input_tensor, 0);

    let mut output_tensor = Vec::new();
    for output in output_tokens {
        let mut list = [1; 10];
        for (idx, word) in output.iter().enumerate() {
            list[idx] = word.clone() as i32;
        }
        let tensor: Tensor<MyBackend, 2, Int> = Tensor::from([list]);
        output_tensor.push(tensor);
    }
    let output_tensor_copy = output_tensor.clone();
    let _output_tensor = Tensor::cat(output_tensor, 0);

    let seq2seq_cfg = Seq2SeqConfig::new(token.count as usize, 64, token.count as usize);
    let mut seq2seq_model = seq2seq_cfg.init::<MyBackend>(&device);
    let mut optim = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init::<MyBackend>(&device);

    let epoch = 100;

    for i in 0..epoch {
        let mut losses = vec![];
        for (idx, input) in input_tensor_copy.iter().enumerate() {
            let target = output_tensor_copy.get(idx).unwrap().clone();

            let pred = seq2seq_model.forward(input.clone(), Some(target.clone()));
            // break;

            let target: Tensor<MyBackend, 1, Int> = target.clone().reshape([-1]);
            let target = Tensor::cat(vec![target, Tensor::from([0])], 0);
            let loss = loss_fn.forward(pred, target);
            losses.push(loss.clone());

            println!(
                "{idx}/{} => {}",
                input_tensor_copy.len(),
                loss.clone().into_scalar()
            );

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &seq2seq_model);
            seq2seq_model = optim.step(0.0001, seq2seq_model, grads);
        }
        // break;
        let losses = Tensor::cat(losses, 0);
        let losses_sum = losses.sum();
        let loss = losses_sum / input_tensor_copy.len() as f32;
        let loss_scalar = loss.clone().into_scalar();
        println!("{i} | loss {loss_scalar}");

        // let grads = loss.backward();
        // let grads = GradientsParams::from_grads(grads, &seq2seq_model);
        // seq2seq_model = optim.step(0.0001, seq2seq_model, grads);

        // clear
        let test_input = input_tensor_copy.get(0).unwrap().clone();
        let pred = seq2seq_model.forward(test_input, None);
        let max_pred = pred.argmax(1);
        let vector_pred: Vec<i32> = max_pred.to_data().to_vec().unwrap();
        for pred in &vector_pred {
            let word = token.index_to_word.get(pred).unwrap();
            println!("{word}");
        }
    }

    seq2seq_model.valid();
    let test_input = input_tensor_copy.get(0).unwrap().clone();
    let pred = seq2seq_model.forward(test_input, None);
    let max_pred = pred.argmax(1);
    let vector_pred: Vec<i32> = max_pred.to_data().to_vec().unwrap();
    for pred in &vector_pred {
        let word = token.index_to_word.get(pred).unwrap();
        println!("{word}");
    }

    // let a: Tensor<MyBackend, 2> =
    //     Tensor::from([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]);
    // let a = a.sum_dim(0);
    // println!("{a}");
}
