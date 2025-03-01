mod model;
mod tokenizing;
use burn::{
    backend::{autodiff::grads::Gradients, wgpu::WgpuDevice, Autodiff, Wgpu},
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdaGradStateItem, AdamConfig, GradientsParams, Optimizer},
    tensor::{activation::softmax, Int, Tensor},
};
use model::*;
use tokenizing::*;

fn main() {
    type MyBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();
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

    // set up model
    let seq2seq_config = Seq2SeqConfig::new(token.count, 124, token.count, 0.3);
    let mut seq2seq_model = seq2seq_config.init::<MyBackend>(&device);
    let loss_fn = CrossEntropyLossConfig::new().init::<MyBackend>(&device);
    let mut optim = AdamConfig::new().init();
    let epochs = 50;

    for epoch in 0..epochs {
        // break;
        let mut avg = 0.0;
        for (idx, (ask, ans)) in dataset.clone().iter().enumerate() {
            println!("{idx}/{}", dataset.len());
            let ask = ask.clone().unsqueeze();
            let context_vector = seq2seq_model.encoder_forward(ask);
            let logits = seq2seq_model.decoder_forward(context_vector);

            // set up
            let logit_shape = logits.dims()[0];
            let target_shape = ans.dims()[0];

            if target_shape < logit_shape {
                let distance = logit_shape - target_shape;
                let fill: Tensor<MyBackend, 1, Int> = Tensor::from(&vec![1; distance][..]);
                let target = Tensor::cat(vec![ans.clone(), fill], 0);

                let loss = loss_fn.forward(logits, target);
                let loss_scalar = loss.clone().into_scalar();
                avg += loss_scalar;

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &seq2seq_model);
                seq2seq_model = optim.step(0.001, seq2seq_model, grads);
            }
        }

        let avg = avg / dataset.len() as f32;
        println!("{epoch} => {avg}");
    }

    // test
    for (ask, ans) in dataset {
        seq2seq_model.valid();
        let ask = ask.clone().unsqueeze();
        let context_vector = seq2seq_model.encoder_forward(ask.clone());
        let logits = seq2seq_model.decoder_forward(context_vector);
        let pred: Tensor<MyBackend, 1, Int> =
            softmax(logits, 1).argmax(1).permute([1, 0]).squeeze(0);

        let ask_vector: Vec<i32> = ask.to_data().to_vec().unwrap();
        let ans_vector: Vec<i32> = ans.to_data().to_vec().unwrap();
        let pred_vector: Vec<i32> = pred.to_data().to_vec().unwrap();

        let ask_sentence = token.index2sentence(ask_vector);
        let ans_sentence = token.index2sentence(ans_vector);
        let pred_sentence = token.index2sentence(pred_vector);
        println!(
            "{ask_sentence} | {pred_sentence} \n {ans_sentence} \n =========================="
        );
    }
}
