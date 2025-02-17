use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::AutodiffModule;
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::nn::Relu;
use burn::tensor::activation::{relu, sigmoid, softmax, softplus};
use burn::tensor::{loss, Int, TensorData};
use burn::{
    config::Config,
    module::Module,
    nn::{
        loss::{MseLoss, Reduction},
        Linear, LinearConfig,
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Tensor},
};

#[derive(Module, Debug)]
struct Model<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    // softmax: Softmax,
    output: Linear<B>,
}

impl<B: Backend> Model<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.input.forward(input);
        let x = relu(x);
        let x = self.hidden.forward(x);
        let x = relu(x);
        let x = self.output.forward(x);
        let output = softmax(x, 0);
        output
    }
}

#[derive(Debug, Config)]
struct ModelConfig {
    input: usize,
    hidden: usize,
    output: usize,
}

impl ModelConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            input: LinearConfig::new(self.input, self.hidden).init(device),
            hidden: LinearConfig::new(self.hidden, self.hidden).init(device),
            output: LinearConfig::new(self.hidden, self.output).init(device),
        }
    }
}

fn main() {
    // type MyBackend = Wgpu<f32, i32>;
    println!("Hello, world!");
    run::<Autodiff<Wgpu>>(&WgpuDevice::default());
}

#[derive(Config)]
struct TrainingConfig {
    #[config(default = 400)]
    num_epoch: usize,

    #[config(default = 42)]
    seed: u64,

    #[config(default = 0.001)]
    lr: f64,

    //
    model: ModelConfig,
    optimazer: AdamConfig,
}

pub fn run<B: AutodiffBackend>(device: &B::Device) {
    let config_model = ModelConfig::new(1, 256, 2);
    let config_optimazer = AdamConfig::new();
    let config = TrainingConfig::new(config_model, config_optimazer);
    B::seed(config.seed);

    let mut model = config.model.init::<B>(&device);
    let mut optim = config.optimazer.init();

    let raw = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    ];
    let mut predicted_raw: Vec<Tensor<B, 1, Int>> = Vec::new();

    for value in raw {
        if value % 2.0 == 0.0 {
            predicted_raw.push(Tensor::from([1]));
        } else {
            predicted_raw.push(Tensor::from([0]));
        }
    }
    let input_raw = raw
        .into_iter()
        .map(|v| Tensor::from([[v]]))
        .collect::<Vec<Tensor<B, 2>>>();
    let loss_config = CrossEntropyLossConfig::new().init::<B>(&device);
    let input = Tensor::cat(input_raw, 0);
    let predicted = Tensor::cat(predicted_raw, 0);

    // let logit = model.forward(input);
    // let loss = loss_config.forward(logit, predicted);

    for epoch in 0..config.num_epoch {
        let logits = model.forward(input.clone());
        // let loss = MseLoss::new().forward(pred.clone(), observed.clone(), Reduction::Auto);

        let loss = loss_config.forward(logits, predicted.clone());

        println!("{epoch}: loss => {:?}", loss.clone().into_scalar());

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        model = optim.step(config.lr, model, grads);
    }

    model.valid();
    let logits = model.forward(input);
    println!("{} \n", logits);
    println!("{} \n", predicted);
}
