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
    hidden_1: Linear<B>,
    hidden_2: Linear<B>,
    output: Linear<B>,
}

impl<B: Backend> Model<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.input.forward(input);
        let x = relu(x);
        let x = self.hidden_1.forward(x);
        let x = relu(x);
        let x = self.hidden_2.forward(x);
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
            hidden_1: LinearConfig::new(self.hidden, self.hidden).init(device),
            hidden_2: LinearConfig::new(self.hidden, self.hidden).init(device),
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
    #[config(default = 1000)]
    num_epoch: usize,

    #[config(default = 42)]
    seed: u64,

    #[config(default = 0.0001)]
    lr: f64,

    //
    model: ModelConfig,
    optimazer: AdamConfig,
}

pub fn run<B: AutodiffBackend>(device: &B::Device) {
    let config_model = ModelConfig::new(2, 32, 3);
    let config_optimazer = AdamConfig::new();
    let config = TrainingConfig::new(config_model, config_optimazer);
    B::seed(config.seed);

    let mut model = config.model.init::<B>(&device);
    let mut optim = config.optimazer.init();
    let loss_config = CrossEntropyLossConfig::new().init::<B>(&device);

    let data = vec![
        // sepal petal label
        (3.5, 0.2, 0.0),
        (3.0, 0.2, 0.0),
        (3.2, 0.2, 0.0),
        (3.1, 0.2, 0.0),
        (3.6, 0.3, 0.0),
        (3.2, 1.4, 1.0),
        (3.2, 1.5, 1.0),
        (3.1, 1.5, 1.0),
        (2.3, 1.3, 1.0),
        (2.8, 1.5, 1.0),
        (3.3, 2.5, 2.0),
        (2.7, 1.9, 2.0),
        (3.0, 2.1, 2.0),
        (2.9, 1.8, 2.0),
        (3.0, 2.2, 2.0),
    ];

    let mut input_data: Vec<Tensor<B, 2>> = vec![];
    let mut label_data: Vec<Tensor<B, 1, Int>> = vec![];
    for (sepal, petal, label) in data {
        input_data.push(Tensor::from([[sepal, petal]]));
        label_data.push(Tensor::from([label as i32]));
    }
    let input_data = Tensor::cat(input_data, 0);
    let label_data = Tensor::cat(label_data, 0);

    for epoch in 0..config.num_epoch {
        let logits = model.forward(input_data.clone());
        let loss = loss_config.forward(logits, label_data.clone());

        // if epoch % 10 == 0 {
        println!("{epoch}: loss => {:?}", loss.clone().into_scalar());
        // }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        model = optim.step(config.lr, model, grads);
    }

    model.valid();
    let logits = model.forward(input_data);
    println!("{} \n", logits);
    println!("{} \n", logits.argmax(1));
    println!("{} \n", label_data);
}
