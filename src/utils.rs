#[allow(unused_imports)]
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};
use rust_bert::RustBertError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tch::{no_grad, Cuda, Device};

// create a enum whose name is ModelType, which includes `LOCAL` and `REMOTE`
enum ModelType {
    LOCAL,
    REMOTE,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelInfo {
    label: String,
    value: i32,
}

fn get_model_path_list(models_dir_path: &str) -> Vec<String> {
    let mut model_list = Vec::new();
    let paths = match fs::read_dir(models_dir_path) {
        Ok(paths) => paths,
        Err(e) => {
            eprintln!("Failed to read directory: {}", e);
            return model_list;
        }
    };
    for path in paths {
        let path = match path {
            Ok(path) => path,
            Err(e) => {
                eprintln!("Failed to read path: {}", e);
                continue;
            }
        };
        let path = path.path();
        if path.is_dir() {
            if let Some(dir_name) = path.file_name().and_then(|s| s.to_str()) {
                model_list.push(dir_name.to_string());
            }
        }
    }
    model_list
}

pub fn get_model_infos(models_dir_path: &str) -> Vec<ModelInfo> {
    let model_path_list = get_model_path_list(models_dir_path);
    model_path_list
        .into_iter()
        .enumerate()
        .map(|(index, label)| ModelInfo {
            label,
            value: (index + 1) as i32,
        })
        .collect()
}

fn load_model(
    model_type: ModelType,
    model_path_or_remote_name: String,
    use_gpu: bool,
) -> Result<SentenceEmbeddingsModel, RustBertError> {
    let device = if use_gpu && Cuda::is_available() {
        Device::cuda_if_available()
    } else {
        Device::Cpu
    };

    match model_type {
        ModelType::LOCAL => {
            let local_model = SentenceEmbeddingsBuilder::local(model_path_or_remote_name)
                .with_device(device)
                .create_model()?;
            println!("Load on Device: {:?}", device);
            Ok(local_model)
        }
        ModelType::REMOTE => Err(RustBertError::InvalidConfigurationError(
            "Remote model loading not implemented yet".into(),
        )),
    }
}

pub fn load_models(
    model_infos: Vec<ModelInfo>,
    models_dir_path: &str,
    // TODO: add ise GPU here.
) -> HashMap<String, SentenceEmbeddingsModel> {
    let mut models: HashMap<String, SentenceEmbeddingsModel> = HashMap::new();
    for model_info in model_infos {
        let model_name = &model_info.label;
        let model_path = PathBuf::from(models_dir_path)
            .join(model_name)
            .to_str()
            .unwrap()
            .to_string();
        match load_model(ModelType::LOCAL, model_path, true) {
            Ok(model) => {
                models.insert(model_name.clone(), model);
            }
            Err(e) => {
                eprintln!("Failed to load model '{}': {:?}", model_name, e);
                continue;
            }
        }
    }
    models
}

pub fn get_prompt_tokens(input: Vec<String>) -> (i32, i32) {
    // TODO: this is a dummy implementation of the prompt lengths, which will be updated later
    let prompt_tokens = 0;
    let total_tokens = prompt_tokens.clone();
    (prompt_tokens, total_tokens)
}
