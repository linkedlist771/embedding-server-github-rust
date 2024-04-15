use actix_web::{
    body::BoxBody, get, http::header::ContentType, middleware::Logger, post, web, App, HttpRequest,
    HttpResponse, HttpServer, Responder,
};
use clap::{App as ClapApp, Arg};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::collections::HashMap;
use std::sync::Arc;
mod utils; 
use utils::{get_model_infos, get_prompt_tokens, load_models, ModelInfo};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsModel
};


#[derive(Deserialize, Serialize, Debug)]
struct EmbeddingRequest {
    model: Option<String>,
    input: Vec<String>,
}

#[derive(Deserialize, Serialize, Debug)]
struct Usage {
    prompt_tokens: i32,
    total_tokens: i32,
}

#[derive(Deserialize, Serialize, Debug)]
struct Embedding {
    object: String,
    index: i32,
    embedding: Vec<f32>,
}

#[derive(Deserialize, Serialize, Debug)]
struct EmbeddingResponse {
    object: String,
    data: Vec<Embedding>,
    model: String,
    usage: Usage,
}

struct AppState {
    models: Arc<HashMap<String, SentenceEmbeddingsModel>>,
    model_infos: Arc<Vec<ModelInfo>>,
}

#[get("/get_model_types")]
async fn get_model_types(data: web::Data<AppState>) -> impl Responder {

    HttpResponse::Ok().json(&*data.model_infos)
}

#[post("/embeddings")]
async fn embeddings(
    data: web::Data<AppState>,
    embedding_request: web::Json<EmbeddingRequest>,
) -> Result<HttpResponse, actix_web::Error> {  // Notice the Result type here
    let input = embedding_request.input.clone();
    let model_name = match embedding_request.model.as_ref() {
        Some(name) => name,
        None => return Ok(HttpResponse::BadRequest().body("Model name is required.")),
    };
    let model = match data.models.get(model_name) {
        Some(m) => m,
        None => return Ok(HttpResponse::NotFound().body(format!("Model '{}' not found.", model_name))),
    };

    let mut dummy_embeddings = Vec::<Embedding>::new();
    for (i, text) in input.iter().enumerate() {
        let text_embedding = model.encode(&[text]).map_err(actix_web::error::ErrorInternalServerError)?;
        let embedding = Embedding {
            object: "embedding".to_string(),
            index: i as i32,
            //error[E0507]: cannot move out of index of `Vec<Vec<f32>>`, get the first element of the Vec
            embedding: text_embedding[0].clone(),
        };
        dummy_embeddings.push(embedding);
    }
    let (prompt_tokens, total_tokens) = get_prompt_tokens(input);

    let usage = Usage {
        prompt_tokens,
        total_tokens,
    };

    let embedding_response = EmbeddingResponse {
        object: "list".to_string(),
        data: dummy_embeddings,
        model: model_name.to_string(),
        usage: usage,
    };

    Ok(HttpResponse::Ok().json(embedding_response))
}


// auto reloading :  cargo watch -x run --host 0.0.0.0 --port 8848

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env::set_var("RUST_LOG", "info");
    env::set_var("RUST_BACKTRACE", "1");
    env_logger::init();
    let matches = ClapApp::new("MyApp")
        .arg(
            Arg::with_name("host")
                .long("host")
                .default_value("127.0.0.1")
                .help("host address to bind"),
        )
        .arg(
            Arg::with_name("port")
                .long("port")
                .default_value("8848")
                .help("port to listen on"),
        )
        // .arg(Arg::with_name("models_dir_path")
        //     .long("models_dir_path")
        //     .required(true)
        //     .help("path to the models directory"))
        // .arg(Arg::with_name("use_gpu")
        //     .long("use_gpu")
        //     .takes_value(false)
        //     .help("flag to use GPU"))
        // .arg(Arg::with_name("tokenizer_model")
        //     .long("tokenizer_model")
        //     .default_value("cl100k_base")
        //     .help("tokenizer model name"))
        .get_matches();

    let host = matches.value_of("host").unwrap();
    let port = matches.value_of("port").unwrap().parse::<u16>().unwrap();
    // let models_dir_path = matches.value_of("models_dir_path").unwrap();
    // let use_gpu = matches.is_present("use_gpu");
    // let tokenizer_model = matches.value_of("tokenizer_model").unwrap();

    let models_dir_path = "/mnt/c/Users/23174/Desktop/GitHub Project/algo-rust-bert-demo/resources";
    let model_infos = get_model_infos(models_dir_path);
    // hash map
    let mut models = load_models(model_infos, models_dir_path);
    let app_state = AppState {
        models: Arc::new(models),
        model_infos: Arc::new(model_infos),
    };
    log::info!("Starting server at {}:{}", host, port);
    // info!("Models directory: {}", models_dir_path);
    // info!("Using GPU: {}", use_gpu);
    // info!("Tokenizer model: {}", tokenizer_model);

    HttpServer::new(|| {
        let logger = Logger::default();
        App::new()
            .wrap(logger)
            .app_data(web::Data::new(&app_state)) // Arc allows us to safely share state with handlers
            .service(
                web::scope("/v1")
                    .service(get_model_types)
                    .service(embeddings),
            )
        // .route("/hey", web::get().to(manual_hello))
    })
    .bind((host, port))?
    .run()
    .await
}
