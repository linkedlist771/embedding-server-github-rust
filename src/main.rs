use actix_web::{
    body::BoxBody, get, http::header::ContentType, middleware::Logger, post, web, App, HttpRequest,
    HttpResponse, HttpServer, Responder,
};
use clap::{App as ClapApp, Arg};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;

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

#[get("/get_model_types")]
async fn get_model_types() -> impl Responder {
    let dummy_model_types = json!([
        {
            "label": "text2vec-base-chinese",
            "value": 1
        }
        ]
    );

    HttpResponse::Ok().json(dummy_model_types)
}

#[post("/embeddings")]
async fn embeddings(embedding_request: web::Json<EmbeddingRequest>) -> impl Responder {
    let model_name = embedding_request.model.clone().unwrap();
    let input = embedding_request.input.clone();
    let mut dummy_embeddings = Vec::<Embedding>::new();
    for (i, text) in input.iter().enumerate() {
        let embedding = Embedding {
            object: "embedding".to_string(),
            index: i as i32,
            embedding: vec![0.1, 0.2, 0.3],
        };
        dummy_embeddings.push(embedding);
    }
    let usage = Usage {
        prompt_tokens: 0,
        total_tokens: 0,
    };
    let embedding_response = EmbeddingResponse {
        object: "list".to_string(),
        data: dummy_embeddings,
        model: model_name,
        usage: usage,
    };
    HttpResponse::Ok().json(embedding_response)
}

// auto reloading :  cargo watch -x run --host 0.0.0.0 --port 8848

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env::set_var("RUST_LOG", "info");
    env::set_var("RUST_BACKTRACE", "1");
    env_logger::init(); // 初始化日志记录器
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

    log::info!("Starting server at {}:{}", host, port);
    // info!("Models directory: {}", models_dir_path);
    // info!("Using GPU: {}", use_gpu);
    // info!("Tokenizer model: {}", tokenizer_model);

    HttpServer::new(|| {
        let logger = Logger::default();
        App::new().wrap(logger).service(
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
