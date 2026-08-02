use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json};
use std::{env, process};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);

    let mut messages = vec![json!({"role": "user", "content": args.prompt})];

    let tools = json!(
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read",
                        }
                    },
                }
            },
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path", "content"],
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to write to",
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file",
                        }
                    },
                }
            }
        }
    );

    loop {
        let response: Value = client
            .chat()
            .create_byot(json!({
                "messages": messages,
                "model": "anthropic/claude-haiku-4.5",
                "tools": tools,
            }))
            .await?;

        let message = &response["choices"][0]["message"];
        messages.push(message.clone());

        if let Some(tool_calls) = response["choices"][0]["message"]["tool_calls"].as_array() {
            for tool_call in tool_calls {
                if tool_call["type"] == "function" {
                    let name = tool_call["function"]["name"].as_str().unwrap();
                    let arguments: Value =
                        serde_json::from_str(tool_call["function"]["arguments"].as_str().unwrap())?;
                    if name == "Read" {
                        let file_path = arguments["file_path"].as_str().unwrap();
                        let contents = std::fs::read_to_string(file_path)?;
                        messages.push(
                            json!({"role": "tool", "tool_call_id": tool_call["id"].as_str(), "content": contents}),
                        );
                    }
                    if name == "Write" {
                        let file_path = arguments["file_path"].as_str().unwrap();
                        let write_contents = arguments["contents"].as_str().unwrap();
                        let results = std::fs::write(file_path, write_contents)?;
                        messages.push(
                            json!({"role": "tool", "tool_call_id": tool_call["id"].as_str(), "content": results}),
                        );
                    }
                }
            }
        } else if let Some(content) = response["choices"][0]["message"]["content"].as_str() {
            println!("{}", content);
            break;
        }
    }

    Ok(())
}
