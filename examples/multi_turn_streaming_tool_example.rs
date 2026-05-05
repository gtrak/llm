//! Multi-turn streaming conversation with tool calls and reasoning content preservation.
//!
//! This example demonstrates:
//! 1. Creating an OpenAI-compatible provider targeting a custom endpoint (gary-desktop:1234)
//! 2. Streaming responses with tool call support
//! 3. Capturing reasoning content from model via StreamChunk::Thinking
//! 4. Preserving reasoning content in message history across turns

use futures::StreamExt;
use llm::{
    chat::{ChatMessage, ChatProvider, ChatRole, MessageType, StreamChunk, Tool},
    providers::openai_compatible::{OpenAICompatibleProvider, OpenAIProviderConfig},
};
use serde_json::json;
use std::io::{self, Write};

struct LocalLLMConfig;

impl OpenAIProviderConfig for LocalLLMConfig {
    const PROVIDER_NAME: &'static str = "LocalLLM";
    const DEFAULT_BASE_URL: &'static str = "http://gary-desktop:1234/";
    const DEFAULT_MODEL: &'static str = "llama-3.1-8b";
    const CHAT_ENDPOINT: &'static str = "v1/chat/completions";
}

fn create_weather_tool() -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: llm::chat::FunctionTool {
            name: "get_weather".to_string(),
            description: "Get the current weather for a given city".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string", "description": "The city name" }
                },
                "required": ["city"]
            }),
        },
        cache_control: None,
    }
}

async fn execute_tool(tool_name: &str, arguments: &str) -> String {
    match tool_name {
        "get_weather" => {
            let parsed: serde_json::Value =
                serde_json::from_str(arguments).unwrap_or_default();
            let city = parsed["city"]
                .as_str()
                .unwrap_or("Unknown")
                .to_string();
            format!("The weather in {} is 72 and sunny.", city)
        }
        _ => format!("Unknown tool: {}", tool_name),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = OpenAICompatibleProvider::<LocalLLMConfig>::new(
        "local-key",
        None,
        Some("llama-3.1-8b".into()),
        Some(2048),
        Some(0.7),
        None,
        Some("You are a helpful assistant.".into()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let tool = create_weather_tool();
    let tools = vec![tool];
    let stdout = io::stdout();

    // Initial user message
    let mut messages: Vec<ChatMessage> = vec![ChatMessage {
        role: ChatRole::User,
        message_type: MessageType::Text,
        content: "What's the weather in San Francisco?".to_string(),
        reasoning_content: None,
    }];

    println!("=== Multi-turn Conversation ===\n");

    let mut full_response = String::new();
    let mut has_tool_calls = false;
    let mut accumulated_reasoning = String::new();

    // First turn
    {
        let mut handle = stdout.lock();
        match llm.chat_stream_with_tools(&messages, Some(&tools)).await {
            Ok(mut stream) => {
                while let Some(chunk) = stream.next().await {
                    match chunk? {
                        StreamChunk::Text(text) => {
                            print!("{}", text);
                            handle.flush()?;
                            full_response.push_str(&text);
                        }
                        StreamChunk::Thinking(text) => {
                            eprint!("\n[REASONING: {}]\n", text);
                            accumulated_reasoning.push_str(&text);
                            handle.flush()?;
                        }
                        StreamChunk::ToolUseComplete { tool_call, .. } => {
                            has_tool_calls = true;
                            let tool_output =
                                execute_tool(&tool_call.function.name, &tool_call.function.arguments)
                                    .await;

                            // Feed reasoning back via ChatMessage.reasoning_content field
                            messages.push(ChatMessage {
                                role: ChatRole::Assistant,
                                message_type: MessageType::Text,
                                content: full_response.clone(),
                                reasoning_content: Some(accumulated_reasoning.clone()),
                            });

                            let result_call = llm::ToolCall {
                                id: tool_call.id.clone(),
                                call_type: tool_call.call_type.clone(),
                                function: llm::FunctionCall {
                                    name: tool_call.function.name,
                                    arguments: tool_output,
                                },
                            };
                            messages.push(ChatMessage {
                                role: ChatRole::User,
                                message_type: MessageType::ToolResult(vec![result_call]),
                                content: String::new(),
                                reasoning_content: None,
                            });
                        }
                        StreamChunk::Done { stop_reason } => {
                            eprintln!("\n[Stream done: {}]", stop_reason);
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    }

    // Second turn
    if has_tool_calls {
        println!("\n--- Second Turn ---\n");
        let mut handle = stdout.lock();
        accumulated_reasoning.clear();

        match llm.chat_stream_with_tools(&messages, Some(&tools)).await {
            Ok(mut stream) => {
                while let Some(chunk) = stream.next().await {
                    match chunk? {
                        StreamChunk::Text(text) => {
                            print!("{}", text);
                            handle.flush()?;
                            full_response.push_str(&text);
                        }
                        StreamChunk::Thinking(text) => {
                            eprint!("\n[REASONING: {}]\n", text);
                            accumulated_reasoning.push_str(&text);
                            handle.flush()?;
                        }
                        StreamChunk::Done { stop_reason } => {
                            eprintln!("\n[Stream done: {}]", stop_reason);
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => eprintln!("Error: {}", e),
        }

        messages.push(ChatMessage {
            role: ChatRole::Assistant,
            message_type: MessageType::Text,
            content: full_response.clone(),
            reasoning_content: Some(accumulated_reasoning.clone()),
        });
    }

    println!("\n\n=== Conversation History ===");
    for msg in &messages {
        let role_str = match msg.role {
            ChatRole::User => "USER",
            ChatRole::Assistant => "ASSISTANT",
        };
        let display = if msg.content.len() > 100 {
            format!("{}...", &msg.content[..100])
        } else {
            msg.content.clone()
        };
        println!("{}: {}", role_str, display);
    }

    println!("\n=== Final Response ===\n{}", full_response);
    Ok(())
}
