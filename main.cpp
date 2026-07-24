#include "llama.h"
#include "haven_sampler.h"
#include "httplib.h"
#include "json.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <algorithm>
#include <chrono>

using json = nlohmann::json;

struct HavenEngineState {
    std::mutex engine_mutex;
    struct llama_model * model = nullptr;
    struct llama_context * ctx = nullptr;
    const struct llama_vocab * vocab = nullptr;
    std::string active_model_path = "";
    std::string active_model_alias = "haven-chat";
    int port = 11436;
    std::string host = "0.0.0.0";
    int default_n_ctx = 16384;
    int default_gpu_layers = 99;
    int n_threads = 8;
    int n_batch = 2048;
    int n_ubatch = 512;

    // Telemetry tracking
    uint64_t total_eval_tokens = 0;
    uint64_t total_gen_tokens = 0;
    double last_gen_tps = 0.0;
};

static HavenEngineState g_state;

static void set_cors_headers(httplib::Response & res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With");
    res.set_header("Connection", "close");
}

static bool load_model(const std::string & path, int n_gpu_layers = 99, int n_ctx = 16384, int n_threads = 8, int n_batch = 2048) {
    std::lock_guard<std::mutex> lock(g_state.engine_mutex);

    if (g_state.ctx) {
        llama_free(g_state.ctx);
        g_state.ctx = nullptr;
    }
    if (g_state.model) {
        llama_model_free(g_state.model);
        g_state.model = nullptr;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    std::cout << "[haven-engine] Loading GGUF Model: " << path << " (GPU Layers: " << n_gpu_layers << ", Threads: " << n_threads << ", CTX: " << n_ctx << ")" << std::endl;
    g_state.model = llama_model_load_from_file(path.c_str(), mparams);
    if (!g_state.model) {
        std::cerr << "[haven-engine] Failed to load GGUF model file." << std::endl;
        return false;
    }

    g_state.vocab = llama_model_get_vocab(g_state.model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_batch;
    cparams.n_ubatch = 512;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.embeddings = true; // Enable embeddings capability

    g_state.ctx = llama_init_from_model(g_state.model, cparams);
    if (!g_state.ctx) {
        std::cerr << "[haven-engine] Failed to initialize llama_context." << std::endl;
        llama_model_free(g_state.model);
        g_state.model = nullptr;
        return false;
    }

    g_state.active_model_path = path;
    std::cout << "[haven-engine] GGUF Model Loaded Successfully!" << std::endl;
    return true;
}

static void handle_embeddings(const httplib::Request & req, httplib::Response & res) {
    set_cors_headers(res);

    json request_json;
    try {
        request_json = json::parse(req.body);
    } catch (...) {
        res.status = 400;
        res.set_content("{\"error\": \"Invalid JSON request\"}", "application/json");
        return;
    }

    std::string input = "";
    if (request_json.contains("input")) {
        if (request_json["input"].is_string()) {
            input = request_json["input"].get<std::string>();
        } else if (request_json["input"].is_array() && !request_json["input"].empty()) {
            if (request_json["input"][0].is_string()) {
                input = request_json["input"][0].get<std::string>();
            }
        }
    }

    if (input.empty()) {
        res.status = 400;
        res.set_content("{\"error\": \"input string required\"}", "application/json");
        return;
    }

    std::lock_guard<std::mutex> lock(g_state.engine_mutex);

    if (!g_state.model || !g_state.ctx) {
        res.status = 500;
        res.set_content("{\"error\": \"No active model loaded\"}", "application/json");
        return;
    }

    // Synchronize and clear memory
    llama_synchronize(g_state.ctx);
    llama_memory_clear(llama_get_memory(g_state.ctx), true);

    // Tokenize
    std::vector<llama_token> tokens(input.length() + 32);
    int n_tokens = llama_tokenize(g_state.vocab, input.c_str(), input.length(), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(g_state.vocab, input.c_str(), input.length(), tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    // Decode prompt for embeddings
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(g_state.ctx, batch) != 0) {
        res.status = 500;
        res.set_content("{\"error\": \"Failed to compute embeddings\"}", "application/json");
        return;
    }

    // Extract sequence embeddings
    int n_embd = llama_model_n_embd(g_state.model);
    float * embd_res = llama_get_embeddings_seq(g_state.ctx, 0);
    if (!embd_res) {
        embd_res = llama_get_embeddings(g_state.ctx);
    }

    std::vector<float> embedding_vec(n_embd, 0.0f);
    if (embd_res) {
        for (int i = 0; i < n_embd; ++i) {
            embedding_vec[i] = embd_res[i];
        }
    }

    json response = {
        {"object", "list"},
        {"data", json::array({{
            {"object", "embedding"},
            {"embedding", embedding_vec},
            {"index", 0}
        }})},
        {"model", g_state.active_model_alias},
        {"usage", {
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }}
    };

    res.set_content(response.dump(), "application/json");
}

static void handle_chat_completion(const httplib::Request & req, httplib::Response & res) {
    set_cors_headers(res);

    json request_json;
    try {
        request_json = json::parse(req.body);
    } catch (...) {
        res.status = 400;
        res.set_content("{\"error\": \"Invalid JSON request\"}", "application/json");
        return;
    }

    std::string prompt = "";

    // Parse messages array if standard OpenAI format
    if (request_json.contains("messages") && request_json["messages"].is_array()) {
        for (const auto & msg : request_json["messages"]) {
            std::string role = msg.value("role", "user");
            std::string content = msg.value("content", "");
            if (role == "system") {
                prompt += "<|im_start|>system\n" + content + "<|im_end|>\n";
            } else if (role == "user") {
                prompt += "<|im_start|>user\n" + content + "<|im_end|>\n";
            } else if (role == "assistant") {
                prompt += "<|im_start|>assistant\n" + content + "<|im_end|>\n";
            }
        }
        prompt += "<|im_start|>assistant\n";
    } else {
        prompt = request_json.value("prompt", "");
    }

    std::string user_gender_str = request_json.value("user_gender", "female");
    std::string user_name = request_json.value("user_name", "Daniel");
    
    // Sampling Hyperparameters
    float temp = request_json.value("temperature", 0.7f);
    float top_p = request_json.value("top_p", 0.9f);
    float min_p = request_json.value("min_p", 0.05f);
    int top_k = request_json.value("top_k", 40);
    float typical_p = request_json.value("typical_p", 1.0f);
    float penalty_repeat = request_json.value("repeat_penalty", request_json.value("frequency_penalty", 1.1f));
    float penalty_freq = request_json.value("presence_penalty", 0.0f);
    float penalty_present = request_json.value("presence_penalty", 0.0f);
    int penalty_last_n = request_json.value("repeat_last_n", 64);
    int max_tokens = request_json.value("max_tokens", request_json.value("n_predict", 1024));

    if (prompt.empty()) {
        res.status = 400;
        res.set_content("{\"error\": \"prompt or messages required\"}", "application/json");
        return;
    }

    // Acquire lock for the ENTIRE inference session to prevent concurrent segfaults
    std::shared_ptr<std::unique_lock<std::mutex>> lock = std::make_shared<std::unique_lock<std::mutex>>(g_state.engine_mutex);

    if (!g_state.model || !g_state.ctx) {
        res.status = 500;
        res.set_content("{\"error\": \"No active model loaded\"}", "application/json");
        return;
    }

    // Synchronize context and reset KV cache data buffers and metadata to position 0
    if (g_state.ctx) {
        llama_synchronize(g_state.ctx);
        llama_memory_clear(llama_get_memory(g_state.ctx), true);
    }

    // Identify exact token ID for <|im_end|> stop sequence
    llama_token im_end_id = LLAMA_TOKEN_NULL;
    int n_eot = llama_tokenize(g_state.vocab, "<|im_end|>", 10, &im_end_id, 1, true, true);
    if (n_eot < 1 || im_end_id == LLAMA_TOKEN_NULL) {
        im_end_id = 107; // Fallback to Gemma <|im_end|> token ID
    }

    // Configure C++ Haven Zero-Drift Sampler
    haven_sampler_options h_opts;
    h_opts.user_name = user_name;
    h_opts.enable_pronoun_masking = true;
    std::string g_lower = user_gender_str;
    std::transform(g_lower.begin(), g_lower.end(), g_lower.begin(), ::tolower);
    if (g_lower.find("female") != std::string::npos || g_lower.find("woman") != std::string::npos || g_lower.find("she") != std::string::npos) {
        h_opts.user_gender = UserGender::Female;
    } else if (g_lower.find("male") != std::string::npos || g_lower.find("man") != std::string::npos || g_lower.find("he") != std::string::npos) {
        h_opts.user_gender = UserGender::Male;
    } else {
        std::string p_lower = prompt;
        std::transform(p_lower.begin(), p_lower.end(), p_lower.begin(), ::tolower);
        if (p_lower.find("female") != std::string::npos || p_lower.find("she/her") != std::string::npos || p_lower.find("woman") != std::string::npos) {
            h_opts.user_gender = UserGender::Female;
        } else if (p_lower.find("male") != std::string::npos || p_lower.find("he/him") != std::string::npos || p_lower.find("man") != std::string::npos) {
            h_opts.user_gender = UserGender::Male;
        } else {
            h_opts.user_gender = UserGender::Unspecified;
        }
    }

    // Tokenize prompt
    std::vector<llama_token> tokens(prompt.length() + 32);
    int n_tokens = llama_tokenize(g_state.vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(g_state.vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    // Build Full Sampler Chain (Haven Zero-Drift + Penalties + Min-P + Top-P + Temp)
    struct llama_sampler * haven_smpl = llama_sampler_init_haven(g_state.vocab, h_opts);
    struct llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, haven_smpl);

    if (penalty_repeat != 1.0f || penalty_freq != 0.0f || penalty_present != 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(penalty_last_n, penalty_repeat, penalty_freq, penalty_present));
    }
    if (top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    }
    if (typical_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_typical(typical_p, 1));
    }
    if (top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
    }
    if (min_p > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 1));
    }
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    auto start_time = std::chrono::high_resolution_clock::now();

    // Chunked prompt token evaluation (handles large prompt inputs cleanly)
    const size_t batch_size = 512;
    for (size_t i = 0; i < tokens.size(); i += batch_size) {
        size_t n_eval = std::min(batch_size, tokens.size() - i);
        llama_batch chunk_batch = llama_batch_get_one(tokens.data() + i, (int)n_eval);
        if (llama_decode(g_state.ctx, chunk_batch) != 0) {
            llama_sampler_free(chain);
            res.status = 500;
            res.set_content("{\"error\": \"llama_decode prompt failed\"}", "application/json");
            return;
        }
    }

    g_state.total_eval_tokens += tokens.size();

    bool stream = request_json.value("stream", true);
    auto now_ts = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::string req_id = "chatcmpl-haven-" + std::to_string(now_ts);

    if (stream) {
        // Stream responses SSE with lock captured until completion
        res.set_chunked_content_provider("text/event-stream", [lock, chain, im_end_id, max_tokens, req_id, now_ts, start_time](size_t offset, httplib::DataSink & sink) mutable {
            if (!chain) {
                return false;
            }

            int n_decoded = 0;
            while (n_decoded < max_tokens) {
                llama_token id = llama_sampler_sample(chain, g_state.ctx, -1);
                llama_sampler_accept(chain, id);

                if (llama_vocab_is_eog(g_state.vocab, id) || id == im_end_id) {
                    break;
                }

                char buf[256];
                int len = llama_token_to_piece(g_state.vocab, id, buf, sizeof(buf), 0, true);
                if (len > 0) {
                    std::string token_str(buf, len);
                    if (token_str.find("<|im_end|>") != std::string::npos || token_str.find("<eos>") != std::string::npos) {
                        break;
                    }
                    if (token_str.find("<|channel") != std::string::npos || token_str.find("<|thought") != std::string::npos || token_str.find("<|call") != std::string::npos) {
                        continue;
                    }
                    json chunk = {
                        {"id", req_id},
                        {"object", "chat.completion.chunk"},
                        {"created", now_ts},
                        {"model", g_state.active_model_alias},
                        {"choices", json::array({{
                            {"index", 0},
                            {"delta", {{"content", token_str}}},
                            {"finish_reason", nullptr}
                        }})},
                        {"token", token_str},
                        {"content", token_str}
                    };
                    std::string event = "data: " + chunk.dump() + "\n\n";
                    if (!sink.write(event.c_str(), event.length())) {
                        break;
                    }
                }

                // Decode next token
                llama_batch token_batch = llama_batch_get_one(&id, 1);
                if (llama_decode(g_state.ctx, token_batch) != 0) {
                    break;
                }
                n_decoded++;
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
            if (elapsed_sec > 0.0 && n_decoded > 0) {
                g_state.last_gen_tps = n_decoded / elapsed_sec;
                g_state.total_gen_tokens += n_decoded;
            }

            std::string done_event = "data: [DONE]\n\n";
            sink.write(done_event.c_str(), done_event.length());
            sink.done();
            if (chain) {
                llama_sampler_free(chain);
                chain = nullptr;
            }
            return false;
        });
    } else {
        // Non-streaming response
        std::string full_response = "";
        int n_decoded = 0;
        while (n_decoded < max_tokens) {
            llama_token id = llama_sampler_sample(chain, g_state.ctx, -1);
            llama_sampler_accept(chain, id);

            if (llama_vocab_is_eog(g_state.vocab, id) || id == im_end_id) {
                break;
            }

            char buf[256];
            int len = llama_token_to_piece(g_state.vocab, id, buf, sizeof(buf), 0, true);
            if (len > 0) {
                std::string token_str(buf, len);
                if (token_str.find("<|im_end|>") != std::string::npos || token_str.find("<eos>") != std::string::npos) {
                    break;
                }
                if (token_str.find("<|channel") != std::string::npos || token_str.find("<|thought") != std::string::npos || token_str.find("<|call") != std::string::npos) {
                    continue;
                }
                full_response.append(buf, len);
            }

            llama_batch token_batch = llama_batch_get_one(&id, 1);
            if (llama_decode(g_state.ctx, token_batch) != 0) {
                break;
            }
            n_decoded++;
        }
        llama_sampler_free(chain);

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
        if (elapsed_sec > 0.0 && n_decoded > 0) {
            g_state.last_gen_tps = n_decoded / elapsed_sec;
            g_state.total_gen_tokens += n_decoded;
        }

        json resp = {
            {"id", req_id},
            {"object", "chat.completion"},
            {"created", now_ts},
            {"model", g_state.active_model_alias},
            {"choices", json::array({{
                {"index", 0},
                {"message", {{"role", "assistant"}, {"content", full_response}}},
                {"finish_reason", "stop"}
            }})},
            {"content", full_response}
        };
        res.set_content(resp.dump(), "application/json");
    }
}

int main(int argc, char ** argv) {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Haven LLM Engine Server (haven-server) v1.0.0           " << std::endl;
    std::cout << "  Native C++ Zero-Drift Sampler & HTTP Server             " << std::endl;
    std::cout << "==========================================================" << std::endl;

    std::string default_model = "C:\\Users\\admin\\gemma4-turbo-family\\haven-chat-v3.0.3.gguf";
    g_state.port = 11436;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            default_model = argv[++i];
        } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            g_state.port = std::stoi(argv[++i]);
        } else if ((arg == "--host" || arg == "-h") && i + 1 < argc) {
            g_state.host = argv[++i];
        } else if ((arg == "--ctx-size" || arg == "--ctx_size" || arg == "-c") && i + 1 < argc) {
            g_state.default_n_ctx = std::stoi(argv[++i]);
        } else if ((arg == "--n-gpu-layers" || arg == "-ngl") && i + 1 < argc) {
            g_state.default_gpu_layers = std::stoi(argv[++i]);
        } else if ((arg == "--threads" || arg == "-t") && i + 1 < argc) {
            g_state.n_threads = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            g_state.n_batch = std::stoi(argv[++i]);
        } else if (arg == "--ubatch-size" && i + 1 < argc) {
            g_state.n_ubatch = std::stoi(argv[++i]);
        } else if (arg == "--alias" && i + 1 < argc) {
            g_state.active_model_alias = argv[++i];
        } else if (arg == "--mmproj" || arg == "--n-predict") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                ++i;
            }
        }
    }

    llama_backend_init();

    if (!load_model(default_model, g_state.default_gpu_layers, g_state.default_n_ctx, g_state.n_threads, g_state.n_batch)) {
        std::cerr << "[haven-engine] Warning: Initial model load failed or file not found. Engine ready for hot-swap." << std::endl;
    }

    httplib::Server svr;

    // OPTIONS preflight for CORS
    svr.Options(".*", [](const httplib::Request &, httplib::Response & res) {
        set_cors_headers(res);
        res.status = 200;
    });

    // 1. Health & Telemetry Metrics Endpoints
    auto health_handler = [](const httplib::Request &, httplib::Response & res) {
        set_cors_headers(res);
        json response = {
            {"status", "ok"},
            {"engine", "haven-llama-cpp"},
            {"active_model", g_state.active_model_path},
            {"alias", g_state.active_model_alias},
            {"zero_drift_sampler", true},
            {"tokens_per_second", g_state.last_gen_tps},
            {"total_eval_tokens", g_state.total_eval_tokens},
            {"total_gen_tokens", g_state.total_gen_tokens}
        };
        res.set_content(response.dump(2), "application/json");
    };
    svr.Get("/health", health_handler);
    svr.Get("/v1/health", health_handler);

    svr.Get("/metrics", [](const httplib::Request &, httplib::Response & res) {
        set_cors_headers(res);
        json response = {
            {"status", "ok"},
            {"engine", "haven-llama-cpp"},
            {"active_model", g_state.active_model_path},
            {"n_ctx", g_state.default_n_ctx},
            {"n_threads", g_state.n_threads},
            {"gpu_layers", g_state.default_gpu_layers},
            {"last_gen_tps", g_state.last_gen_tps},
            {"total_eval_tokens", g_state.total_eval_tokens},
            {"total_gen_tokens", g_state.total_gen_tokens}
        };
        res.set_content(response.dump(2), "application/json");
    });

    svr.Get("/v1/models", [](const httplib::Request &, httplib::Response & res) {
        set_cors_headers(res);
        json response = {
            {"object", "list"},
            {"data", json::array({{
                {"id", g_state.active_model_alias},
                {"object", "model"},
                {"created", 1677610602},
                {"owned_by", "haven"}
            }})}
        };
        res.set_content(response.dump(2), "application/json");
    });

    // 2. Vector Embeddings Endpoints
    svr.Post("/v1/embeddings", handle_embeddings);
    svr.Post("/embedding", handle_embeddings);

    // 3. Model Hot-Swap Endpoint
    svr.Post("/v1/haven/model/load", [](const httplib::Request & req, httplib::Response & res) {
        set_cors_headers(res);
        try {
            auto body = json::parse(req.body);
            std::string model_path = body.value("model_path", "");
            int n_gpu_layers = body.value("n_gpu_layers", 99);
            int n_ctx = body.value("n_ctx", 16384);
            int n_threads = body.value("n_threads", g_state.n_threads);

            if (model_path.empty()) {
                res.status = 400;
                res.set_content("{\"error\": \"model_path required\"}", "application/json");
                return;
            }

            bool success = load_model(model_path, n_gpu_layers, n_ctx, n_threads, g_state.n_batch);
            if (success) {
                res.set_content("{\"status\": \"success\", \"active_model\": \"" + model_path + "\"}", "application/json");
            } else {
                res.status = 500;
                res.set_content("{\"error\": \"failed to load model\"}", "application/json");
            }
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(std::string("{\"error\": \"") + e.what() + "\"}", "application/json");
        }
    });

    // 4. Multi-Route Chat & Completion Endpoints
    svr.Post("/v1/chat/completions", handle_chat_completion);
    svr.Post("/v1/completions", handle_chat_completion);
    svr.Post("/completion", handle_chat_completion);
    svr.Post("/v1/haven/chat/completions", handle_chat_completion);

    std::cout << "[haven-engine] HTTP Server listening on http://" << g_state.host << ":" << g_state.port << std::endl;
    svr.listen(g_state.host.c_str(), g_state.port);

    llama_backend_free();
    return 0;
}
