#include "llama.h"
#include "haven_sampler.h"
#include "httplib.h"
#include "json.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <atomic>

using json = nlohmann::json;

struct HavenEngineState {
    std::mutex engine_mutex;
    struct llama_model * model = nullptr;
    struct llama_context * ctx = nullptr;
    const struct llama_vocab * vocab = nullptr;
    std::string active_model_path = "";
    int port = 8088;
};

static HavenEngineState g_state;

static bool load_model(const std::string & path, int n_gpu_layers = 99, int n_ctx = 4096) {
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

    std::cout << "[haven-engine] Loading GGUF Model: " << path << std::endl;
    g_state.model = llama_model_load_from_file(path.c_str(), mparams);
    if (!g_state.model) {
        std::cerr << "[haven-engine] Failed to load GGUF model file." << std::endl;
        return false;
    }

    g_state.vocab = llama_model_get_vocab(g_state.model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = 512;

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

int main(int argc, char ** argv) {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Haven LLM Engine Server (haven-server) v1.0.0           " << std::endl;
    std::cout << "  Native C++ Zero-Drift Sampler & HTTP Streaming Server   " << std::endl;
    std::cout << "==========================================================" << std::endl;

    std::string default_model = "C:\\Users\\admin\\gemma4-turbo-family\\haven-chat-v3.0.3.gguf";
    int port = 8088;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" || arg == "-m") {
            if (i + 1 < argc) default_model = argv[++i];
        } else if (arg == "--port" || arg == "-p") {
            if (i + 1 < argc) port = std::stoi(argv[++i]);
        }
    }

    llama_backend_init();

    if (!load_model(default_model)) {
        std::cerr << "[haven-engine] Warning: Initial model load failed or file not found. Engine ready for hot-swap." << std::endl;
    }

    httplib::Server svr;

    // 1. Health Endpoint
    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        json response = {
            {"status", "ok"},
            {"engine", "haven-llama-cpp"},
            {"active_model", g_state.active_model_path},
            {"zero_drift_sampler", true}
        };
        res.set_content(response.dump(2), "application/json");
    });

    // 2. Model Hot-Swap Endpoint
    svr.Post("/v1/haven/model/load", [](const httplib::Request & req, httplib::Response & res) {
        try {
            auto body = json::parse(req.body);
            std::string model_path = body.value("model_path", "");
            int n_gpu_layers = body.value("n_gpu_layers", 99);
            int n_ctx = body.value("n_ctx", 4096);

            if (model_path.empty()) {
                res.status = 400;
                res.set_content("{\"error\": \"model_path required\"}", "application/json");
                return;
            }

            bool success = load_model(model_path, n_gpu_layers, n_ctx);
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

    // 3. Streaming Chat Completions Endpoint
    svr.Post("/v1/haven/chat/completions", [](const httplib::Request & req, httplib::Response & res) {
        json request_json;
        try {
            request_json = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content("{\"error\": \"Invalid JSON request\"}", "application/json");
            return;
        }

        std::string prompt = request_json.value("prompt", "");
        std::string user_gender_str = request_json.value("user_gender", "female");
        std::string user_name = request_json.value("user_name", "Daniel");
        float temp = request_json.value("temperature", 0.7f);
        int max_tokens = request_json.value("max_tokens", 1024);

        if (prompt.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"prompt required\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(g_state.engine_mutex);
        if (!g_state.model || !g_state.ctx) {
            res.status = 500;
            res.set_content("{\"error\": \"No active model loaded\"}", "application/json");
            return;
        }

        // Configure C++ Haven Sampler
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
            h_opts.user_gender = UserGender::Unspecified;
        }

        // Tokenize prompt
        std::vector<llama_token> tokens(prompt.length() + 32);
        int n_tokens = llama_tokenize(g_state.vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(g_state.vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
        }
        tokens.resize(n_tokens);

        // Build Sampler Chain
        struct llama_sampler * haven_smpl = llama_sampler_init_haven(g_state.vocab, h_opts);
        struct llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain, haven_smpl);
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        // Evaluate prompt tokens
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(g_state.ctx, batch) != 0) {
            llama_sampler_free(chain);
            res.status = 500;
            res.set_content("{\"error\": \"llama_decode prompt failed\"}", "application/json");
            return;
        }

        // Stream responses SSE
        res.set_chunked_content_provider("text/event-stream", [chain, max_tokens](size_t offset, httplib::DataSink & sink) mutable {
            int n_decoded = 0;
            while (n_decoded < max_tokens) {
                llama_token id = llama_sampler_sample(chain, g_state.ctx, -1);
                llama_sampler_accept(chain, id);

                if (llama_vocab_is_eog(g_state.vocab, id)) {
                    break;
                }

                char buf[256];
                int len = llama_token_to_piece(g_state.vocab, id, buf, sizeof(buf), 0, true);
                if (len > 0) {
                    std::string token_str(buf, len);
                    json chunk = {{"token", token_str}};
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

            std::string done_event = "data: [DONE]\n\n";
            sink.write(done_event.c_str(), done_event.length());
            sink.done();
            llama_sampler_free(chain);
            return true;
        });
    });

    std::cout << "[haven-engine] HTTP Server listening on http://0.0.0.0:" << port << std::endl;
    svr.listen("0.0.0.0", port);

    llama_backend_free();
    return 0;
}
