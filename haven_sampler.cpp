#include "haven_sampler.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>

struct haven_sampler_ctx {
    const struct llama_vocab * vocab;
    haven_sampler_options options;
    std::vector<llama_token> female_masked_tokens; // Token IDs for male pronouns to mask out
    std::vector<llama_token> male_masked_tokens;   // Token IDs for female pronouns to mask out
};

static const char * haven_sampler_name(const struct llama_sampler * /*smpl*/) {
    return "haven-zero-drift-sampler";
}

static void haven_sampler_accept(struct llama_sampler * /*smpl*/, llama_token /*token*/) {
    // Optional: track token generation phase (e.g. inside <thought>, inside <call>)
}

static void haven_sampler_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (haven_sampler_ctx *) smpl->ctx;
    if (!ctx || !ctx->options.enable_pronoun_masking) {
        return;
    }

    if (ctx->options.user_gender == UserGender::Female) {
        // Mask out male pronouns so they cannot be sampled for female user
        for (size_t i = 0; i < cur_p->size; ++i) {
            llama_token id = cur_p->data[i].id;
            for (llama_token masked_id : ctx->female_masked_tokens) {
                if (id == masked_id) {
                    cur_p->data[i].logit = -1e9f;
                    break;
                }
            }
        }
    } else if (ctx->options.user_gender == UserGender::Male) {
        // Mask out female pronouns so they cannot be sampled for male user
        for (size_t i = 0; i < cur_p->size; ++i) {
            llama_token id = cur_p->data[i].id;
            for (llama_token masked_id : ctx->male_masked_tokens) {
                if (id == masked_id) {
                    cur_p->data[i].logit = -1e9f;
                    break;
                }
            }
        }
    }
}

static void haven_sampler_reset(struct llama_sampler * /*smpl*/) {
    // Reset state between turns
}

static struct llama_sampler * haven_sampler_clone(const struct llama_sampler * smpl) {
    auto * ctx = (haven_sampler_ctx *) smpl->ctx;
    return llama_sampler_init_haven(ctx->vocab, ctx->options);
}

static void haven_sampler_free(struct llama_sampler * smpl) {
    if (smpl && smpl->ctx) {
        delete (haven_sampler_ctx *) smpl->ctx;
        smpl->ctx = nullptr;
    }
}

static struct llama_sampler_i haven_sampler_i = {
    /* .name   = */ haven_sampler_name,
    /* .accept = */ haven_sampler_accept,
    /* .apply  = */ haven_sampler_apply,
    /* .reset  = */ haven_sampler_reset,
    /* .clone  = */ haven_sampler_clone,
    /* .free   = */ haven_sampler_free,
};

static void cache_pronoun_tokens(haven_sampler_ctx * ctx) {
    if (!ctx || !ctx->vocab) return;

    int n_vocab = llama_vocab_n_tokens(ctx->vocab);
    std::vector<std::string> male_words = {"he", "him", "his", "himself", "He", "Him", "His", "Himself"};
    std::vector<std::string> female_words = {"she", "her", "hers", "herself", "She", "Her", "Hers", "Herself"};

    for (int i = 0; i < n_vocab; ++i) {
        char buf[256];
        int len = llama_token_to_piece(ctx->vocab, i, buf, sizeof(buf), 0, true);
        if (len <= 0) continue;
        std::string piece(buf, len);

        // Strip leading whitespace indicator if present (e.g.   or space)
        std::string clean_piece = piece;
        if (clean_piece.rfind(" ", 0) == 0) clean_piece = clean_piece.substr(1);
        if (clean_piece.rfind("\xe2\x96\x81", 0) == 0) clean_piece = clean_piece.substr(3);

        for (const auto & w : male_words) {
            if (clean_piece == w) {
                ctx->female_masked_tokens.push_back(i);
                break;
            }
        }

        for (const auto & w : female_words) {
            if (clean_piece == w) {
                ctx->male_masked_tokens.push_back(i);
                break;
            }
        }
    }
}

struct llama_sampler * llama_sampler_init_haven(
    const struct llama_vocab * vocab,
    const haven_sampler_options & options
) {
    auto * ctx = new haven_sampler_ctx();
    ctx->vocab = vocab;
    ctx->options = options;
    cache_pronoun_tokens(ctx);

    auto * smpl = new struct llama_sampler();
    smpl->iface = &haven_sampler_i;
    smpl->ctx = ctx;

    return smpl;
}

void haven_sampler_update_options(struct llama_sampler * smpl, const haven_sampler_options & options) {
    if (!smpl || !smpl->ctx) return;
    auto * ctx = (haven_sampler_ctx *) smpl->ctx;
    ctx->options = options;
}
