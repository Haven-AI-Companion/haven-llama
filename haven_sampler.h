#ifndef HAVEN_SAMPLER_H
#define HAVEN_SAMPLER_H

#include "llama.h"
#include <string>
#include <vector>

enum class UserGender {
    Unspecified = 0,
    Female = 1,
    Male = 2
};

struct haven_sampler_options {
    UserGender user_gender = UserGender::Unspecified;
    std::string user_name = "User";
    bool enable_pronoun_masking = true;
    bool enable_phase_events = true;
};

// Initializes the custom Haven C++ Sampler Plugin
struct llama_sampler * llama_sampler_init_haven(
    const struct llama_vocab * vocab,
    const haven_sampler_options & options
);

// Updates dynamic options on an active Haven sampler
void haven_sampler_update_options(struct llama_sampler * smpl, const haven_sampler_options & options);

#endif // HAVEN_SAMPLER_H
