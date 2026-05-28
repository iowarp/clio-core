/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_CAE_CORE_LABEL_CLIENT_H_
#define CLIO_CAE_CORE_LABEL_CLIENT_H_

#include <string>

namespace clio::cae::core {

/**
 * Synchronously call an Ollama-compatible /api/generate endpoint and
 * return the model's response text.
 *
 * @param endpoint_base   Base URL of the inference server, e.g.
 *                        "http://127.0.0.1:11434". The "/api/generate"
 *                        path is appended internally.
 * @param model           Model name (e.g. "gemma3:1b").
 * @param prompt          Full prompt text (template + user data, already
 *                        composed by the caller).
 * @param context_length  Tokens to allocate for this request. Sent as
 *                        `options.num_ctx` in the JSON body. 0 means
 *                        "omit the option" — Ollama then uses its own
 *                        default (typically 2048), which silently
 *                        truncates anything larger.
 * @param num_predict     Hard cap on response length in tokens. Sent
 *                        as `options.num_predict`. 0 means "omit" —
 *                        Ollama generates until EOS or context fills.
 * @param out_response    Filled with the parsed `response` field on
 *                        success; cleared on failure.
 * @return true on HTTP 200 with a parseable JSON body, false on any
 *         transport, server, or parse error. Errors are logged via HLOG
 *         at kWarning so a labeling failure doesn't propagate up and
 *         break the underlying PutBlob.
 *
 * Implementation is blocking (libcurl easy interface). Callers should
 * invoke from a worker thread that can tolerate the latency, or wrap in
 * a coroutine that off-loads to a dedicated labeling worker. The
 * function reuses no state between calls — it's safe to call from
 * multiple workers concurrently.
 */
bool OllamaGenerate(const std::string &endpoint_base,
                    const std::string &model,
                    const std::string &prompt,
                    int context_length,
                    int num_predict,
                    std::string &out_response);

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_LABEL_CLIENT_H_
