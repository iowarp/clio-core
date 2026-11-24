/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_CAE_SUMMARIZER_LABEL_CLIENT_H_
#define CLIO_CAE_SUMMARIZER_LABEL_CLIENT_H_

#include <string>

namespace clio::cae::summarizer {

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

/**
 * Synchronously call an OpenAI-compatible /chat/completions endpoint and
 * return the first choice's message content.
 *
 * This is the protocol hosted inference services speak -- e.g. the ALCF
 * Inference Endpoints (Metis, Sophia) behind
 * https://inference-api.alcf.anl.gov/resource_server/<cluster>/.../v1.
 *
 * @param endpoint_base   Base URL up to and including the API version, e.g.
 *                        ".../resource_server/metis/api/v1". The
 *                        "/chat/completions" path is appended internally.
 * @param api_key         Bearer token sent as `Authorization: Bearer ...`.
 *                        Empty sends no Authorization header (a local vLLM
 *                        or llama.cpp server typically needs none).
 * @param model           Model id exactly as the server lists it (e.g.
 *                        "Mistral-Large-3-675B-Instruct-2512").
 * @param prompt          Full prompt text, sent as a single user message.
 * @param num_predict     Hard cap on response tokens, sent as `max_tokens`.
 *                        0 omits it and lets the server decide.
 * @param out_response    Filled with choices[0].message.content on success;
 *                        cleared on failure.
 * @return true on HTTP 200 with a parseable body, false on any transport,
 *         server, or parse error (logged at kWarning, never thrown).
 */
bool OpenAIChatComplete(const std::string &endpoint_base,
                        const std::string &api_key,
                        const std::string &model,
                        const std::string &prompt,
                        int num_predict,
                        std::string &out_response);

/** Wire protocol a summarizer pool uses to reach its inference server. */
enum class LabelApi {
  kOllama,  /**< POST {endpoint}/api/generate (default) */
  kOpenAI,  /**< POST {endpoint}/chat/completions with a bearer token */
};

/**
 * Map the compose `label_api` string onto a LabelApi.
 *
 * @param name "ollama" or "openai" (case-insensitive). Empty means ollama.
 * @param out  Set to the parsed protocol on success; untouched otherwise.
 * @return false for any other spelling, so a typo is reported rather than
 *         silently falling back to Ollama.
 */
bool ParseLabelApi(const std::string &name, LabelApi &out);

/**
 * Where and how one summarizer pool reaches its model.
 *
 * The bearer token is never stored in the compose YAML: it is resolved per
 * request from a file or environment variable, so a token refreshed on disk
 * (ALCF access tokens expire after 48 h) is picked up without a restart.
 */
struct InferenceTarget {
  std::string endpoint_;      /**< Base URL of the inference server */
  LabelApi api_ = LabelApi::kOllama;  /**< Wire protocol */
  std::string api_key_file_;  /**< Token file; ${VAR} is expanded */
  std::string api_key_env_;   /**< Env var holding the token */
};

/**
 * Resolve the bearer token for a target.
 *
 * `api_key_file_` wins when set and readable; otherwise `api_key_env_` is
 * consulted. Surrounding whitespace (a trailing newline from `echo` or the
 * ALCF token script) is stripped.
 *
 * @param target The inference target.
 * @return The token, or "" when neither source yields one.
 */
std::string ResolveApiKey(const InferenceTarget &target);

/**
 * Run one prompt against a target using its configured protocol.
 *
 * @param target         Endpoint, protocol, and token source.
 * @param model          Model name / id.
 * @param prompt         Full prompt text.
 * @param context_length Ollama `num_ctx`; ignored by the OpenAI protocol,
 *                       whose servers fix the context window per model.
 * @param num_predict    Response token cap (0 = server default).
 * @param out_response   Model output on success; cleared on failure.
 * @return true on success, false on any error (logged, never thrown).
 */
bool LabelGenerate(const InferenceTarget &target, const std::string &model,
                   const std::string &prompt, int context_length,
                   int num_predict, std::string &out_response);

namespace detail {

/**
 * The CURLOPT_NOPROXY list OllamaGenerate installs: loopback, followed by
 * whatever the environment's no_proxy (or NO_PROXY, when no_proxy is unset or
 * empty) already asked to bypass.
 *
 * Exposed because the behaviour worth pinning cannot be observed through
 * OllamaGenerate. CURLOPT_NOPROXY REPLACES libcurl's own reading of no_proxy
 * rather than adding to it, so a client that listed only loopback would
 * silently push a site's internal hosts back through the proxy -- yet a test
 * calling a loopback endpoint succeeds either way. Only the list itself
 * distinguishes the two.
 */
std::string LoopbackNoProxyList();

}  // namespace detail

}  // namespace clio::cae::summarizer

#endif  // CLIO_CAE_SUMMARIZER_LABEL_CLIENT_H_
