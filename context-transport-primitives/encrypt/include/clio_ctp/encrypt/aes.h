/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_UTIL_ENCRYPT_AES_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_UTIL_ENCRYPT_AES_H_

#if CTP_ENABLE_ENCRYPT

#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#include <string>

#include "clio_ctp/data_structures/all.h"

namespace ctp {

class AES {
 public:
  std::string key_;
  std::string iv_;
  std::string salt_;

 public:
  void CreateInitialVector(const std::string &salt = "") {
    salt_ = salt;
    iv_ = std::string(EVP_CIPHER_iv_length(EVP_aes_256_cbc()), 0);
    RAND_bytes((unsigned char *)iv_.c_str(), iv_.size());
  }

  void GenerateKey(const std::string &password) {
    const EVP_CIPHER *cipher = EVP_aes_256_cbc();
    const EVP_MD *digest = EVP_sha256();
    key_ = std::string(32, 0);
    int ret = EVP_BytesToKey(cipher, digest, (unsigned char *)salt_.c_str(),
                             (unsigned char *)password.c_str(), password.size(),
                             1, (unsigned char *)key_.c_str(),
                             (unsigned char *)iv_.c_str());
    if (!ret) {
      HLOG(kError, "Failed to generate key");
    }
  }

  bool Encrypt(char *output, size_t &output_size, char *input,
               size_t input_size) {
    EVP_CIPHER_CTX *ctx;
    int ret;

    if (!(ctx = EVP_CIPHER_CTX_new())) return false;

    ret = EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL,
                             (unsigned char *)key_.c_str(),
                             (unsigned char *)iv_.c_str());
    if (1 != ret) return false;

    int output_len_int = input_size;
    ret =
        EVP_EncryptUpdate(ctx, (unsigned char *)output, (int *)&output_len_int,
                          (unsigned char *)input, input_size);
    if (1 != ret) return false;

    int ciphertext_len;
    ret = EVP_EncryptFinal_ex(ctx, (unsigned char *)output + input_size,
                              &ciphertext_len);
    output_size = input_size + ciphertext_len;
    if (1 != ret) return false;

    EVP_CIPHER_CTX_free(ctx);
    return true;
  }

  bool Decrypt(char *output, size_t &output_size, char *input,
               size_t input_size) {
    EVP_CIPHER_CTX *ctx;
    int ret;

    if (!(ctx = EVP_CIPHER_CTX_new())) return false;

    ret = EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL,
                             (unsigned char *)key_.c_str(),
                             (unsigned char *)iv_.c_str());
    if (1 != ret) return false;

    int output_size_int;
    ret = EVP_DecryptUpdate(ctx, (unsigned char *)output, &output_size_int,
                            (unsigned char *)input, input_size);
    if (1 != ret) return false;
    output_size = output_size_int;

    int plaintext_len;
    ret = EVP_DecryptFinal_ex(ctx, (unsigned char *)output + output_size_int,
                              &plaintext_len);
    if (1 != ret) return false;

    EVP_CIPHER_CTX_free(ctx);
    return true;
  }
};

}  // namespace ctp

#endif  // CTP_ENABLE_ENCRYPT

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_UTIL_ENCRYPT_AES_H_
