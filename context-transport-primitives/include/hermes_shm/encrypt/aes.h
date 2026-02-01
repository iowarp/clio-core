/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_UTIL_ENCRYPT_AES_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_UTIL_ENCRYPT_AES_H_

#if HSHM_ENABLE_ENCRYPT

#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#include <string>

#include "hermes_shm/data_structures/all.h"

namespace hshm {

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

}  // namespace hshm

#endif  // HSHM_ENABLE_ENCRYPT

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_UTIL_ENCRYPT_AES_H_
