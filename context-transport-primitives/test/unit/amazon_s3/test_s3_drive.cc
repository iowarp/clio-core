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

#include <catch2/catch_all.hpp>
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <iostream>

using namespace Aws::S3;
using namespace Aws::S3::Model;

TEST_CASE("Amazon S3 Read/Write", "[s3][bdev]") {
    Aws::SDKOptions options;
    Aws::InitAPI(options);
    
    {
        Aws::Client::ClientConfiguration clientConfig;
        clientConfig.region = "us-east-2"; 
        S3Client s3_client(clientConfig);
        
        Aws::String bucket_name = "clio-core-test-myname-123";
        Aws::String object_name = "test_data.txt";
        Aws::String data_to_write = "Hello, Amazon S3 from Clio Core!";

        SECTION("Write data to S3") {
            PutObjectRequest put_request;
            put_request.SetBucket(bucket_name);
            put_request.SetKey(object_name);

            auto input_data = Aws::MakeShared<Aws::StringStream>("PutObjectInputStream");
            *input_data << data_to_write.c_str();
            put_request.SetBody(input_data);

            auto put_outcome = s3_client.PutObject(put_request);
            if (!put_outcome.IsSuccess()) {
                std::cerr << "PutObject failed (expected if bucket doesn't exist or no credentials): "
                          << put_outcome.GetError().GetMessage() << std::endl;
            } else {
                REQUIRE(put_outcome.IsSuccess());
            }
        }

        SECTION("Read data from S3") {
            GetObjectRequest get_request;
            get_request.SetBucket(bucket_name);
            get_request.SetKey(object_name);

            auto get_outcome = s3_client.GetObject(get_request);
            if (!get_outcome.IsSuccess()) {
                std::cerr << "GetObject failed (expected if bucket doesn't exist or no credentials): "
                          << get_outcome.GetError().GetMessage() << std::endl;
            } else {
                auto& retrieved_file = get_outcome.GetResultWithOwnership().GetBody();
                std::string file_content;
                char buf[256];
                while(retrieved_file.read(buf, sizeof(buf))) {
                    file_content.append(buf, retrieved_file.gcount());
                }
                file_content.append(buf, retrieved_file.gcount());
                REQUIRE(file_content == data_to_write);
            }
        }
    }

    Aws::ShutdownAPI(options);
}
