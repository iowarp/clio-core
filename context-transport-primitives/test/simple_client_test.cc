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

#include "hermes_shm/lightbeam/lightbeam.h"
#include "hermes_shm/util/timer.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <cstring>
#include <cereal/types/string.hpp>
#include "hermes_shm/lightbeam/thallium/client.h"

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <ip_address> <protocol>" << std::endl;
    std::cout << "  ip_address: IP address of the server (e.g., 127.0.0.1)" << std::endl;
    std::cout << "  protocol:   'zmq' (or 'tcp') or 'thallium'" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " 127.0.0.1 zmq" << std::endl;
    std::cout << "  " << program_name << " 192.168.1.100 thallium" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Simple LightBeam Round-Trip Test" << std::endl;
    std::cout << "=================================" << std::endl;
    
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string ip_address = argv[1];
    std::string protocol = argv[2];
    
    // Validate IP address format (basic check)
    if (ip_address.empty() || ip_address.find_first_not_of("0123456789.") != std::string::npos) {
        std::cerr << "❌ ERROR: Invalid IP address format: " << ip_address << std::endl;
        return 1;
    }
    
    // Determine URL (both zmq and libfabric use tcp transport in our implementation)
    std::string url;
    int rdma_flags = 0;
    
    if (protocol == "zmq" || protocol == "tcp") {
        url = "tcp://" + ip_address + ":9413";
    } else if (protocol == "thallium") {
        url = "tcp://" + ip_address + ":5557";
    } else {
        std::cerr << "❌ ERROR: Invalid protocol: " << protocol << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    std::cout << "Protocol: " << protocol << std::endl;
    std::cout << "Server URL: " << url << std::endl;
    
    // Create client using the selected backend
    if (protocol == "thallium") {
        hshm::lbm::thallium::Client client;
        std::cout << "Connecting to Thallium server..." << std::endl;
        client.Connect(url);
        std::string message = "Hello Thallium!";
        std::cout << "Sending echo message: '" << message << "'..." << std::endl;
        std::string response = client.SyncCall<std::string>(url, "echo", message);
        std::cout << "Received from server: '" << response << "'" << std::endl;
        if (response == message) {
            std::cout << "✅ PASS: Thallium round-trip successful" << std::endl;
        } else {
            std::cout << "❌ FAIL: Thallium round-trip failed" << std::endl;
        }
    } else {
        hshm::lbm::Client client;
        std::cout << "Connecting to ZMQ server..." << std::endl;
        client.Connect(url);
        // Add your ZMQ test logic here (existing logic)
        std::cout << "✅ ZMQ client connected successfully to " << url << std::endl;
        // Prepare test message
        std::string message = "Hello Server - Round Trip Test";
        auto bulk = client.Expose(url, message.c_str(), message.length(), rdma_flags);
        char recv_buffer[1024];
        
        // Start timing the complete round-trip
        std::cout << "Starting round-trip test..." << std::endl;
        
        hshm::Timer timer;
        timer.Resume();
        
        // Send message (async)
        auto send_event = client.Send(bulk);
        if (!send_event) {
            std::cerr << "❌ ERROR: Failed to initiate send" << std::endl;
            return 1;
        }
        
        // Start receive (async)  
        auto recv_event = client.Recv(recv_buffer, sizeof(recv_buffer), url);
        if (!recv_event) {
            std::cerr << "❌ ERROR: Failed to initiate receive" << std::endl;
            return 1;
        }
        
        // Wait for both operations to complete
        bool completed = false;
        auto start = std::chrono::steady_clock::now();
        const double timeout_seconds = 5.0;
        
        std::cout << "Waiting for completion..." << std::endl;
        
        while (!completed) {
            client.ProcessCompletions();
            
            // Check if both operations are done
            if (send_event->is_done && recv_event->is_done) {
                completed = true;
                timer.Pause();
                break;
            }
            
            // Check for timeout
            auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (elapsed > timeout_seconds) {
                timer.Pause();
                std::cerr << "❌ TIMEOUT: Operations did not complete within " << timeout_seconds << " seconds" << std::endl;
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        // Get round-trip time
        double round_trip_time = timer.GetMsecFromStart();
        
        // Display results
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Round-trip time: " << round_trip_time << " ms" << std::endl;
        std::cout << "Send completed: " << (send_event->is_done ? "YES" : "NO") << std::endl;
        std::cout << "Recv completed: " << (recv_event->is_done ? "YES" : "NO") << std::endl;
        
        if (send_event->is_done && recv_event->is_done) {
            std::cout << "Send bytes: " << send_event->bytes_transferred << std::endl;
            std::cout << "Recv bytes: " << recv_event->bytes_transferred << std::endl;
            std::cout << "Send error: " << send_event->error_code << std::endl;
            std::cout << "Recv error: " << recv_event->error_code << std::endl;
            
            if (send_event->error_code == 0 && recv_event->error_code == 0) {
                // Null-terminate received data for display
                size_t recv_bytes = std::min(recv_event->bytes_transferred, sizeof(recv_buffer) - 1);
                recv_buffer[recv_bytes] = '\0';
                
                std::cout << "Received message: \"" << recv_buffer << "\"" << std::endl;
                
                if (round_trip_time < 1000.0) { // Less than 1 second
                    std::cout << "✅ PASS: Round-trip successful and fast (" << round_trip_time << " ms)" << std::endl;
                } else {
                    std::cout << "⚠️  PASS: Round-trip successful but slow (" << round_trip_time << " ms)" << std::endl;
                }
            } else {
                std::cout << "❌ FAIL: Operations completed with errors" << std::endl;
                if (send_event->error_code != 0) {
                    std::cout << "Send error: " << send_event->error_message << std::endl;
                }
                if (recv_event->error_code != 0) {
                    std::cout << "Recv error: " << recv_event->error_message << std::endl;
                }
            }
        } else {
            std::cout << "❌ FAIL: Operations did not complete" << std::endl;
        }
    }
    
    // Disconnect
    std::cout << "\nDisconnecting..." << std::endl;
    // The client object is local to the if/else block, so it will be destroyed.
    // No explicit disconnect needed here unless the client object itself manages a global state.
    
    std::cout << "Test completed." << std::endl;
    return 0;
}