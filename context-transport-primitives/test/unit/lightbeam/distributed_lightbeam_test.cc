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

#include <arpa/inet.h>
#if CTP_ENABLE_THALLIUM
#include <clio_ctp/lightbeam/thallium_transport.h>
#endif
#include <clio_ctp/lightbeam/transport_factory_impl.h>
#ifndef _WIN32
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <mpi.h>

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

using namespace ctp::lbm;

std::vector<std::string> ReadHosts(const std::string& hostfile) {
  std::vector<std::string> hosts;
  std::ifstream in(hostfile);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) hosts.push_back(line);
  }
  return hosts;
}

TransportType ParseTransport(const std::string& s) {
  if (s == "zeromq") return TransportType::kZeroMq;
#if CTP_ENABLE_THALLIUM
  if (s == "thallium") return TransportType::kThallium;
#endif
  throw std::runtime_error("Unknown transport type: " + s);
}

void Clients(std::vector<std::unique_ptr<ZeroMqTransport>>& clients,
             const std::string& magic) {
  int my_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  std::ostringstream oss;
  oss << std::this_thread::get_id();
  std::cout << "[Rank " << my_rank << "] [Clients] Thread ID: " << oss.str()
            << std::endl;
  for (size_t i = 0; i < clients.size(); ++i) {
    std::cout << "[Rank " << my_rank << "] [Clients] Sending to server " << i
              << std::endl;
    LbmMeta<> meta;
    Bulk bulk = clients[i]->Expose(
        ctp::ipc::FullPtr<char>(const_cast<char*>(magic.data())),
        magic.size(), BULK_XFER);
    meta.send.push_back(bulk);
    int rc = clients[i]->Send(meta);
    std::cout << "[Rank " << my_rank << "] [Clients] Sent to server " << i
              << ", rc=" << rc << std::endl;
    assert(rc == 0);
  }
}

void ServerThread(Transport& server, size_t num_clients,
                  const std::string& magic) {
  std::ostringstream oss;
  oss << std::this_thread::get_id();
  std::cout << "[ServerThread] Thread ID: " << oss.str() << std::endl;
  for (size_t i = 0; i < num_clients; ++i) {
    std::cout << "[Server] Waiting for message " << i << std::endl;

    // Recv with retry loop (does everything - metadata + bulks)
    LbmMeta<> meta;
    int rc;
    while (true) {
      auto info = server.Recv(meta);
      rc = info.rc;
      if (rc == 0) break;
      if (rc != EAGAIN) {
        std::cerr << "[Server] Recv failed with error: " << rc << "\n";
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::cout << "[Server] Received message " << i << ", rc=" << rc
              << std::endl;

    std::string received(meta.recv[0].data.ptr_,
                         meta.recv[0].data.ptr_ + meta.recv[0].size);
    std::cout << "[Server] Received: " << received << std::endl;
    assert(received == magic);
  }
  std::cout << "[ServerThread] Exiting after receiving all messages"
            << std::endl;
}

std::string WaitForServerAddr(const std::string& filename) {
  // Wait for the file to appear
  for (int i = 0; i < 100; ++i) {
    struct stat buffer;
    if (stat(filename.c_str(), &buffer) == 0) {
      std::ifstream in(filename);
      std::string addr;
      std::getline(in, addr);
      return addr;
    }
    usleep(100000);  // 100ms
  }
  throw std::runtime_error("Timeout waiting for server address file: " +
                           filename);
}

std::string GetPrimaryIp() {
  // If LBM_BENCH_DEV is set, prefer that interface (e.g. "enp47s0np0" for
  // the Ares 40 GbE rail). Otherwise fall back to the first non-loopback
  // UP IPv4 address, which on multi-rail hosts is often the 1 GbE
  // management NIC and silently sandbags throughput tests.
  const char *dev_env = std::getenv("LBM_BENCH_DEV");
  std::string preferred_dev = (dev_env && *dev_env) ? dev_env : "";

  struct ifaddrs *ifaddr, *ifa;
  char ip[INET_ADDRSTRLEN];
  std::string preferred_ip;
  std::string fallback_ip;
  getifaddrs(&ifaddr);
  for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
    if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
    if (ifa->ifa_flags & IFF_LOOPBACK) continue;
    if (!(ifa->ifa_flags & IFF_UP)) continue;
    void* addr_ptr = &((struct sockaddr_in*)ifa->ifa_addr)->sin_addr;
    inet_ntop(AF_INET, addr_ptr, ip, INET_ADDRSTRLEN);
    if (!preferred_dev.empty() && ifa->ifa_name &&
        preferred_dev == ifa->ifa_name) {
      preferred_ip = ip;
      break;
    }
    if (fallback_ip.empty()) fallback_ip = ip;
  }
  freeifaddrs(ifaddr);
  return preferred_ip.empty() ? fallback_ip : preferred_ip;
}

void PrintAllInterfaces() {
  struct ifaddrs *ifaddr, *ifa;
  char host[NI_MAXHOST];
  if (getifaddrs(&ifaddr) == -1) {
    perror("getifaddrs");
    return;
  }
  freeifaddrs(ifaddr);
}

int main(int argc, char** argv) {
  // PrintAllInterfaces();
  MPI_Init(&argc, &argv);
  int my_rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int num_msgs = 10;  // default
  int msg_size = 32;  // default small message
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <zeromq|thallium|libfabric> <hostfile> <protocol> <domain> "
                 "<port> [num_msgs] [msg_size]\n";
    std::cerr
        << "All parameters are required except [num_msgs] and [msg_size]. "
           "Number of MPI processes (mpirun -n) should match the number "
           "of hosts in the hostfile."
        << std::endl;
    MPI_Finalize();
    return 1;
  }
  if (argc > 6) num_msgs = std::stoi(argv[6]);
  if (argc > 7) msg_size = std::stoi(argv[7]);
  std::string transport_str = argv[1];
  std::string hostfile = argv[2];
  std::string protocol = argv[3];
  std::string domain = argv[4];
  int port = std::stoi(argv[5]);
  std::string magic(msg_size, 'x');

  TransportType transport = ParseTransport(transport_str);
  std::vector<std::string> hosts = ReadHosts(hostfile);
  if ((int)hosts.size() != world_size) {
    std::cerr << "Error: Number of MPI processes (" << world_size
              << ") does not match number of hosts in hostfile ("
              << hosts.size() << ")." << std::endl;
    MPI_Finalize();
    return 1;
  }

  int my_port = port + my_rank;
  std::string bind_addr = GetPrimaryIp();
  std::string domain_arg = domain;

  // ZeroMqTransport is ROUTER/DEALER only now (the kPushPull alternative
  // has been removed).  Bench just constructs it directly.
  if (my_rank == 0) {
    std::cout << "[Bench] topology=router_dealer" << std::endl;
  }

  // ZMQ path uses ZeroMqTransport's constructor directly; other transports
  // go through the factory. The bench treats clients/servers as base
  // Transport pointers from this point on so the timing / parallel-send
  // / Allgather logic is transport-agnostic.
  std::unique_ptr<Transport> server_owned;
  if (transport == TransportType::kZeroMq) {
    server_owned = std::unique_ptr<Transport>(new ZeroMqTransport(
        TransportMode::kServer, bind_addr, protocol, my_port));
#if CTP_ENABLE_THALLIUM
  } else if (transport == TransportType::kThallium) {
    server_owned = std::unique_ptr<Transport>(new ThalliumTransport(
        TransportMode::kServer, bind_addr, protocol, my_port));
#endif
  } else {
    std::cerr << "Bench: unsupported transport.\n";
    MPI_Finalize();
    return 1;
  }
  Transport *server_ptr = server_owned.get();
  std::string actual_addr = server_ptr->GetAddress();
  std::cout << "[Rank " << my_rank << "] Server address: " << actual_addr
            << ", port: " << my_port << std::endl;
  // Match chimaera's PeerRecvThread polling cadence: chimaera sleeps 1 µs
  // on EAGAIN (busy-poll with backoff), not 1 ms. Env LBM_BENCH_EAGAIN_US
  // overrides for diagnostics.
  long eagain_us = 1;
  if (const char *e = std::getenv("LBM_BENCH_EAGAIN_US")) {
    eagain_us = std::atol(e);
    if (eagain_us < 0) eagain_us = 0;
  }
  // Per-message stdout prints in the hot loop were also eating wall time
  // (PRTE OOB serialises all rank stdout). Suppress unless
  // LBM_BENCH_VERBOSE=1.
  bool verbose = []() {
    const char *v = std::getenv("LBM_BENCH_VERBOSE");
    return v && *v && std::atoi(v) != 0;
  }();

  // LBM_BENCH_SKIP_SELF=1: each rank skips its self-send. Useful for
  // thallium where RPC-to-self via the same engine can deadlock under
  // certain argobots configurations; this matches chimaera's actual
  // usage (self fan-out goes via local lanes, not the transport).
  bool skip_self = []() {
    const char *v = std::getenv("LBM_BENCH_SKIP_SELF");
    return v && *v && std::atoi(v) != 0;
  }();
  int peers_send_to = skip_self ? (world_size - 1) : world_size;
  if (my_rank == 0) {
    std::cout << "[Bench] skip_self=" << (skip_self ? "yes" : "no")
              << " peers_send_to=" << peers_send_to << std::endl;
  }

  // Start timing before any send
  auto global_start = std::chrono::high_resolution_clock::now();
  // Start server thread with num_msgs
  std::thread server_thread([&]() {
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    std::cout << "[ServerThread] Thread ID: " << oss.str() << std::endl;
    int received = 0;
    for (int i = 0; i < num_msgs * peers_send_to; ++i) {
      auto recv_time = std::chrono::high_resolution_clock::now();

      // Recv with retry loop (does everything - metadata + bulks). Match
      // chimaera: 1 µs sleep_for on EAGAIN.
      LbmMeta<> meta;
      int rc;
      while (true) {
        auto info = server_ptr->Recv(meta);
        rc = info.rc;
        if (rc == 0) break;
        if (rc != EAGAIN) {
          std::cerr << "[Server] Recv failed with error: " << rc
                    << "\n";
          return;
        }
        if (eagain_us > 0) {
          std::this_thread::sleep_for(std::chrono::microseconds(eagain_us));
        }
      }
      received++;

      if (verbose) {
        double t =
            std::chrono::duration<double>(recv_time - global_start).count();
        std::cout << "[Rank " << my_rank << "] Received message "
                  << received << " at " << t << " s" << std::endl;
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - global_start).count();
    std::cout << "[Server] Received " << received
              << " messages. Time: " << elapsed << " s" << std::endl;
    std::cout << "[ServerThread] Exiting after receiving all messages"
              << std::endl;
  });

  MPI_Barrier(MPI_COMM_WORLD);
  // Gather all server addresses using MPI_Allgather
  const int addr_len = 256;
  std::vector<char> addr_buf(addr_len, 0);
  strncpy(addr_buf.data(), actual_addr.c_str(), addr_len - 1);
  std::vector<char> all_addrs(world_size * addr_len, 0);
  MPI_Allgather(addr_buf.data(), addr_len, MPI_CHAR, all_addrs.data(), addr_len,
                MPI_CHAR, MPI_COMM_WORLD);
  std::vector<std::string> server_addrs;
  for (int i = 0; i < world_size; ++i) {
    server_addrs.emplace_back(&all_addrs[i * addr_len]);
  }
  auto make_client = [&](const std::string &peer_addr,
                         int target_port) -> std::unique_ptr<Transport> {
    if (transport == TransportType::kZeroMq) {
      return std::unique_ptr<Transport>(new ZeroMqTransport(
          TransportMode::kClient, peer_addr, protocol, target_port));
    }
#if CTP_ENABLE_THALLIUM
    if (transport == TransportType::kThallium) {
      return std::unique_ptr<Transport>(new ThalliumTransport(
          TransportMode::kClient, peer_addr, protocol, target_port));
    }
#endif
    return nullptr;
  };
  std::vector<std::unique_ptr<Transport>> clients;
  for (int i = 0; i < world_size; ++i) {
    int target_port = port + i;
    clients.emplace_back(make_client(server_addrs[i], target_port));
  }
  // LBM_BENCH_PARALLEL_SEND=1: one sender thread per peer (the prior
  //   parallel-send mode; equivalent to LBM_BENCH_THREADS_PER_PEER=1
  //   with parallel_send on).
  // LBM_BENCH_THREADS_PER_PEER=K (default 1): K dedicated PUSH sockets
  //   + K threads per peer (mirrors zmq_ring.py's --threads-per-peer).
  //   Each thread sends num_msgs/K messages so the total per peer-pair
  //   stays constant (num_msgs) and the server's recv expectation
  //   (num_msgs * world_size) doesn't change. Implies parallel_send.
  bool parallel_send = []() {
    const char *v = std::getenv("LBM_BENCH_PARALLEL_SEND");
    return v && *v && std::atoi(v) != 0;
  }();
  int threads_per_peer = 1;
  if (const char *v = std::getenv("LBM_BENCH_THREADS_PER_PEER")) {
    threads_per_peer = std::atoi(v);
    if (threads_per_peer < 1) threads_per_peer = 1;
  }
  if (threads_per_peer > 1) parallel_send = true;
  if (my_rank == 0) {
    std::cout << "[Bench] parallel_send=" << (parallel_send ? "yes" : "no")
              << " threads_per_peer=" << threads_per_peer << std::endl;
  }

  // For threads_per_peer > 1, open additional PUSH sockets per peer so
  // each sender thread has its own ZMQ pipe (the Python smoketest
  // pattern). Each peer's bin has clients[i*tpp + t] for t=0..tpp-1.
  // We construct the extras here; clients[0..world_size-1] from above
  // are reused as the first thread per peer.
  std::vector<std::unique_ptr<Transport>> extra_clients;
  if (threads_per_peer > 1) {
    extra_clients.reserve(world_size * (threads_per_peer - 1));
    for (int i = 0; i < world_size; ++i) {
      int target_port = port + i;
      for (int t = 1; t < threads_per_peer; ++t) {
        extra_clients.emplace_back(make_client(server_addrs[i], target_port));
      }
    }
  }
  auto peer_socket = [&](int peer_idx, int thread_idx) -> Transport * {
    if (thread_idx == 0) return clients[peer_idx].get();
    return extra_clients[peer_idx * (threads_per_peer - 1) +
                          (thread_idx - 1)].get();
  };

  std::atomic<int> sent_total{0};
  if (parallel_send) {
    int msgs_per_thread = std::max(1, num_msgs / threads_per_peer);
    int remainder_thread = num_msgs - (msgs_per_thread * threads_per_peer);
    std::vector<std::thread> sender_threads;
    sender_threads.reserve(static_cast<size_t>(world_size) * threads_per_peer);
    for (int i = 0; i < world_size; ++i) {
      if (skip_self && i == my_rank) continue;
      for (int t = 0; t < threads_per_peer; ++t) {
        // Distribute the remainder messages onto the first few threads
        // of each peer so the total per peer-pair is exactly num_msgs.
        int my_msgs = msgs_per_thread + (t < remainder_thread ? 1 : 0);
        Transport *sock = peer_socket(i, t);
        sender_threads.emplace_back([&, i, t, my_msgs, sock]() {
          for (int m = 0; m < my_msgs; ++m) {
            LbmMeta<> meta;
            Bulk bulk = sock->Expose(
                ctp::ipc::FullPtr<char>(const_cast<char*>(magic.data())),
                magic.size(), BULK_XFER);
            meta.send.push_back(bulk);
            int rc = sock->Send(meta);
            if (rc != 0) {
              std::cerr << "[Rank " << my_rank
                        << "] parallel Send to peer " << i
                        << " thread " << t << " rc=" << rc << "\n";
              return;
            }
            sent_total.fetch_add(1, std::memory_order_relaxed);
          }
        });
      }
    }
    for (auto &th : sender_threads) th.join();
  } else {
    int sent = 0;
    for (int m = 0; m < num_msgs; ++m) {
      for (size_t i = 0; i < clients.size(); ++i) {
        if (skip_self && static_cast<int>(i) == my_rank) continue;
        auto send_time = std::chrono::high_resolution_clock::now();
        LbmMeta<> meta;
        Bulk bulk = clients[i]->Expose(
            ctp::ipc::FullPtr<char>(const_cast<char*>(magic.data())),
            magic.size(), BULK_XFER);
        meta.send.push_back(bulk);
        int rc = clients[i]->Send(meta);
        assert(rc == 0);
        sent++;
        if (verbose) {
          double t =
              std::chrono::duration<double>(send_time - global_start).count();
          std::cout << "[Rank " << my_rank << "] Sent message " << sent
                    << " to server " << i << " at " << t << " s" << std::endl;
        }
      }
    }
    sent_total.store(sent);
  }
  server_thread.join();
  auto global_end = std::chrono::high_resolution_clock::now();
  double global_elapsed =
      std::chrono::duration<double>(global_end - global_start).count();
  std::cout << "[Rank " << my_rank << "] All server messages received!"
            << std::endl;
  std::cout << "[Rank " << my_rank
            << "] Overall runtime (first send to last receive): "
            << global_elapsed << " s" << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}