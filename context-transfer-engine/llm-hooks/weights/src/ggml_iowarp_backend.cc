#include "clio_llm/weights/ggml_iowarp_backend.h"

// ggml internal vtable headers (part of llama.cpp)
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

// Public ggml tensor API
#include "ggml.h"

#include "ggml-impl.h"  // full ggml_cgraph struct for sub-graph views

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// IOWarp buffer context: associates a ggml_backend_buffer with the
// WeightManager that owns its memory.
// ---------------------------------------------------------------------------
struct ggml_iowarp_buffer_context {
    clio_llm::weights::WeightManager* mgr;
    // The base CUDA virtual address returned by GpuVmm.
    char*  base;
    size_t allocated;  // bytes handed out so far (bump allocator)
};

// ---------------------------------------------------------------------------
// Buffer-type vtable
// ---------------------------------------------------------------------------

static const char* ggml_iowarp_buft_get_name(ggml_backend_buffer_type_t) {
    return GGML_BACKEND_IOWARP_NAME;
}

static ggml_backend_buffer_t ggml_iowarp_buft_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    auto* mgr        = reinterpret_cast<clio_llm::weights::WeightManager*>(buft->context);
    char* base       = reinterpret_cast<char*>(mgr->BaseAddress());
    size_t page      = mgr->PageSize();
    size_t aligned_sz = (size + page - 1) & ~(page - 1);

    auto* ctx = new ggml_iowarp_buffer_context{mgr, base, aligned_sz};

    static const ggml_backend_buffer_i iface = {
        // free_buffer
        [](ggml_backend_buffer_t buf) {
            delete reinterpret_cast<ggml_iowarp_buffer_context*>(buf->context);
        },
        // get_base
        [](ggml_backend_buffer_t buf) -> void* {
            return reinterpret_cast<ggml_iowarp_buffer_context*>(buf->context)->base;
        },
        // init_tensor — nothing extra needed; tensor->data set by ggml-alloc
        [](ggml_backend_buffer_t, struct ggml_tensor*) -> enum ggml_status {
            return GGML_STATUS_SUCCESS;
        },
        // memset_tensor — delegate to CUDA memset
        [](ggml_backend_buffer_t, struct ggml_tensor* t, uint8_t v,
                size_t offset, size_t size) {
            cudaMemset(static_cast<char*>(t->data) + offset, v, size);
        },
        // set_tensor — copy host data into GpuVmm pages
        [](ggml_backend_buffer_t buf, struct ggml_tensor* t,
                const void* data, size_t offset, size_t size) {
            auto* ctx    = reinterpret_cast<ggml_iowarp_buffer_context*>(buf->context);
            char*  dst   = static_cast<char*>(t->data) + offset;
            char*  base  = ctx->base;
            size_t pagesz = ctx->mgr->PageSize();
            size_t p0    = (dst - base) / pagesz;
            size_t p1    = (dst - base + size + pagesz - 1) / pagesz;
            for (size_t p = p0; p < p1; ++p) ctx->mgr->Vmm().touchPage(p);
            cudaMemcpy(dst, data, size, cudaMemcpyHostToDevice);
        },
        // get_tensor — copy device data to host
        [](ggml_backend_buffer_t buf, const struct ggml_tensor* t,
                void* data, size_t offset, size_t size) {
            auto* ctx      = reinterpret_cast<ggml_iowarp_buffer_context*>(buf->context);
            const char* src = static_cast<const char*>(t->data) + offset;
            char*  base    = ctx->base;
            size_t pagesz  = ctx->mgr->PageSize();
            size_t p0      = (src - base) / pagesz;
            size_t p1      = (src - base + size + pagesz - 1) / pagesz;
            for (size_t p = p0; p < p1; ++p) ctx->mgr->Vmm().touchPage(p);
            cudaMemcpy(data, src, size, cudaMemcpyDeviceToHost);
        },
        // set_tensor_2d / get_tensor_2d — optional strided copies, not needed
        nullptr,
        nullptr,
        // cpy_tensor — device-to-device copy within the same Vmm space
        [](ggml_backend_buffer_t, const struct ggml_tensor* src,
                struct ggml_tensor* dst) -> bool {
            if (ggml_nbytes(src) != ggml_nbytes(dst)) return false;
            cudaMemcpy(dst->data, src->data, ggml_nbytes(src),
                       cudaMemcpyDeviceToDevice);
            return true;
        },
        // clear
        [](ggml_backend_buffer_t buf, uint8_t v) {
            auto* ctx = reinterpret_cast<ggml_iowarp_buffer_context*>(buf->context);
            cudaMemset(ctx->base, v, ctx->allocated);
        },
        // reset — nothing needed
        [](ggml_backend_buffer_t) {},
    };

    return ggml_backend_buffer_init(buft, iface, ctx, aligned_sz);
}

static size_t ggml_iowarp_buft_get_alignment(ggml_backend_buffer_type_t) {
    // Match CUDA's 256-byte alignment requirement.
    return 256;
}

static size_t ggml_iowarp_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    auto* mgr = reinterpret_cast<clio_llm::weights::WeightManager*>(buft->context);
    return mgr->Vmm().getPageSize() * mgr->Vmm().getTotalPages();
}

static bool ggml_iowarp_buft_is_host(ggml_backend_buffer_type_t) {
    // Memory lives in CUDA virtual address space, not host memory.
    return false;
}

// Shared buffer-type interface (all instances use the same vtable).
static const ggml_backend_buffer_type_i k_iowarp_buft_iface = {
    ggml_iowarp_buft_get_name,
    ggml_iowarp_buft_alloc_buffer,
    ggml_iowarp_buft_get_alignment,
    ggml_iowarp_buft_get_max_size,
    nullptr,  // get_alloc_size: default = ggml_nbytes
    ggml_iowarp_buft_is_host,
};

// ---------------------------------------------------------------------------
// Public API — ggml_backend_iowarp_buffer_type()
// ---------------------------------------------------------------------------
ggml_backend_buffer_type_t ggml_backend_iowarp_buffer_type(
        clio_llm::weights::WeightManager* weight_mgr) {
    // One static buffer type per process.  buft.device is patched once by
    // ggml_backend_iowarp_init() — it must remain valid for the program
    // lifetime (the device is a static singleton, not freed per-context).
    static ggml_backend_buffer_type buft;
    buft.iface   = k_iowarp_buft_iface;
    buft.context = weight_mgr;
    // buft.device patched by ggml_backend_iowarp_init() on first call.
    return &buft;
}

// ---------------------------------------------------------------------------
// IOWarp device context
// ---------------------------------------------------------------------------

struct IOWarpDevCtx {
    clio_llm::weights::WeightManager* mgr;
    ggml_backend_dev_t               cuda_dev;    // global CUDA device (stable)
    ggml_backend_buffer_type_t       iowarp_buft; // our custom GpuVmm buft
};

// ---------------------------------------------------------------------------
// Activation offloading: FlexGen lifecycle.
//
// Phase 2a (per-layer D2H/H2D pipeline):
//   Before execute_range(L) (L>0):  H2D restore l_out-{L-1}: pinned_host → buf[(L-1)%2]
//   After  synchronize(L):          D2H save l_out-{L}: buf[L%2] → pinned_host
//
// Phase 2b (ping-pong GPU buffers, GPU memory reduction):
//   graph_optimize pre-sets l_out-N->data = buf[N%2] BEFORE gallocr allocates
//   compute buffers.  gallocr sees data != NULL and skips those tensors.
//   GPU activation memory: 2 × l_out_size instead of N × l_out_size.
//
// Ping-pong correctness: adjacent layers use different buffers (L%2 ≠ (L+1)%2),
// so execute_range(L+1) reads l_out-{L} from buf[L%2] while writing l_out-{L+1}
// to buf[(L+1)%2] — no data races within a single forward pass.
//
// D2H/H2D persistence: saves l_out to host after compute; restores on next call
// (same data, redundant within a pass but provides cross-call persistence).
// ---------------------------------------------------------------------------

// Persistent pinned host buffer for one layer's l_out activation.
// Grows on demand (capacity >= current activation size); never shrinks.
struct ActivationPinnedSlot {
    void*  host_ptr = nullptr;
    size_t capacity = 0;
    bool   is_saved = false;  // guards against restoring uninitialized memory on first call
};

// Persistent per-context activation offload state (Phase 2a host persistence).
struct ActivationOffloadCtx {
    std::map<int, ActivationPinnedSlot> pinned;  // layer → persistent pinned host buf
};

// Phase 2b: Two ping-pong GPU buffers for all N l_out activation tensors.
// Even layers (0, 2, 4, ...) map to buf[0]; odd layers (1, 3, 5, ...) map to buf[1].
// graph_optimize pre-sets l_out-N->data = buf[N%2] AND l_out-N->buffer = cuda_buf
// before gallocr runs, preventing gallocr from allocating those tensors while
// keeping tensor->buffer valid so ggml-cuda's buffer->context dereferences don't crash.
struct ActivationPingPong {
    void*                 buf[2]   = {nullptr, nullptr};  // halves of cuda_buf
    size_t                capacity = 0;                    // size per half; grow-only
    ggml_backend_buffer_t cuda_buf = nullptr;              // one ggml CUDA buf (2×capacity)
};

// Per-context backend context.
struct IOWarpCtx {
    clio_llm::weights::WeightManager* mgr;
    ggml_backend_t                   cuda_be;       // per-context CUDA compute stream
    ActivationOffloadCtx             act_offload_;  // Phase 2a: host-side persistence
    ActivationPingPong               act_pp_;       // Phase 2b: ping-pong GPU bufs
};

// ---------------------------------------------------------------------------
// Layer-index extraction helper
// ---------------------------------------------------------------------------

// Extract "blk.N." layer index from a graph node by inspecting its source
// tensor names.  Returns -1 if no layer index is found.
static int iowarp_node_layer(const struct ggml_tensor* node) {
    if (!node) return -1;

    // Check source tensors first — weight tensors carry "blk.N." names.
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        if (!node->src[s]) break;
        const char* name = ggml_get_name(node->src[s]);
        if (!name || !name[0]) continue;
        const char* blk = strstr(name, "blk.");
        if (blk) {
            int l = atoi(blk + 4);
            if (l >= 0) return l;
        }
    }

    // Fall back to the node's own name.
    const char* name = ggml_get_name(node);
    if (name && name[0]) {
        const char* blk = strstr(name, "blk.");
        if (blk) {
            int l = atoi(blk + 4);
            if (l >= 0) return l;
        }
    }

    return -1;
}

// ---------------------------------------------------------------------------
// IOWarp device vtable (operates on the singleton device)
// ---------------------------------------------------------------------------

static const char* iowarp_dev_get_name(ggml_backend_dev_t) {
    return GGML_BACKEND_IOWARP_NAME;
}

static const char* iowarp_dev_get_description(ggml_backend_dev_t) {
    return "IOWarp FlexGen weight offloading via GpuVmm demand paging";
}

static void iowarp_dev_get_memory(ggml_backend_dev_t dev,
                                   size_t* free, size_t* total) {
    auto* ctx = static_cast<IOWarpDevCtx*>(dev->context);
    if (ctx->cuda_dev && ctx->cuda_dev->iface.get_memory)
        ctx->cuda_dev->iface.get_memory(ctx->cuda_dev, free, total);
    else { *free = 0; *total = 0; }
}

static enum ggml_backend_dev_type iowarp_dev_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void iowarp_dev_get_props(ggml_backend_dev_t dev,
                                  struct ggml_backend_dev_props* props) {
    auto* ctx = static_cast<IOWarpDevCtx*>(dev->context);
    if (ctx->cuda_dev && ctx->cuda_dev->iface.get_props)
        ctx->cuda_dev->iface.get_props(ctx->cuda_dev, props);
}

static ggml_backend_t iowarp_dev_init_backend(ggml_backend_dev_t dev,
                                               const char* params) {
    // Delegate to the underlying CUDA device.
    auto* ctx = static_cast<IOWarpDevCtx*>(dev->context);
    if (ctx->cuda_dev && ctx->cuda_dev->iface.init_backend)
        return ctx->cuda_dev->iface.init_backend(ctx->cuda_dev, params);
    return nullptr;
}

static ggml_backend_buffer_type_t iowarp_dev_get_buffer_type(
        ggml_backend_dev_t dev) {
    // For scratch / intermediate tensors return the regular CUDA buffer type.
    // Only weight tensors should live in the IOWarp GpuVmm buffer.
    auto* ctx = static_cast<IOWarpDevCtx*>(dev->context);
    if (ctx->cuda_dev && ctx->cuda_dev->iface.get_buffer_type)
        return ctx->cuda_dev->iface.get_buffer_type(ctx->cuda_dev);
    return nullptr;
}

static ggml_backend_buffer_type_t iowarp_dev_get_host_buffer_type(
        ggml_backend_dev_t dev) {
    auto* ctx = static_cast<IOWarpDevCtx*>(dev->context);
    if (ctx->cuda_dev && ctx->cuda_dev->iface.get_host_buffer_type)
        return ctx->cuda_dev->iface.get_host_buffer_type(ctx->cuda_dev);
    return nullptr;
}


static bool iowarp_dev_supports_op(ggml_backend_dev_t dev,
                                    const struct ggml_tensor* op) {
    auto* ctx = static_cast<IOWarpDevCtx*>(dev->context);

    // Claim ALL ops that CUDA supports.  This routes the entire transformer
    // graph to iowarp_be_graph_compute in one call, giving us full visibility
    // of all layers and l_out-N tensors needed for FlexGen activation paging.
    // All ops are executed via ctx->cuda_be, so correctness is unchanged.
    if (ctx->cuda_dev && ctx->cuda_dev->iface.supports_op)
        return ctx->cuda_dev->iface.supports_op(ctx->cuda_dev, op);
    return false;
}

static bool iowarp_dev_supports_buft(ggml_backend_dev_t dev,
                                      ggml_backend_buffer_type_t buft) {
    auto* ctx = static_cast<IOWarpDevCtx*>(dev->context);

    // Accept our IOWarp GpuVmm buffer type explicitly.
    if (buft == ctx->iowarp_buft) return true;

    // Also accept whatever the CUDA backend accepts (for intermediate tensors,
    // KV cache buffers, etc.) so the scheduler routes mixed ops here.
    if (ctx->cuda_dev && ctx->cuda_dev->iface.supports_buft)
        return ctx->cuda_dev->iface.supports_buft(ctx->cuda_dev, buft);

    return false;
}

static bool iowarp_dev_offload_op(ggml_backend_dev_t dev,
                                   const struct ggml_tensor* op) {
    auto* ctx = static_cast<IOWarpDevCtx*>(dev->context);
    if (ctx->cuda_dev && ctx->cuda_dev->iface.offload_op)
        return ctx->cuda_dev->iface.offload_op(ctx->cuda_dev, op);
    return false;
}

static ggml_backend_event_t iowarp_dev_event_new(ggml_backend_dev_t dev) {
    auto* ctx = static_cast<IOWarpDevCtx*>(dev->context);
    if (ctx->cuda_dev && ctx->cuda_dev->iface.event_new)
        return ctx->cuda_dev->iface.event_new(ctx->cuda_dev);
    return nullptr;
}

static void iowarp_dev_event_free(ggml_backend_dev_t dev,
                                   ggml_backend_event_t event) {
    auto* ctx = static_cast<IOWarpDevCtx*>(dev->context);
    if (ctx->cuda_dev && ctx->cuda_dev->iface.event_free)
        ctx->cuda_dev->iface.event_free(ctx->cuda_dev, event);
}

static void iowarp_dev_event_synchronize(ggml_backend_dev_t dev,
                                          ggml_backend_event_t event) {
    auto* ctx = static_cast<IOWarpDevCtx*>(dev->context);
    if (ctx->cuda_dev && ctx->cuda_dev->iface.event_synchronize)
        ctx->cuda_dev->iface.event_synchronize(ctx->cuda_dev, event);
}

// ---------------------------------------------------------------------------
// IOWarp backend vtable (per-context)
// ---------------------------------------------------------------------------

static const char* iowarp_be_get_name(ggml_backend_t) {
    return GGML_BACKEND_IOWARP_NAME;
}

// Free the per-context backend.
// The device (iowarp_dev singleton) is NOT freed here — it must outlive all
// buffers that have buft->device pointing to it.
static void iowarp_be_free(ggml_backend_t be) {
    auto* ctx = static_cast<IOWarpCtx*>(be->context);
    // Release pinned host buffers (Phase 2a host persistence).
    for (auto& [layer, slot] : ctx->act_offload_.pinned) {
        if (slot.host_ptr) cudaFreeHost(slot.host_ptr);
    }
    // Release ping-pong GPU buffer (Phase 2b).  buf[0/1] are halves of cuda_buf.
    if (ctx->act_pp_.cuda_buf) ggml_backend_buffer_free(ctx->act_pp_.cuda_buf);
    ggml_backend_free(ctx->cuda_be);  // free per-context CUDA stream
    delete ctx;
    delete be;
    // NOTE: be->device (the singleton IOWarp device) is intentionally NOT freed.
}

static void iowarp_be_set_tensor_async(ggml_backend_t be,
        struct ggml_tensor* t, const void* data, size_t offset, size_t size) {
    auto* ctx = static_cast<IOWarpCtx*>(be->context);
    if (ctx->cuda_be->iface.set_tensor_async)
        ctx->cuda_be->iface.set_tensor_async(ctx->cuda_be, t, data, offset, size);
}

static void iowarp_be_get_tensor_async(ggml_backend_t be,
        const struct ggml_tensor* t, void* data, size_t offset, size_t size) {
    auto* ctx = static_cast<IOWarpCtx*>(be->context);
    if (ctx->cuda_be->iface.get_tensor_async)
        ctx->cuda_be->iface.get_tensor_async(ctx->cuda_be, t, data, offset, size);
}

static void iowarp_be_synchronize(ggml_backend_t be) {
    auto* ctx = static_cast<IOWarpCtx*>(be->context);
    ggml_backend_synchronize(ctx->cuda_be);
}

// Full FlexGen graph compute: weight paging + activation lifecycle.
//
// Weight paging (double-buffering pipeline with deferred release):
//   Before loop: PrepareLayer(0) + syncTransfer() — prime pipeline.
//   Per layer L (iteration i):
//     1. execute_range(L)       — queue CUDA kernels [async]
//     2. PrepareLayer(L+1)      — H2D next-layer weights [overlaps step 1]
//     3. synchronize(cuda_be)   — wait for L's kernels
//     4. syncTransfer()         — wait for L+1's H2D
//     5. ReleaseLayer(L-1)      — deferred async D2H evict for PREVIOUS layer
//                                 [overlaps L+1 compute in next iteration]
//   After loop: ReleaseLayer(last).
//
// WHY DEFERRED RELEASE: adjacent layer page ranges overlap (e.g. layer 0 [0,6)
// and layer 1 [5,11) share page 5).  PrepareLayer(L+1) maps the shared page for
// layer L+1's data.  If we called ReleaseLayer(L) immediately after, it would
// UNMAP the shared page, crashing execute_range(L+1).  By deferring the release
// to iteration L+1 (after L+1's compute), the shared page remains valid.
//
// Activation lifecycle (Phase 2a + 2b):
//   Phase 2b: iowarp_be_graph_optimize pre-set l_out-N->data = buf[N%2] BEFORE
//     gallocr ran → gallocr skipped l_out in compute buffer allocation.
//     orig_data (captured in pre-pass) = buf[L%2] (the ping-pong buffer).
//     GPU activation memory: 2 × l_out_size instead of N × l_out_size.
//
//   Phase 2a D2H/H2D (host persistence):
//     Before execute_range(L) (L>0):
//       H2D restore l_out-{L-1}: pinned_host → buf[(L-1)%2]  (sync cudaMemcpy)
//       Within a single forward pass this is redundant (buf holds current data),
//       but provides cross-call persistence for future use.
//     After synchronize(L):
//       D2H save l_out-{L}: buf[L%2] → pinned_host            (sync)
//
// supports_op claims ALL CUDA ops → full transformer graph in one call.
// Per-call pre-pass scans l_out-N nodes directly (handles graph rebuilds).
static enum ggml_status iowarp_be_graph_compute(ggml_backend_t be,
                                                  struct ggml_cgraph* gf) {
    auto* ctx = static_cast<IOWarpCtx*>(be->context);
    auto* mgr = ctx->mgr;

    // If weight manager is not ready, delegate to CUDA directly.
    if (!mgr || !mgr->IsReady()) {
        return ggml_backend_graph_compute(ctx->cuda_be, gf);
    }

    int n = ggml_graph_n_nodes(gf);

    // -----------------------------------------------------------------------
    // Per-call pre-pass: scan ALL nodes for "l_out-N" tensors.
    // Done every call because the graph may be rebuilt (e.g. prefill→decode
    // transition changes batch size, creating new tensor pointers).
    // -----------------------------------------------------------------------
    struct ActNode {
        struct ggml_tensor* node;
        size_t              size;
        void*               orig_data;  // gallocr GPU address (where compute writes)
    };
    std::map<int, ActNode> act_nodes;

    for (int i = 0; i < n; ++i) {
        struct ggml_tensor* node = ggml_graph_node(gf, i);
        const char* name = ggml_get_name(node);
        // llama-context.cpp:graph_get_cb uses "%s-%d" (dash separator)
        if (!name || strncmp(name, "l_out-", 6) != 0) continue;
        int L = atoi(name + 6);
        if (L < 0) continue;
        act_nodes[L] = {node, ggml_nbytes(node), node->data};
    }

    // Ensure pinned host buffers (persistent, grow-only).
    for (auto& [L, an] : act_nodes) {
        auto& ps = ctx->act_offload_.pinned[L];
        if (!ps.host_ptr || an.size > ps.capacity) {
            if (ps.host_ptr) cudaFreeHost(ps.host_ptr);
            if (cudaMallocHost(&ps.host_ptr, an.size) != cudaSuccess) {
                fprintf(stderr, "IOWarp: WARNING — cudaMallocHost failed for l_out-%d\n", L);
                ps.host_ptr = nullptr;
            }
            ps.capacity = an.size;
            // is_saved intentionally NOT reset — previous data still valid if no realloc
        }
    }

    // -----------------------------------------------------------------------
    // Find first node index per blk.N layer (for weight-paging boundaries).
    // -----------------------------------------------------------------------
    std::map<int, int> first_node;
    for (int i = 0; i < n; ++i) {
        int layer = iowarp_node_layer(ggml_graph_node(gf, i));
        if (layer >= 0 && first_node.find(layer) == first_node.end())
            first_node[layer] = i;
    }

    // No blk. layers → delegate entirely to CUDA (e.g., embedding-only graphs).
    if (first_node.empty()) {
        return ggml_backend_graph_compute(ctx->cuda_be, gf);
    }

    // Ordered layer list (sorted by std::map).
    std::vector<int> layers;
    layers.reserve(first_node.size());
    for (auto& [l, _] : first_node) layers.push_back(l);

    // Execute nodes [s, e) as a CUDA sub-graph view (borrows parent's nodes[]).
    auto execute_range = [&](int s, int e) -> enum ggml_status {
        if (s >= e) return GGML_STATUS_SUCCESS;
        struct ggml_cgraph sub = *gf;
        sub.nodes   = ggml_graph_nodes(gf) + s;
        sub.n_nodes = e - s;
        sub.n_leafs = 0;
        return ggml_backend_graph_compute(ctx->cuda_be, &sub);
    };

    // -----------------------------------------------------------------------
    // Preamble [0, first_layer): embeddings / positional encodings.
    // Non-blk. weight pages are always mapped; no PrepareLayer needed.
    // -----------------------------------------------------------------------
    {
        enum ggml_status st = execute_range(0, first_node[layers[0]]);
        if (st != GGML_STATUS_SUCCESS) return st;
    }

    // Prime the pipeline: map layer 0's pages before the loop starts.
    mgr->PrepareLayer(layers[0]);
    mgr->Vmm().syncTransfer();  // wait until layer 0's pages are on-GPU

    // -----------------------------------------------------------------------
    // Per-layer FlexGen loop: weight paging + activation lifecycle.
    // -----------------------------------------------------------------------
    for (size_t i = 0; i < layers.size(); ++i) {
        int layer = layers[i];
        int seg_s = first_node[layer];
        int seg_e = (i + 1 < layers.size()) ? first_node[layers[i + 1]] : n;

        // --- Activation Step 0: H2D restore to orig_data (L > 0 only) ---
        // Restore l_out-{L-1} from pinned host directly into orig_data.
        // No pointer patching: orig_data is valid CUDA memory throughout the call
        // (ReleaseLayer only unmaps GpuVmm weight pages, not compute tensors).
        // Direct restore avoids CUDA graph exec-update issues (USE_GRAPHS=1).
        // Guard: skip on the first-ever call (is_saved=false → no stale data).
        int prev = layer - 1;
        if (layer > 0 && act_nodes.count(prev)) {
            auto& an_prev = act_nodes[prev];
            auto& ps_prev = ctx->act_offload_.pinned[prev];
            if (ps_prev.host_ptr && ps_prev.is_saved) {
                cudaMemcpy(an_prev.orig_data, ps_prev.host_ptr, an_prev.size,
                           cudaMemcpyHostToDevice);
                fprintf(stderr,
                        "IOWarp: act FlexGen restore l_out-%d from host "
                        "(%.1f KiB) -> orig_data\n",
                        prev, an_prev.size / 1024.0f);
            }
        }

        // --- Step 1: Queue this layer's CUDA kernels (async) ---
        // Weight pages already mapped from PrepareLayer (primed or from prev iter).
        // Reads l_out-{L-1} from orig_data (restored in Step 0 if is_saved).
        enum ggml_status st = execute_range(seg_s, seg_e);

        // --- Step 2: Prefetch next layer's weights (overlaps with step 1) ---
        if (i + 1 < layers.size()) {
            mgr->PrepareLayer(layers[i + 1]);
        }

        // --- Step 3: Wait for this layer's kernels ---
        // CRITICAL before evicting pages or reading l_out-{L}.
        ggml_backend_synchronize(ctx->cuda_be);

        // --- Step 4: Wait for next layer's weight H2D ---
        if (i + 1 < layers.size()) {
            mgr->Vmm().syncTransfer();
        }

        // --- Activation Step 4b: D2H save l_out-{L} ---
        // l_out-{L} is now stable at orig_data (gallocr address, synchronize done).
        // Save before execute_range(L+1) could reuse the gallocr slot.
        if (act_nodes.count(layer)) {
            auto& an = act_nodes[layer];
            auto& ps = ctx->act_offload_.pinned[layer];
            if (ps.host_ptr) {
                cudaMemcpy(ps.host_ptr, an.orig_data, an.size,
                           cudaMemcpyDeviceToHost);
                ps.is_saved = true;
                fprintf(stderr,
                        "IOWarp: act FlexGen save l_out-%d to host "
                        "(%.1f KiB)\n",
                        layer, an.size / 1024.0f);
            }
        }

        // --- Step 5: Deferred async-evict of the PREVIOUS layer's weight pages ---
        // IMPORTANT: Adjacent layer page ranges overlap (e.g. layer 0 [0,6) and
        // layer 1 [5,11) both include page 5).  If we released layer L in the same
        // iteration that PrepareLayer(L+1) ran, we would unmap the shared page
        // AFTER PrepareLayer(L+1) already loaded layer L+1's data into it, leaving
        // layer L+1's execute_range with an unmapped page → illegal access.
        //
        // Solution: release layer L-1 here (after layer L's compute+sync), so the
        // shared page is only unmapped once layer L is fully done.
        // The D2H eviction of layer L-1 now overlaps with layer L+1's compute.
        if (i > 0) {
            mgr->ReleaseLayer(layers[i - 1]);
        }

        if (st != GGML_STATUS_SUCCESS) return st;
    }

    // Release the final layer's pages (its deferred release was never triggered
    // because there is no iteration i = layers.size()).
    if (!layers.empty()) {
        mgr->ReleaseLayer(layers.back());
    }

    return GGML_STATUS_SUCCESS;
}

// Phase 2b: Pre-set l_out-N tensor data AND buffer pointers to ping-pong GPU
// buffers BEFORE gallocr runs.  Called from ggml_backend_sched_split_graph.
//
// ggml_gallocr_is_allocated (ggml-alloc.c) returns true when t->data != NULL,
// causing ggml_gallocr_allocate_node to skip those tensors.  This prevents gallocr
// from including l_out in the compute buffer, reducing GPU activation memory from
// N × l_out_size to 2 × l_out_size (ping-pong, one buffer per layer parity).
//
// WHY we also set tensor->buffer: ggml-cuda's mul_mat dispatch (ggml-cuda.cu)
// accesses src1->buffer->context->device for device selection.  Setting
// tensor->buffer to a valid CUDA buffer (cuda_buf) prevents NULL dereference.
//
// Ping-pong rule: l_out-L uses buf[L%2].  Adjacent layers use different buffers,
// so execute_range(L+1) reads l_out-{L} from buf[L%2] while writing l_out-{L+1}
// to buf[(L+1)%2] — no data races within a single forward pass.
//
// CUDA graph stability: buf[0] and buf[1] are halves of a single stable allocation.
// Their addresses never change → CUDA graph exec-update succeeds across calls.
static void iowarp_be_graph_optimize(ggml_backend_t be,
                                      struct ggml_cgraph* gf) {
    auto* ctx = static_cast<IOWarpCtx*>(be->context);
    int n = ggml_graph_n_nodes(gf);

    // First pass: find the maximum l_out activation size.
    size_t max_sz = 0;
    for (int i = 0; i < n; ++i) {
        struct ggml_tensor* node = ggml_graph_node(gf, i);
        const char* name = ggml_get_name(node);
        if (!name || strncmp(name, "l_out-", 6) != 0) continue;
        int L = atoi(name + 6);
        if (L < 0) continue;
        max_sz = std::max(max_sz, ggml_nbytes(node));
    }

    if (max_sz == 0) return;  // no l_out tensors in this graph split

    // Grow ping-pong buffers if needed (grow-only; never shrink).
    // Allocate via ggml's CUDA buffer API so tensor->buffer is a valid CUDA buffer.
    // ggml-cuda accesses tensor->buffer->context->device during mul_mat dispatch;
    // a NULL buffer causes a segfault.  Using ggml_backend_buft_alloc_buffer keeps
    // the buffer context valid (device=0, dev_ptr=base).
    if (max_sz > ctx->act_pp_.capacity) {
        if (ctx->act_pp_.cuda_buf) {
            ggml_backend_buffer_free(ctx->act_pp_.cuda_buf);
            ctx->act_pp_.cuda_buf = nullptr;
            ctx->act_pp_.buf[0]   = nullptr;
            ctx->act_pp_.buf[1]   = nullptr;
        }
        // Allocate a single CUDA buffer covering both ping and pong halves.
        ggml_backend_buffer_type_t cuda_buft = ggml_backend_cuda_buffer_type(0);
        ctx->act_pp_.cuda_buf = ggml_backend_buft_alloc_buffer(cuda_buft, max_sz * 2);
        if (!ctx->act_pp_.cuda_buf) {
            fprintf(stderr,
                    "IOWarp: WARNING — ping-pong CUDA buf alloc failed (%.1f KiB × 2)\n",
                    max_sz / 1024.0f);
            ctx->act_pp_.capacity = 0;
            return;
        }
        void* base = ggml_backend_buffer_get_base(ctx->act_pp_.cuda_buf);
        ctx->act_pp_.buf[0]   = base;
        ctx->act_pp_.buf[1]   = static_cast<char*>(base) + max_sz;
        ctx->act_pp_.capacity = max_sz;
        fprintf(stderr,
                "IOWarp: act Phase2b — ping-pong CUDA bufs allocated "
                "(%.1f KiB each × 2 = %.1f KiB GPU)\n",
                max_sz / 1024.0f, 2.0f * max_sz / 1024.0f);
    }

    // Second pass: pre-set each l_out-N tensor's data AND buffer pointers.
    // gallocr (ggml_gallocr_is_allocated) returns true for data != NULL → skips them.
    // Setting buffer keeps tensor->buffer->context valid for ggml-cuda dispatch.
    for (int i = 0; i < n; ++i) {
        struct ggml_tensor* node = ggml_graph_node(gf, i);
        const char* name = ggml_get_name(node);
        if (!name || strncmp(name, "l_out-", 6) != 0) continue;
        int L = atoi(name + 6);
        if (L < 0) continue;
        node->data   = ctx->act_pp_.buf[L % 2];
        node->buffer = ctx->act_pp_.cuda_buf;  // valid CUDA buffer context
    }
}

static void iowarp_be_event_record(ggml_backend_t be,
                                    ggml_backend_event_t event) {
    auto* ctx = static_cast<IOWarpCtx*>(be->context);
    if (ctx->cuda_be->iface.event_record)
        ctx->cuda_be->iface.event_record(ctx->cuda_be, event);
}

static void iowarp_be_event_wait(ggml_backend_t be,
                                  ggml_backend_event_t event) {
    auto* ctx = static_cast<IOWarpCtx*>(be->context);
    if (ctx->cuda_be->iface.event_wait)
        ctx->cuda_be->iface.event_wait(ctx->cuda_be, event);
}

// ---------------------------------------------------------------------------
// Singleton IOWarp device — created once, lives for the program lifetime.
// Must not be freed because buft->device points to it and buft is static.
// ---------------------------------------------------------------------------

// Device vtable (static, all per-device functions delegate via the ctx).
static const ggml_backend_device_i k_iowarp_dev_iface = {
    /* .get_name              = */ iowarp_dev_get_name,
    /* .get_description       = */ iowarp_dev_get_description,
    /* .get_memory            = */ iowarp_dev_get_memory,
    /* .get_type              = */ iowarp_dev_get_type,
    /* .get_props             = */ iowarp_dev_get_props,
    /* .init_backend          = */ iowarp_dev_init_backend,
    /* .get_buffer_type       = */ iowarp_dev_get_buffer_type,
    /* .get_host_buffer_type  = */ iowarp_dev_get_host_buffer_type,
    /* .buffer_from_host_ptr  = */ nullptr,
    /* .supports_op           = */ iowarp_dev_supports_op,
    /* .supports_buft         = */ iowarp_dev_supports_buft,
    /* .offload_op            = */ iowarp_dev_offload_op,
    /* .event_new             = */ iowarp_dev_event_new,
    /* .event_free            = */ iowarp_dev_event_free,
    /* .event_synchronize     = */ iowarp_dev_event_synchronize,
};

// The singleton device and its context.  Allocated on first
// ggml_backend_iowarp_init() call and never freed.
static ggml_backend_device* s_iowarp_dev     = nullptr;
static IOWarpDevCtx*         s_iowarp_dev_ctx = nullptr;

// ---------------------------------------------------------------------------
// Backend vtable (static).
// ---------------------------------------------------------------------------
static const ggml_backend_i k_iowarp_be_iface = {
    /* .get_name           = */ iowarp_be_get_name,
    /* .free               = */ iowarp_be_free,
    /* .set_tensor_async   = */ iowarp_be_set_tensor_async,
    /* .get_tensor_async   = */ iowarp_be_get_tensor_async,
    /* .set_tensor_2d_async= */ nullptr,  // optional strided copies
    /* .get_tensor_2d_async= */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,  // fall back to sync copy
    /* .synchronize        = */ iowarp_be_synchronize,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ iowarp_be_graph_compute,
    /* .event_record       = */ iowarp_be_event_record,
    /* .event_wait         = */ iowarp_be_event_wait,
    /* .graph_optimize     = */ iowarp_be_graph_optimize,
};

// Unique GUID bytes for the IOWarp-GpuVmm backend ("IOWarp-GpuVmm-01")
static ggml_guid k_iowarp_guid = {
    0x49, 0x4f, 0x57, 0x61, 0x72, 0x70, 0x2d,
    0x47, 0x70, 0x75, 0x56, 0x6d, 0x6d, 0x2d,
    0x30, 0x31,
};

// ---------------------------------------------------------------------------
// Public API — ggml_backend_iowarp_init()
// ---------------------------------------------------------------------------

ggml_backend_t ggml_backend_iowarp_init(
        clio_llm::weights::WeightManager* weight_mgr) {
    // Create a per-context CUDA backend for actual CUDA kernel execution.
    ggml_backend_t cuda_be = ggml_backend_cuda_init(0);
    if (!cuda_be) {
        fprintf(stderr, "IOWarp: failed to init CUDA backend\n");
        return nullptr;
    }

    // The CUDA device pointer is stable (from the global CUDA registration).
    ggml_backend_dev_t cuda_dev = ggml_backend_get_device(cuda_be);

    ggml_backend_buffer_type_t iowarp_buft = ggml_backend_iowarp_buffer_type(weight_mgr);

    // Create the singleton IOWarp device on the first call.
    // It must outlive all buffers allocated with the IOWarp buft, so we
    // never free it.  Subsequent calls update cuda_dev in the context so
    // device-level queries (get_memory, supports_op, etc.) stay current.
    if (!s_iowarp_dev) {
        s_iowarp_dev_ctx = new IOWarpDevCtx{weight_mgr, cuda_dev, iowarp_buft};
        s_iowarp_dev     = new ggml_backend_device{k_iowarp_dev_iface,
                                                   nullptr,  // no reg
                                                   s_iowarp_dev_ctx};
        // Patch buft->device once.  After this call, any access to
        // buft->device (e.g. ggml_backend_dev_host_buffer_type in load_tensors)
        // will use this stable singleton pointer.
        iowarp_buft->device = s_iowarp_dev;

        fprintf(stderr, "IOWarp: singleton device created, buft->device patched\n");
    } else {
        // Update the CUDA device in case there's been a device reset (rare).
        s_iowarp_dev_ctx->cuda_dev = cuda_dev;
    }

    // Create a per-context backend that wraps its own CUDA compute stream.
    auto* be_ctx = new IOWarpCtx{weight_mgr, cuda_be};
    auto* be     = new ggml_backend{};
    be->guid     = &k_iowarp_guid;
    be->iface    = k_iowarp_be_iface;
    be->device   = s_iowarp_dev;   // singleton device
    be->context  = be_ctx;

    fprintf(stderr,
            "IOWarp: weight backend created (per-context CUDA stream + "
            "singleton GpuVmm device)\n");
    return be;
}

// ---------------------------------------------------------------------------
// AutoRegisterLayers
// ---------------------------------------------------------------------------

// Extract layer index from a tensor name containing "blk.N." (returns -1
// if no such pattern is found).
static int iowarp_extract_layer(const char* name) {
    if (!name || name[0] == '\0') return -1;
    const char* blk = strstr(name, "blk.");
    if (!blk) return -1;
    int layer = atoi(blk + 4);
    return (layer >= 0) ? layer : -1;
}

void ggml_backend_iowarp_auto_register_layers(
        clio_llm::weights::WeightManager* mgr,
        struct ggml_context*             ctx) {
    if (!mgr || !ctx) return;

    auto&        vmm   = mgr->Vmm();
    const char*  base  = reinterpret_cast<const char*>(vmm.getBasePtr());
    size_t       psz   = vmm.getPageSize();
    size_t       total = psz * vmm.getTotalPages();

    // layer_idx → { min_page, max_page }  (exclusive upper bound)
    std::map<int, std::pair<size_t, size_t>> layer_pages;

    for (struct ggml_tensor* t = ggml_get_first_tensor(ctx);
            t != nullptr;
            t = ggml_get_next_tensor(ctx, t)) {
        if (!t->data) continue;

        const char* p = static_cast<const char*>(t->data);
        if (p < base || p >= base + total) continue;  // not in our VMM

        int layer = iowarp_extract_layer(ggml_get_name(t));
        if (layer < 0) continue;

        size_t offset = static_cast<size_t>(p - base);
        size_t sz     = ggml_nbytes(t);
        size_t p0     = offset / psz;
        size_t p1     = (offset + sz + psz - 1) / psz;

        auto& r = layer_pages[layer];
        if (r.first == 0 && r.second == 0) {
            r = {p0, p1};
        } else {
            r.first  = std::min(r.first,  p0);
            r.second = std::max(r.second, p1);
        }
    }

    for (auto& [layer, r] : layer_pages) {
        size_t page_count = r.second - r.first;
        mgr->RegisterLayer(layer, r.first, page_count);
        fprintf(stderr,
                "IOWarp weights: layer %3d -> pages [%zu, %zu)  %.1f MiB\n",
                layer, r.first, r.second,
                page_count * psz / (1024.0 * 1024.0));
    }

    fprintf(stderr,
            "IOWarp weights: AutoRegisterLayers complete -- %zu layers registered\n",
            layer_pages.size());
}
