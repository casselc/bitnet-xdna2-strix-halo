/* Same-process concurrent-context safety for the XDNA runtime.
 *
 * The direct-output path publishes PROCESS-GLOBAL state in accumulate() --
 * g_direct (which output slots hold this tensor's results), g_acc, g_cur_shape
 * -- and then returns. The scaling epilogue reads that state afterwards, on
 * every ggml worker thread, holding no lock.
 *
 * Within one graph execution the ggml barrier orders those two phases. Across
 * two independent inference contexts in the same process it orders nothing:
 * context B can enter accumulate and overwrite g_direct, and reuse the same
 * output slots, while context A's workers are still reading them.
 *
 * This harness reproduces exactly that schedule without needing two llama
 * contexts. Each worker thread plays the role of one context's driver:
 *
 *      accumulate(shape, T)      <- publishes global plan
 *      (small gap)               <- where the other context interleaves
 *      epilogue(...)             <- consumes the plan it expects to still own
 *
 * Driving the runtime API directly is both more aggressive than two llama
 * contexts (the gap is tighter and the shapes alternate every iteration) and
 * deterministic to run, so it is the primary evidence. Results are compared
 * against a reference captured single-threaded beforehand; the existing
 * test_xdna_shapes proves that reference is bit-exact against a scalar oracle.
 *
 * Two modes:
 *   unleased  -- current raw API. REPORT ONLY; a race may or may not fire, so
 *                this never fails the suite. It exists to measure the hazard.
 *   leased    -- bitnet_xdna_invocation_begin/end around the whole lifetime.
 *                MUST be exact. This is the assertion.
 */
#include "../runtime/bitnet_xdna.h"
#include "../runtime/bitnet_i2s.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

static uint32_t rng_state = 0x1234567u;
static uint32_t xs32() {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}

struct Shape { const char *name; const char *file; int64_t K, N; };

struct Job {
    const Shape *shape;
    int64_t T;
    std::vector<int8_t> act;
    std::vector<float>  scales;
    std::vector<float>  ref;      // captured single-threaded
};

static bool load(const Shape &s, std::vector<uint8_t> &blob) {
    const size_t packed = i2s_packed_bytes(s.K, s.N);
    blob.resize(packed + 4);
    FILE *f = std::fopen(s.file, "rb");
    if (!f) { std::printf("  cannot open %s\n", s.file); return false; }
    const size_t got = std::fread(blob.data(), 1, packed + 4, f);
    std::fclose(f);
    return got == packed + 4;
}

/* One invocation, exactly as the ggml hook sequences it. */
static bool invoke(const std::vector<uint8_t> &blob, Job &j, float *dst, bool leased) {
    if (leased) bitnet_xdna_invocation_begin();
    const int ok = bitnet_xdna_accumulate(blob.data(), j.shape->K, j.shape->N,
                                          j.act.data(), j.T, (size_t)j.shape->K);
    if (ok) {
        /* The window a second context would slip into. Kept deliberately: it is
         * the same window the real hook has between accumulate and the epilogue. */
        std::this_thread::yield();
        bitnet_xdna_epilogue(j.shape->N, 0, j.T, j.scales.data(), 1.0f, dst,
                             (size_t)(j.shape->N * sizeof(float)));
    }
    if (leased) bitnet_xdna_invocation_end();
    return ok != 0;
}

/* The unleased mode is expected to corrupt state and may crash outright, which
 * would take the harness with it. So each mode runs in its own exec'd child and
 * the parent reports what happened -- including death by signal, which is itself
 * a result. exec (not bare fork) because XRT holds device state that is not
 * documented fork-safe. */
static int spawn_mode(const char *self, const char *mode, int pairs) {
    char pbuf[32]; std::snprintf(pbuf, sizeof pbuf, "%d", pairs);
    pid_t pid = fork();
    if (pid == 0) {
        char *const av[] = {(char *)self, pbuf, (char *)mode, nullptr};
        execv(self, av);
        _exit(127);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    if (WIFSIGNALED(st)) {
        std::printf("     CHILD KILLED BY SIGNAL %d (%s)\n", WTERMSIG(st),
                    WTERMSIG(st) == SIGSEGV ? "SIGSEGV" : "signal");
        return -1;
    }
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

int main(int argc, char **argv) {
    const int pairs = (argc > 1) ? std::atoi(argv[1]) : 150;
    const char *mode = (argc > 2) ? argv[2] : nullptr;

    static const Shape shapes[] = {
        {"attn_q 2560x2560",   "artifacts/correctness/tensors/attn_q_l0.packed",   2560, 2560},
        {"ffn_gate 2560x6912", "artifacts/correctness/tensors/ffn_gate_l0.packed", 2560, 6912},
        {"ffn_down 6912x2560", "artifacts/correctness/tensors/ffn_down_l0.packed", 6912, 2560},
    };
    const int64_t token_counts[] = {1024, 2048};

    std::printf("test_xdna_concurrent  (two contexts, one process)\n");
    if (!bitnet_xdna_available()) {
        std::printf("  NPU unavailable (set BITNET_XDNA=1 and BITNET_XDNA_ARTIFACTS)\n");
        return 77;
    }

    /* Weight blobs must outlive every call: residency is keyed by data pointer. */
    std::vector<std::vector<uint8_t>> blobs(3);
    for (int i = 0; i < 3; ++i)
        if (!load(shapes[i], blobs[i])) return 1;

    /* Build the job list and capture the single-threaded reference. */
    std::vector<Job> jobs;
    for (int i = 0; i < 3; ++i)
        for (int64_t T : token_counts) {
            Job j; j.shape = &shapes[i]; j.T = T;
            j.act.resize((size_t)(T * shapes[i].K));
            for (auto &v : j.act) v = (int8_t)((int)(xs32() % 255) - 127);
            j.scales.assign((size_t)T, 1.0f);
            j.ref.assign((size_t)(T * shapes[i].N), 0.0f);
            jobs.push_back(std::move(j));
        }
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (!invoke(blobs[jobs[i].shape - shapes], jobs[i], jobs[i].ref.data(), false)) {
            std::printf("  reference capture declined for %s T=%lld\n",
                        jobs[i].shape->name, (long long)jobs[i].T);
            return 1;
        }
    }
    std::printf("  captured %zu single-threaded references\n", jobs.size());

    struct Result { std::atomic<long> mismatch{0}, declined{0}, ran{0}; };

    auto run_mode = [&](bool leased, Result &res) {
        auto worker = [&](int id) {
            std::vector<float> dst;
            for (int it = 0; it < pairs; ++it) {
                /* The two threads walk the job list in opposite directions, so
                 * each iteration pairs different shapes -- the case where a
                 * stale DirectOutPlan has the wrong n_chunks and slot set. */
                Job &j = jobs[(id == 0 ? it : jobs.size() - 1 - (it % jobs.size())) % jobs.size()];
                dst.assign((size_t)(j.T * j.shape->N), 0.0f);
                if (!invoke(blobs[j.shape - shapes], j, dst.data(), leased)) {
                    res.declined.fetch_add(1); continue;
                }
                res.ran.fetch_add(1);
                if (std::memcmp(dst.data(), j.ref.data(), dst.size() * sizeof(float)) != 0)
                    res.mismatch.fetch_add(1);
            }
        };
        std::thread a(worker, 0), b(worker, 1);
        a.join(); b.join();
    };

    /* Child: run exactly one mode and report via exit status. */
    if (mode) {
        Result r;
        run_mode(std::strcmp(mode, "leased") == 0, r);
        std::printf("     invocations %ld   mismatches %ld   declined %ld\n",
                    r.ran.load(), r.mismatch.load(), r.declined.load());
        std::fflush(stdout);
        return (r.mismatch.load() || r.declined.load()) ? 1 : 0;
    }

    int rc = 0;

    std::printf("\n  -- unleased (current raw API), %d pairs/thread --\n", pairs);
    std::fflush(stdout);
    const int un = spawn_mode(argv[0], "unleased", pairs);
    if (un != 0)
        std::printf("     ^ RACE OBSERVED: concurrent contexts corrupt or crash each other\n");
    else
        std::printf("     no corruption observed this run. A race need not fire every time;\n"
                    "     absence here is not proof of safety -- the leased result is the claim.\n");

    std::printf("\n  -- leased (invocation_begin/end), %d pairs/thread --\n", pairs);
    std::fflush(stdout);
    const int le = spawn_mode(argv[0], "leased", pairs);
    if (le != 0) {
        std::printf("     FAIL: the lease did not make concurrent contexts safe\n");
        rc = 1;
    } else {
        std::printf("     ok  all concurrent invocations bit-exact under the lease\n");
    }

    std::printf("\n%s\n", rc ? "FAILURE" : "concurrent-context contract holds");
    return rc;
}
