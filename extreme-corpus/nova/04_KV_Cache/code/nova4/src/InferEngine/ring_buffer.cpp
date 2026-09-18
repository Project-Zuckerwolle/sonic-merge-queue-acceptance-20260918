// ring_buffer.cpp — siehe ring_buffer.h
#include "InferEngine/ring_buffer.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace nova::infer {

namespace {
using clock = std::chrono::steady_clock;
double secs(clock::duration d) { return std::chrono::duration<double>(d).count(); }

// Beschränkte, blockierende Queue (Kapazität = Triple-Buffer-Tiefe).
template <class T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t cap) : cap_(cap) {}

    bool push(T v) {
        std::unique_lock<std::mutex> lk(m_);
        cv_push_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
        if (closed_) return false;
        q_.push_back(std::move(v));
        cv_pop_.notify_one();
        return true;
    }
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_pop_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;  // geschlossen + leer
        out = std::move(q_.front());
        q_.pop_front();
        cv_push_.notify_one();
        return true;
    }
    void close() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        cv_push_.notify_all();
        cv_pop_.notify_all();
    }

private:
    std::deque<T> q_;
    size_t        cap_;
    bool          closed_ = false;
    std::mutex    m_;
    std::condition_variable cv_push_, cv_pop_;
};

struct RawItem  { int idx; std::vector<uint8_t>  raw;  };
struct WorkItem { int idx; std::vector<uint16_t> work; };
}  // namespace

PipelineStats TripleBuffer::run(int num_chunks, LoadFn load, DecompFn decomp,
                                ComputeFn compute, std::string* err) {
    PipelineStats st;
    st.chunks = uint64_t(num_chunks < 0 ? 0 : num_chunks);

    BoundedQueue<RawItem>  q_ab(static_cast<size_t>(depth_));
    BoundedQueue<WorkItem> q_bc(static_cast<size_t>(depth_));

    std::atomic<bool> failed{false};
    std::mutex   errm;
    std::string  errmsg;
    auto set_err = [&](const std::string& e) {
        std::lock_guard<std::mutex> lk(errm);
        if (errmsg.empty()) errmsg = e;
        failed.store(true);
    };

    double load_busy = 0.0, decomp_busy = 0.0;  // je nur von einer Thread beschrieben
    const auto t_start = clock::now();

    // Stream A — Load
    std::thread ta([&] {
        for (int i = 0; i < num_chunks && !failed.load(); ++i) {
            RawItem r; r.idx = i;
            const auto b = clock::now();
            const bool okk = load(i, r.raw);
            load_busy += secs(clock::now() - b);
            if (!okk) { set_err("Load-Stufe fehlgeschlagen @" + std::to_string(i)); break; }
            if (!q_ab.push(std::move(r))) break;
        }
        q_ab.close();
    });

    // Stream B — Decomp
    std::thread tb([&] {
        RawItem r;
        while (!failed.load() && q_ab.pop(r)) {
            WorkItem w; w.idx = r.idx;
            const auto b = clock::now();
            const bool okk = decomp(r.idx, r.raw, w.work);
            decomp_busy += secs(clock::now() - b);
            if (!okk) { set_err("Decomp-Stufe fehlgeschlagen @" + std::to_string(r.idx)); break; }
            if (!q_bc.push(std::move(w))) break;
        }
        q_bc.close();
    });

    // Stream C — Compute (auf diesem Thread)
    double compute_busy = 0.0, compute_idle = 0.0, max_gap_ms = 0.0;
    bool first = true;
    clock::time_point last_end = t_start;
    WorkItem w;
    while (!failed.load() && q_bc.pop(w)) {
        const auto start = clock::now();
        if (!first) {
            const double gap = secs(start - last_end);   // Lücke seit letztem Compute
            compute_idle += gap;
            if (gap * 1e3 > max_gap_ms) max_gap_ms = gap * 1e3;
        }
        first = false;
        const bool okk = compute(w.idx, w.work);
        last_end = clock::now();
        compute_busy += secs(last_end - start);
        if (!okk) { set_err("Compute-Stufe fehlgeschlagen @" + std::to_string(w.idx)); break; }
    }

    // Upstream-Threads sicher entriegeln und einsammeln.
    q_ab.close();
    q_bc.close();
    ta.join();
    tb.join();

    st.total_seconds      = secs(clock::now() - t_start);
    st.load_busy_s        = load_busy;
    st.decomp_busy_s      = decomp_busy;
    st.compute_busy_s     = compute_busy;
    st.compute_idle_s     = compute_idle;
    st.max_compute_gap_ms = max_gap_ms;
    st.ok                 = !failed.load();
    if (!st.ok && err) { std::lock_guard<std::mutex> lk(errm); *err = errmsg; }
    return st;
}

PipelineStats TripleBuffer::run_sequential(int num_chunks, LoadFn load, DecompFn decomp,
                                           ComputeFn compute, std::string* err) {
    PipelineStats st;
    st.chunks = uint64_t(num_chunks < 0 ? 0 : num_chunks);
    const auto t_start = clock::now();
    std::vector<uint8_t>  raw;
    std::vector<uint16_t> work;
    for (int i = 0; i < num_chunks; ++i) {
        auto b = clock::now();
        if (!load(i, raw)) { if (err) *err = "Load @" + std::to_string(i); st.ok = false; break; }
        st.load_busy_s += secs(clock::now() - b);
        b = clock::now();
        if (!decomp(i, raw, work)) { if (err) *err = "Decomp @" + std::to_string(i); st.ok = false; break; }
        st.decomp_busy_s += secs(clock::now() - b);
        b = clock::now();
        if (!compute(i, work)) { if (err) *err = "Compute @" + std::to_string(i); st.ok = false; break; }
        st.compute_busy_s += secs(clock::now() - b);
    }
    st.total_seconds = secs(clock::now() - t_start);
    return st;
}

}  // namespace nova::infer
