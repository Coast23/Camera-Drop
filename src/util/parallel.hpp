#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace camdrop::util {

inline size_t resolve_thread_count(int requested) {
    if (requested > 0) {
        return static_cast<size_t>(requested);
    }
    const unsigned int hc = std::thread::hardware_concurrency();
    return hc == 0 ? 4u : static_cast<size_t>(hc);
}

template <typename InitFn, typename Fn>
void parallel_for_with_state(size_t count, size_t threads, InitFn init, Fn fn) {
    if (count == 0) return;
    if (threads <= 1) {
        auto state = init();
        for (size_t i = 0; i < count; ++i) {
            fn(state, i);
        }
        return;
    }

    std::atomic<size_t> next{0};
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (size_t t = 0; t < threads; ++t) {
        workers.emplace_back([&]() {
            auto state = init();
            for (;;) {
                const size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= count) break;
                fn(state, i);
            }
        });
    }
    for (auto& th : workers) {
        th.join();
    }
}

template <typename Fn>
void parallel_for(size_t count, size_t threads, Fn fn) {
    parallel_for_with_state(count, threads, []() { return 0; },
                            [&](int&, size_t i) { fn(i); });
}

template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t max_size = 0) : max_size_(max_size) {}

    void push(T value) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_not_full_.wait(lock, [&]() {
            return closed_ || max_size_ == 0 || queue_.size() < max_size_;
        });
        if (closed_) return;
        queue_.push_back(std::move(value));
        cv_not_empty_.notify_one();
    }

    bool pop(T& out) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_not_empty_.wait(lock, [&]() { return closed_ || !queue_.empty(); });
        if (queue_.empty()) {
            return false;
        }
        out = std::move(queue_.front());
        queue_.pop_front();
        cv_not_full_.notify_one();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            closed_ = true;
        }
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

    bool closed() const {
        std::lock_guard<std::mutex> lock(mu_);
        return closed_;
    }

private:
    size_t max_size_ = 0;
    mutable std::mutex mu_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    std::deque<T> queue_;
    bool closed_ = false;
};

}  // namespace camdrop::util
