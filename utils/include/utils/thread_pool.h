
#pragma once
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fish {
namespace utils {
namespace parallel {
class ThreadPool {
public:
    ThreadPool(size_t);
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();

    // return the resource!
    int get_thread_index() { return thread_id_lut[std::this_thread::get_id()]; }

    size_t get_pool_size() { return workers.size(); }

private:
    // need to keep track of threads so we can join them
    std::vector<std::thread> workers;
    // the task queue
    std::queue<std::function<void()>> tasks;

    // synchronization
    std::mutex              queue_mutex;
    std::condition_variable condition;
    bool                    stop;
    // prepare this to map a resource with each source!
    std::unordered_map<std::thread::id, int> thread_id_lut;
    int                                      thread_idx;

    // maybe we will not use it!
    std::unordered_map<std::thread::id, int>& get_thread_id_lut_ref() { return thread_id_lut; }

    const std::unordered_map<std::thread::id, int> get_thread_id_lut_cref() const {
        return thread_id_lut;
    }
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads)
    : stop(false)
    , thread_idx(0) {
    for (size_t i = 0; i < threads; ++i)
        workers.emplace_back([this] {
            auto thread_id           = std::this_thread::get_id();
            thread_id_lut[thread_id] = thread_idx;
            ++thread_idx;
            // loop for task!
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock,
                                         [this] { return this->stop || !this->tasks.empty(); });
                    if (this->stop && this->tasks.empty()) return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                task();
            }
        });
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f,
                         Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) worker.join();
}
}   // namespace parallel
}   // namespace utils
}   // namespace fish
