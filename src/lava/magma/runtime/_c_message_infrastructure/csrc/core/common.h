// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CORE_COMMON_H_
#define CORE_COMMON_H_

#include <message_infrastructure/csrc/core/utils.h>

#include <vector>
#include <atomic>
#include <memory>
#include <string>
#include <cassert>

namespace message_infrastructure {

template<class T>
class RecvQueue{
 public:
  RecvQueue(const std::string& name, const size_t &size)
    : name_(name), size_(size), read_index_(0), write_index_(0), done_(false) {
    array_.resize(size_);
  }
  ~RecvQueue() {
    Free();
  }
  void Push(T val) {
    auto const curr_write_index = write_index_.load(std::memory_order_relaxed);
    auto next_write_index = curr_write_index + 1;
    if (next_write_index == size_) {
      next_write_index = 0;
    }
    if (next_write_index != read_index_.load(std::memory_order_acquire)) {
      array_[curr_write_index] = val;
      write_index_.store(next_write_index, std::memory_order_release);
    }
  }
  T Pop(bool block) {
    while (block && Empty()) {
      helper::Sleep();
      if (done_)
        return nullptr;
    }
    auto const curr_read_index = read_index_.load(std::memory_order_relaxed);
    assert(curr_read_index != write_index_.load(std::memory_order_acquire));
    T data_ = array_[curr_read_index];
    auto next_read_index = curr_read_index + 1;
    if (next_read_index == size_) {
      next_read_index = 0;
    }
    read_index_.store(next_read_index, std::memory_order_release);
    return data_;
  }
  int AvailableCount() {
    auto const curr_read_index = read_index_.load(std::memory_order_acquire);
    auto const curr_write_index = write_index_.load(std::memory_order_acquire);
    if (curr_read_index == curr_write_index) {
      return size_;
    }
    if (curr_write_index > curr_read_index) {
      return size_ - curr_write_index + curr_read_index - 1;
    }
    return curr_read_index - curr_write_index - 1;
  }
  T Front() {
    while (Empty()) {
      helper::Sleep();
      if (done_)
        return nullptr;
    }
    auto curr_read_index = read_index_.load(std::memory_order_acquire);
    T ptr = array_[curr_read_index];
    return ptr;
  }
  bool Empty() {
    auto const curr_read_index = read_index_.load(std::memory_order_acquire);
    auto const curr_write_index = write_index_.load(std::memory_order_acquire);
    return curr_read_index == curr_write_index;
  }
  void Free() {
    if (!Empty()) {
      auto const curr_read_index = read_index_.load(std::memory_order_acquire);
      auto const curr_write_index = write_index_.load(std::memory_order_acquire); // NOLINT
      int max, min;
      if (curr_read_index < curr_write_index) {
        max = curr_write_index;
        min = curr_read_index;
      } else {
        min = curr_write_index + 1;
        max = curr_read_index + 1;
      }
      for (int i = min; i < max; i++) {
        FreeData(array_[i]);
        array_[i] = nullptr;
      }
      read_index_.store(0, std::memory_order_release);
      write_index_.store(0, std::memory_order_release);
    }
  }
  bool Probe() {
    return !Empty();
  }
  void Stop() {
    done_ = true;
  }

 private:
  std::vector<T> array_;
  std::atomic<uint32_t> read_index_;
  std::atomic<uint32_t> write_index_;
  std::string name_;
  size_t size_;
  std::atomic_bool done_;

  void FreeData(T data);
};

}  // namespace message_infrastructure

#endif  // CORE_COMMON_H_
