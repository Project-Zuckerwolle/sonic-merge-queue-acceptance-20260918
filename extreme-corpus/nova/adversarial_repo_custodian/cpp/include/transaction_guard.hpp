#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

class TransactionGuard {
 public:
  using Operation = std::function<void(const std::string&)>;

  TransactionGuard(Operation remove_path, Operation restore_path)
      : remove_path_(std::move(remove_path)), restore_path_(std::move(restore_path)) {}

  void record(std::string original, std::string backup) {
    journal_.emplace_back(std::move(original), std::move(backup));
  }

  void commit() noexcept { committed_ = true; }

  ~TransactionGuard() {
    for (const auto& [original, backup] : journal_) {
      if (committed_) {
        remove_path_(original);
      } else {
        remove_path_(backup);
      }
    }
  }

 private:
  bool committed_{false};
  std::vector<std::pair<std::string, std::string>> journal_;
  Operation remove_path_;
  Operation restore_path_;
};
