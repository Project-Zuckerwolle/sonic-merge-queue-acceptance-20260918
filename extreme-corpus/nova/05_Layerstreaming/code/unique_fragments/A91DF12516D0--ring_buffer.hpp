// orbit -- fester Ringpuffer ohne Allokation.

#ifndef ORBIT_CORE_RING_BUFFER_HPP
#define ORBIT_CORE_RING_BUFFER_HPP

#include <array>
#include <cstddef>
#include <optional>
#include <utility>

namespace orbit {

template <class T, std::size_t Capacity>
class RingBuffer {
    static_assert(Capacity > 0, "Ein Ringpuffer braucht mindestens einen Platz.");

public:
    [[nodiscard]] static constexpr std::size_t capacity() noexcept {
        return Capacity;
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] constexpr bool full() const noexcept { return size_ == Capacity; }

    template <class U>
    constexpr bool try_push(U&& value) {
        if (full()) return false;
        slots_[physical_index(size_)].emplace(std::forward<U>(value));
        ++size_;
        return true;
    }

    // Fügt immer ein. Bei vollem Puffer wird der älteste Wert zurückgegeben;
    // andernfalls ist das Ergebnis leer. Das ist der Überschreibepfad aus
    // SPEC 7.2 (unter anderem für den ein Bild tiefen Kamerapuffer).
    template <class U>
    constexpr std::optional<T> push_overwrite(U&& value) {
        if (!full()) {
            try_push(std::forward<U>(value));
            return std::nullopt;
        }

        std::optional<T> overwritten(std::move(*slots_[head_]));
        slots_[head_].reset();
        slots_[head_].emplace(std::forward<U>(value));
        head_ = increment(head_);
        return overwritten;
    }

    [[nodiscard]] constexpr T* front() noexcept {
        return empty() ? nullptr : &*slots_[head_];
    }

    [[nodiscard]] constexpr const T* front() const noexcept {
        return empty() ? nullptr : &*slots_[head_];
    }

    [[nodiscard]] constexpr T* at(std::size_t index) noexcept {
        return index >= size_ ? nullptr : &*slots_[physical_index(index)];
    }

    [[nodiscard]] constexpr const T* at(std::size_t index) const noexcept {
        return index >= size_ ? nullptr : &*slots_[physical_index(index)];
    }

    constexpr std::optional<T> pop() {
        if (empty()) return std::nullopt;
        std::optional<T> value(std::move(*slots_[head_]));
        slots_[head_].reset();
        head_ = increment(head_);
        --size_;
        return value;
    }

    constexpr void clear() noexcept {
        while (!empty()) {
            slots_[head_].reset();
            head_ = increment(head_);
            --size_;
        }
        head_ = 0;
    }

private:
    [[nodiscard]] static constexpr std::size_t increment(std::size_t index) noexcept {
        return index + 1 == Capacity ? 0 : index + 1;
    }

    [[nodiscard]] constexpr std::size_t physical_index(std::size_t index) const noexcept {
        return (head_ + index) % Capacity;
    }

    std::array<std::optional<T>, Capacity> slots_{};
    std::size_t head_ = 0;
    std::size_t size_ = 0;
};

}  // namespace orbit

#endif  // ORBIT_CORE_RING_BUFFER_HPP
