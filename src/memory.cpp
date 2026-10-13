#include "memora/memory.hpp"

namespace memora {

namespace {
template <typename F>
std::vector<double> column(const std::vector<Memory>& items, F f) {
    std::vector<double> out;
    out.reserve(items.size());
    for (const auto& m : items) out.push_back(f(m));
    return out;
}
}  // namespace

std::vector<double> Store::sizes() const {
    return column(items, [](const Memory& m) { return m.size_mb; });
}
std::vector<double> Store::observed_importance() const {
    return column(items, [](const Memory& m) { return m.observed_importance; });
}
std::vector<double> Store::true_importance() const {
    return column(items, [](const Memory& m) { return m.true_importance; });
}
std::vector<double> Store::age_days() const {
    return column(items, [](const Memory& m) { return m.age_days; });
}
std::vector<double> Store::recency_days() const {
    return column(items, [](const Memory& m) { return m.recency_days; });
}
std::vector<double> Store::access_counts() const {
    return column(items, [](const Memory& m) { return double(m.access_count); });
}

}  // namespace memora
