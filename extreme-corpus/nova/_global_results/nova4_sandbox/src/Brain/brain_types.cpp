// brain_types.cpp — Datums-Helfer (Design §11.2 Konfidenz-Konsolidierung).
#include "Brain/brain_types.h"

#include <cstdlib>

namespace nova::brain {

namespace {
// Tage seit Epoche (Howard Hinnants days_from_civil).
long days_from_civil(int y, unsigned m, unsigned d) {
    y -= m <= 2;
    const long era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = unsigned(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097L + long(doe) - 719468L;
}
bool parse(const std::string& s, int& y, unsigned& m, unsigned& d) {
    if (s.size() < 10 || s[4] != '-' || s[7] != '-') return false;
    y = std::atoi(s.substr(0, 4).c_str());
    m = unsigned(std::atoi(s.substr(5, 2).c_str()));
    d = unsigned(std::atoi(s.substr(8, 2).c_str()));
    return m >= 1 && m <= 12 && d >= 1 && d <= 31;
}
}  // namespace

bool valid_date(const std::string& s) {
    int y; unsigned m, d; return parse(s, y, m, d);
}

int days_between(const std::string& from, const std::string& to) {
    int y1, y2; unsigned m1, d1, m2, d2;
    if (!parse(from, y1, m1, d1) || !parse(to, y2, m2, d2)) return 0;
    return int(days_from_civil(y2, m2, d2) - days_from_civil(y1, m1, d1));
}

}  // namespace nova::brain
