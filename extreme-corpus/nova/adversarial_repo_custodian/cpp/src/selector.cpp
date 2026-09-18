#include <algorithm>
#include <string>
#include <vector>

struct Candidate {
  std::string path;
  bool tracked;
  bool generated;
  bool protected_branch;
};

std::vector<Candidate> select_candidates(std::vector<Candidate> values) {
  std::erase_if(values, [](const Candidate& value) {
    return value.tracked && !value.generated && !value.protected_branch;
  });
  return values;
}
