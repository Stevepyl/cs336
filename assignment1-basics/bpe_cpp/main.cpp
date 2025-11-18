#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;
using pair = std::pair<uint32_t, uint32_t>;
using string_pair = std::pair<std::string, std::string>;

// Cpp 的标准库中并没有提供 std::pair<int, int>
// 的哈希函数，所以需要我们自己实现 从而用在 unordered_map
// 中作为哈希表的哈希 From Boost
struct pair_hash {
	template <class T1, class T2>
	std::size_t operator()(const std::pair<T1, T2>& p) const {
		auto h1 = std::hash<T1>{}(p.first);
		auto h2 = std::hash<T2>{}(p.second);

		return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
	}
};

// std::vector<pair> _merge(
// 	std::unordered_map<int, std::vector<int>>& pre_tokens,
// 	std::unordered_map<int, std::vector<int>>& pre_token_counts,
// 	std::unordered_map<pair, int, pair_hash>& pair_counts,
// 	std::unordered_map<pair, std::unordered_set<int>, pair_hash>&
// pair_to_tokens, 	int merge_time, ) {

// }

void _merge_pair(std::vector<std::string>& tokens, const string_pair& pair) {
	std::vector<std::string> new_token;
	int i = 0;

	while (i < tokens.size()) {
		if (i < tokens.size() - 1 && tokens[i] == pair.first && tokens[i + 1] == pair.second) {
			new_token.emplace_back(pair.first + pair.second);
			i += 2;
		} else {
			new_token.emplace_back(tokens[i]);
			i++;
		}
	}
	tokens = new_token;
}

std::vector<string_pair> merge_cpp(
	std::unordered_map<std::string, std::vector<std::string>>& pre_tokens,
	std::unordered_map<std::string, int>& pre_token_counts,
	std::unordered_map<string_pair, int, pair_hash>& pair_counts,
	std::unordered_map<string_pair, std::unordered_set<std::string>, pair_hash>& pair_to_tokens,
	int merge_time
) {


    // for (auto& [pair, count] : pair_counts) {
    //     std::cout << "Cpp pair count: " << pair.first << ", " << pair.second << " -> " << count << std::endl;
    // }

    // for (auto& [token, bytes] : pre_tokens) {
    //     std::cout << "Cpp pre_token: " << token << " -> ";
    //     for (const auto& byte : bytes) {
    //         std::cout << byte << " ";
    //     }
    //     std::cout << std::endl;
    // }

	std::vector<string_pair> merges;
	for (int i = 0; i < merge_time; i++) {
		// Finding top pair
		string_pair top_pair;
		int top_count = -1;
		for (const auto& [pair, count] : pair_counts) {
			if (count > top_count) {
				top_count = count;
				top_pair = pair;
			} else if (count == top_count && pair > top_pair) {
				top_count = count;
				top_pair = pair;
			}
		}

		merges.push_back(top_pair);

		std::unordered_set<std::string> affected_tokens = pair_to_tokens[top_pair];
		for (auto& affected_token : affected_tokens) {
			std::vector<std::string>& affected_token_bytes = pre_tokens[affected_token];
			const int affected_token_bytes_size = affected_token_bytes.size();
			if (affected_token_bytes_size < 2) {
				continue;
			}
			//Decrement all pair counts in affected token
			for (int i = 0; i < affected_token_bytes_size - 1; i++) {
				string_pair old_pair(affected_token_bytes[i], affected_token_bytes[i + 1]);
				pair_counts[old_pair] -= pre_token_counts[affected_token];
				pair_to_tokens[old_pair].erase(affected_token);
				if (0 >= pair_counts[old_pair]) {
					pair_counts.erase(old_pair);
					pair_to_tokens.erase(old_pair);
				}
			}

			_merge_pair(affected_token_bytes, top_pair);

			// Increment all pair counts in affected token
            // ATTENTION: affected_token_bytes size has changed after merge
			const int new_size = affected_token_bytes.size();
			for (int i = 0; i < new_size - 1; i++) {
				string_pair new_pair(affected_token_bytes[i], affected_token_bytes[i + 1]);
				pair_counts[new_pair] += pre_token_counts[affected_token];
				pair_to_tokens[new_pair].insert(affected_token);
			}
			pre_tokens[affected_token] = affected_token_bytes;
		}
	}
    
	return merges;
}

PYBIND11_MODULE(bpe_cpp, m) {
	m.doc() = "cpp implemented merge";
	m.def("merge_cpp", &merge_cpp, py::return_value_policy::take_ownership);
}