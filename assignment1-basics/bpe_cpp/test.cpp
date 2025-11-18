#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

// Custom hash for std::pair<std::string, std::string>
struct string_pair_hash {
	std::size_t operator()(const std::pair<std::string, std::string>& p) const {
		auto h1 = std::hash<std::string>{}(p.first);
		auto h2 = std::hash<std::string>{}(p.second);
		return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
	}
};

// Helper function: Merges a specific pair in a token sequence
// Operates purely on std::string (fast, binary-safe)
void _merge_pair_impl(
	std::vector<std::string>& tokens, const std::pair<std::string, std::string>& pair
) {
	std::vector<std::string> new_token;
	new_token.reserve(tokens.size());

	for (size_t i = 0; i < tokens.size(); ++i) {
		if (i < tokens.size() - 1 && tokens[i] == pair.first && tokens[i + 1] == pair.second) {
			new_token.push_back(pair.first + pair.second);
			i++;  // Skip next token
		} else {
			new_token.push_back(tokens[i]);
		}
	}
	tokens = std::move(new_token);
}

// Main function
// 1. Takes py::bytes to avoid UnicodeDecodeError at the interface
// 2. Converts to std::string internally for performance
std::vector<std::pair<py::bytes, py::bytes>> merge_cpp(
	std::unordered_map<py::bytes, std::vector<py::bytes>> pre_tokens_py,
	std::unordered_map<py::bytes, int> pre_token_counts_py,
	std::unordered_map<std::pair<py::bytes, py::bytes>, int, pybind11::hash>
		pair_counts_py,	 // Use pybind11::hash for python objects
	std::unordered_map<
		std::pair<py::bytes, py::bytes>,
		std::unordered_set<py::bytes>,
		pybind11::hash> pair_to_tokens_py,
	int merge_time
) {
	// --- PHASE 1: DATA "DOWNLOAD" (Convert py::bytes -> std::string) ---
	// This is where we manually extract bytes. This does NOT decode UTF-8.
	// It just copies the raw buffer.

	std::unordered_map<std::string, std::vector<std::string>> pre_tokens;
	std::unordered_map<std::string, int> pre_token_counts;
	std::unordered_map<std::pair<std::string, std::string>, int, string_pair_hash> pair_counts;
	std::unordered_map<
		std::pair<std::string, std::string>, std::unordered_set<std::string>, string_pair_hash>
		pair_to_tokens;

	// Pre-allocate to avoid resizing overhead
	pre_tokens.reserve(pre_tokens_py.size());
	pre_token_counts.reserve(pre_token_counts_py.size());
	pair_counts.reserve(pair_counts_py.size());
	pair_to_tokens.reserve(pair_to_tokens_py.size());

	// Convert pre_tokens
	for (auto& item : pre_tokens_py) {
		std::string key = item.first;  // Implicit cast: calls PyBytes_AsStringAndSize
		std::vector<std::string> val;
		val.reserve(item.second.size());
		for (auto& b : item.second) val.push_back(static_cast<std::string>(b));
		pre_tokens[key] = std::move(val);
	}

	// Convert pre_token_counts
	for (auto& item : pre_token_counts_py) {
		pre_token_counts[static_cast<std::string>(item.first)] = item.second;
	}

	// Convert pair_counts
	for (auto& item : pair_counts_py) {
		std::pair<std::string, std::string> key = {
			static_cast<std::string>(item.first.first), static_cast<std::string>(item.first.second)
		};
		pair_counts[key] = item.second;
	}

	// Convert pair_to_tokens
	for (auto& item : pair_to_tokens_py) {
		std::pair<std::string, std::string> key = {
			static_cast<std::string>(item.first.first), static_cast<std::string>(item.first.second)
		};
		std::unordered_set<std::string> val;
		val.reserve(item.second.size());
		for (auto& b : item.second) val.insert(static_cast<std::string>(b));
		pair_to_tokens[key] = std::move(val);
	}

	// --- PHASE 2: CORE LOGIC (Pure C++, Fast) ---
	std::vector<std::pair<std::string, std::string>> merges;

	for (int i = 0; i < merge_time; ++i) {
		std::pair<std::string, std::string> best_pair;
		int max_count = -1;

		// Find best pair
		for (const auto& it : pair_counts) {
			if (it.second > max_count) {
				max_count = it.second;
				best_pair = it.first;
			} else if (it.second == max_count) {
				if (it.first < best_pair) {	 // Lexicographical tie-break
					best_pair = it.first;
				}
			}
		}

		if (max_count < 0) break;
		merges.push_back(best_pair);

		// Update tokens
		// Use a copy of the set because we modify the map during iteration
		std::unordered_set<std::string> tokens_to_update = pair_to_tokens[best_pair];

		for (const auto& token_key : tokens_to_update) {
			std::vector<std::string>& tokens = pre_tokens[token_key];
			int freq = pre_token_counts[token_key];

			if (tokens.size() < 2) continue;

			// 1. Decrement old pairs
			for (size_t j = 0; j < tokens.size() - 1; ++j) {
				std::pair<std::string, std::string> p = {tokens[j], tokens[j + 1]};
				pair_counts[p] -= freq;
				pair_to_tokens[p].erase(token_key);
				if (pair_counts[p] <= 0) {
					pair_counts.erase(p);
					if (pair_to_tokens[p].empty()) pair_to_tokens.erase(p);
				}
			}

			// 2. Merge
			_merge_pair_impl(tokens, best_pair);

			// 3. Increment new pairs
			for (size_t j = 0; j < tokens.size() - 1; ++j) {
				std::pair<std::string, std::string> p = {tokens[j], tokens[j + 1]};
				pair_counts[p] += freq;
				pair_to_tokens[p].insert(token_key);
			}
		}
	}

	// --- PHASE 3: OUTPUT CONVERSION (std::string -> py::bytes) ---
	std::vector<std::pair<py::bytes, py::bytes>> result;
	result.reserve(merges.size());
	for (const auto& p : merges) {
		result.emplace_back(py::bytes(p.first), py::bytes(p.second));
	}
	return result;
}

PYBIND11_MODULE(bpe_cpp, m) {
	m.def("merge_cpp", &merge_cpp, "Merge function with safe bytes handling");
}