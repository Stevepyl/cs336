#include <omp.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <chrono>
#include <iostream>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;
using pair = std::pair<uint32_t, uint32_t>;
using string_pair = std::pair<std::string, std::string>;
using py_byte_pair = std::pair<py::bytes, py::bytes>;
using Clock = std::chrono::steady_clock;

#if defined(__cpp_lib_hardware_interference_size)
static constexpr std::size_t CACHE_LINE = std::hardware_constructive_interference_size;
#else
static constexpr std::size_t CACHE_LINE = 128;	// Apple Silicon, Modify this to 64 if on linux
#endif

// Avoid "False Sharing"
// std::hardware_constructive_interference_size
// 表示两个对象为了避免相互干扰（即伪共享）所需的最小内存间隔
// 保证不同线程操作的 LocalTop 实例位于不同的物理缓存行中，互不干扰。
// Also can use:
// struct alignas(CACHE_LINE) LocalTop {
// 	int top_count = -1;
// 	string_pair top_pair;
// };
struct LocalTop {
	int top_count = -1;
	string_pair top_pair;
	char padding[CACHE_LINE];
};

// Cpp 的标准库中并没有提供 std::pair<int, int>
// 的哈希函数，所以需要我们自己实现 从而用在 unordered_map
// 中作为哈希表的哈希
// From Boost
struct pair_hash {
	template <class T1, class T2>
	std::size_t operator()(const std::pair<T1, T2>& p) const {
		auto h1 = std::hash<T1>{}(p.first);
		auto h2 = std::hash<T2>{}(p.second);

		return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
	}
};

void _merge_pair(std::vector<std::string>& tokens, const string_pair& pair) {
	std::vector<std::string> new_token;
	new_token.reserve(tokens.size());
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
	tokens = std::move(new_token);
}

std::vector<py_byte_pair> merge_cpp(
	py::dict pre_tokens_py,	 // Accept raw python dicts
	py::dict pre_token_counts_py,
	py::dict pair_counts_py,
	py::dict pair_to_tokens_py,
	int merge_time
) {
	std::unordered_map<std::string, std::vector<std::string>> pre_tokens;
	std::unordered_map<std::string, int> pre_token_counts;
	std::unordered_map<string_pair, int, pair_hash> pair_counts;
	std::unordered_map<string_pair, std::unordered_set<std::string>, pair_hash> pair_to_tokens;

	auto convert_start = Clock::now();

	pre_tokens.reserve(pre_tokens_py.size());
	pre_token_counts.reserve(pre_token_counts_py.size());
	pair_counts.reserve(pair_counts_py.size());
	pair_to_tokens.reserve(pair_to_tokens_py.size());

    // Convert parameters to cpp unordered_map
	// Convert pre_tokens
	for (auto item : pre_tokens_py) {
		std::string key = item.first.cast<std::string>();
		std::vector<std::string> val;
		// item.second is a list of bytes
		auto val_list = item.second.cast<py::list>();
		val.reserve(val_list.size());
		for (auto b : val_list) {
			val.push_back(b.cast<std::string>());
		}
		pre_tokens[key] = std::move(val);
	}

	// Convert pre_token_counts
	for (auto item : pre_token_counts_py) {
		pre_token_counts[item.first.cast<std::string>()] = item.second.cast<int>();
	}

	// Convert pair_counts
	for (auto item : pair_counts_py) {
		auto pair_tuple = item.first.cast<std::pair<py::bytes, py::bytes>>();
		std::pair<std::string, std::string> key = {
			pair_tuple.first.cast<std::string>(), pair_tuple.second.cast<std::string>()
		};
		pair_counts[key] = item.second.cast<int>();
	}

	// Convert pair_to_tokens
	for (auto item : pair_to_tokens_py) {
		auto pair_tuple = item.first.cast<std::pair<py::bytes, py::bytes>>();
		std::pair<std::string, std::string> key = {
			pair_tuple.first.cast<std::string>(), pair_tuple.second.cast<std::string>()
		};

		std::unordered_set<std::string> val_set;
		auto py_set = item.second.cast<py::set>();
		for (auto token : py_set) {
			val_set.insert(token.cast<std::string>());
		}
		pair_to_tokens[key] = std::move(val_set);
	}
	auto convert_end = Clock::now();
	auto convert_time =
		std::chrono::duration_cast<std::chrono::milliseconds>(convert_end - convert_start);
	std::cout << "Convert using: " << convert_time.count() << " ms" << std::endl;


	std::vector<string_pair> merges;
	Clock::duration get_top_time = Clock::duration::zero();
	Clock::duration update_time = Clock::duration::zero();

	for (int i = 0; i < merge_time; i++) {
		// Finding top pair
		LocalTop top;

		// C++的unordered_map把不同的 hashcode 相同的key对应的pair都放在同一个桶(bucket)里，
		// 每个非空桶指向一个链表，用来存储这些hashcode相同的pair
		// unordered_map有一个bucket
		// api，它可以使得我们实现类似于vector的方式遍历，从而实现并行求max

		Clock::time_point start_time = Clock::now();
		size_t num_buckets = pair_counts.bucket_count();
		std::vector<LocalTop> local_tops;

		// parallel start
#pragma omp parallel
		{
#pragma omp single
			{
				int num_threads = omp_get_num_threads();
				local_tops.resize(num_threads);
			}
			// 每一个线程中的top_pair
			LocalTop local_top;

#pragma omp for schedule(runtime)
			for (int i = 0; i < num_buckets; ++i) {
				// 局部迭代器 (Local Iterators)：
				//    标准库提供了重载版本的 begin() 和 end() 方法，接受一个桶的索引作为参数：
				//    pair_counts.begin(bucket_idx)：返回指向第 bucket_idx
				//    个桶中第一个元素的局部迭代器。
				//	  pair_counts.end(bucket_idx)：返回指向第 bucket_idx个桶末尾的局部迭代器
				for (auto it = pair_counts.begin(i); it != pair_counts.end(i); ++it) {
					const auto& pair = it->first;
					const auto& count = it->second;
					if (count > local_top.top_count) {
						local_top.top_count = count;
						local_top.top_pair = pair;
					} else if (count == local_top.top_count && pair > local_top.top_pair) {
						local_top.top_count = count;
						local_top.top_pair = pair;
					}
				}
			}
			int thread_id = omp_get_thread_num();
			local_tops[thread_id] = local_top;
		}
		// 串行在每个线程的top_pair里求最top的
		for (const auto& t : local_tops) {
			if (-1 == t.top_count) {
				continue;
			} else {
				if (t.top_count > top.top_count) {
					top.top_count = t.top_count;
					top.top_pair = t.top_pair;
				} else if (t.top_count == top.top_count && t.top_pair > top.top_pair) {
					top.top_count = t.top_count;
					top.top_pair = t.top_pair;
				}
			}
		}

		Clock::time_point end_time = Clock::now();
		get_top_time += (end_time - start_time);
		merges.push_back(top.top_pair);

		auto update_start = Clock::now();

        // Update affected tokens
		std::unordered_set<std::string> affected_tokens = pair_to_tokens[top.top_pair];
		for (auto& affected_token : affected_tokens) {
			std::vector<std::string>& affected_token_bytes = pre_tokens[affected_token];
			const int affected_token_bytes_size = affected_token_bytes.size();
			if (affected_token_bytes_size < 2) {
				continue;
			}
			// Decrement all pair counts in affected token
			for (int i = 0; i < affected_token_bytes_size - 1; i++) {
				string_pair old_pair(affected_token_bytes[i], affected_token_bytes[i + 1]);
				pair_counts[old_pair] -= pre_token_counts[affected_token];
				pair_to_tokens[old_pair].erase(affected_token);
				if (0 >= pair_counts[old_pair]) {
					pair_counts.erase(old_pair);
					pair_to_tokens.erase(old_pair);
				}
			}

			_merge_pair(affected_token_bytes, top.top_pair);

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

		auto update_end = Clock::now();
		update_time += (update_end - update_start);
	}

    // Convert result from std::unordered_map into py::bytes
	std::vector<py_byte_pair> result;
	result.reserve(merges.size());
	for (const auto& p : merges) {
		result.emplace_back(py::bytes(p.first), py::bytes(p.second));
	}

	auto get_top_ms = std::chrono::duration_cast<std::chrono::milliseconds>(get_top_time);
	std::cout << "Get top pair using: " << get_top_ms.count() << " ms" << std::endl;

	auto update_ms = std::chrono::duration_cast<std::chrono::milliseconds>(update_time);
    std::cout << "Update affected pairs using: " << update_ms.count() << "ms" << std::endl;
    return result;
}

PYBIND11_MODULE(bpe_cpp, m) {
	m.doc() = "cpp implemented merge";
	m.def("merge_cpp", &merge_cpp, py::return_value_policy::take_ownership);
}