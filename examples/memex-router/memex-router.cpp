// Compute the routing decision in our own code, and check it against the model's.
//
// Until now the engine replayed expert ids from a recorded trace. That proved the
// machinery around routing - residency, precision, dispatch - but not the routing
// itself, and an engine that cannot decide which experts to use is not an engine. The
// pieces needed are small: the layer's router is a single [n_embd, n_expert] matrix, now
// stored in the blob beside the experts it selects, and the input it consumes is the
// normalised hidden state, now captured by the tracer.
//
// So this is a verification, not a demonstration: same activation, same weights, and the
// top-k our code produces must be the top-k the model produced. Anything less means the
// engine would quietly route to the wrong experts - the kind of fault that leaves speed
// intact and correctness gone.
//
// Only the ordering matters for the comparison, so the softmax is not needed: it is
// monotone, and the top-k of the logits is the top-k of the probabilities.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "memex/blob_loader.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Trace records: tag >= 0 ids, -(l+1) chosen weights, -(l+1)-10000 distribution,
// -(l+1)-20000 the activation.
bool load_trace(const std::string& path, int layer, int want_width, int kind,
                std::vector<std::vector<float>>* fout,
                std::vector<std::vector<int>>* iout) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        printf("нет трассы: %s\n", path.c_str());
        return false;
    }
    const int want_tag = kind == 0 ? layer
                                   : (kind == 1 ? -layer - 20001 : -layer - 10001);
    int32_t hdr[3];
    while (fread(hdr, sizeof(int32_t), 3, f) == 3) {
        const int tag = hdr[0], nu = hdr[1], nt = hdr[2];
        const size_t n = size_t(nu) * size_t(nt);
        if (tag != want_tag || (want_width > 0 && nu != want_width)) {
            // 64-bit seek is mandatory: traces run to gigabytes and MSVC's long is 32-bit,
            // so this skip would silently wrap and land inside a record instead of failing.
#ifdef _WIN32
            if (_fseeki64(f, (__int64)(4 * n), SEEK_CUR) != 0) break;
#else
            if (fseeko(f, (off_t)(4 * n), SEEK_CUR) != 0) break;
#endif
            continue;
        }
        if (kind == 0) {
            std::vector<int32_t> buf(n);
            if (fread(buf.data(), sizeof(int32_t), n, f) != n) break;
            for (int t = 0; t < nt; ++t) {
                iout->emplace_back(buf.begin() + size_t(t) * nu,
                                   buf.begin() + size_t(t + 1) * nu);
            }
        } else {
            std::vector<float> buf(n);
            if (fread(buf.data(), sizeof(float), n, f) != n) break;
            for (int t = 0; t < nt; ++t) {
                fout->emplace_back(buf.begin() + size_t(t) * nu,
                                   buf.begin() + size_t(t + 1) * nu);
            }
        }
    }
    fclose(f);
    return kind == 0 ? !iout->empty() : !fout->empty();
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string blob = "D:/MemeX/blob/l20r.bin";
    std::string man = "D:/MemeX/blob/l20r.json";
    std::string act_trace = "D:/MemeX/results/tr_act.bin";
    std::string ids_trace = "D:/MemeX/results/tr_code.bin";
    int layer = 20;
    int n_embd = 2048;
    int n_experts = 128;
    int top_k = 8;
    int max_tokens = 512;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--blob") && i + 1 < argc) blob = argv[++i];
        else if (!strcmp(argv[i], "--manifest") && i + 1 < argc) man = argv[++i];
        else if (!strcmp(argv[i], "--act") && i + 1 < argc) act_trace = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_trace = argv[++i];
        else if (!strcmp(argv[i], "--layer") && i + 1 < argc) layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) max_tokens = atoi(argv[++i]);
    }

    memex::BlobLoader loader;
    std::string err;
    if (!loader.open(blob, man, &err)) {
        printf("блоб не открылся: %s\n", err.c_str());
        return 1;
    }
    const memex::BlobEntry* rt = loader.find(uint32_t(layer), /*tensor=*/3, 0,
                                             /*ggml_type=*/0);
    if (!rt) {
        // The router is stored as F32, whose ggml type id is 0; look it up directly if
        // the keyed lookup misses.
        for (const auto& e : loader.entries()) {
            if (int(e.layer) == layer && e.tensor == 3) {
                rt = &e;
                break;
            }
        }
    }
    if (!rt) {
        printf("в блобе нет роутера для слоя %d — пересобери блоб\n", layer);
        return 1;
    }
    printf("роутер: %u x %u, %u байт, тип ggml %d\n", rt->rows, rt->cols, rt->bytes,
           rt->ggml_type);

    memex::AlignedBuffer buf(memex::AlignedBuffer::round_up(rt->bytes) + 4096);
    if (!loader.read_sync(*rt, buf.data(), buf.size())) {
        printf("роутер не читается\n");
        return 1;
    }
    const float* W = (const float*)buf.data();     // [n_experts, n_embd] row-major

    std::vector<std::vector<float>> acts;
    std::vector<std::vector<int>> ids;
    std::vector<std::vector<float>> unused_f;
    std::vector<std::vector<int>> unused_i;
    if (!load_trace(act_trace, layer, n_embd, /*kind=*/1, &acts, &unused_i)) {
        printf("нет активаций для слоя %d\n", layer);
        return 1;
    }
    if (!load_trace(ids_trace, layer, top_k, /*kind=*/0, &unused_f, &ids)) {
        printf("нет записанных номеров экспертов для слоя %d\n", layer);
        return 1;
    }
    const int n = std::min<int>({max_tokens, int(acts.size()), int(ids.size())});
    printf("активаций %zu, записанных решений %zu, сверяю %d токенов\n",
           acts.size(), ids.size(), n);

    int exact = 0, same_set = 0, total_overlap = 0;
    double ms = 0.0;
    // resize instead of a constructor argument: `vector<float> v(size_t(n))` parses as
    // a function declaration, and it has bitten this project four times now
    std::vector<float> logits;
    std::vector<int> order;
    logits.resize(size_t(n_experts));
    order.resize(size_t(n_experts));
    for (int t = 0; t < n; ++t) {
        const std::vector<float>& x = acts[size_t(t)];
        const auto t0 = Clock::now();
        for (int e = 0; e < n_experts; ++e) {
            const float* w = W + size_t(e) * n_embd;
            float s = 0.0f;
            for (int j = 0; j < n_embd; ++j) {
                s += w[j] * x[size_t(j)];
            }
            logits[size_t(e)] = s;
        }
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                          [&](int a, int b) { return logits[size_t(a)] > logits[size_t(b)]; });
        ms += ms_since(t0);

        const std::vector<int>& want = ids[size_t(t)];
        std::vector<int> got(order.begin(), order.begin() + top_k);
        bool ordered_match = true;
        for (int i = 0; i < top_k; ++i) {
            if (got[size_t(i)] != want[size_t(i)]) {
                ordered_match = false;
                break;
            }
        }
        std::vector<int> a = got, b = want;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        int overlap = 0;
        for (int i = 0, j = 0; i < top_k && j < top_k;) {
            if (a[size_t(i)] == b[size_t(j)]) { overlap++; i++; j++; }
            else if (a[size_t(i)] < b[size_t(j)]) i++;
            else j++;
        }
        total_overlap += overlap;
        if (a == b) same_set++;
        if (ordered_match) exact++;
    }

    printf("\n=== сверка нашего роутера с моделью ===\n");
    printf("  тот же набор из %d: %d из %d (%.2f%%)\n", top_k, same_set, n,
           100.0 * same_set / n);
    printf("  тот же порядок    : %d из %d (%.2f%%)\n", exact, n, 100.0 * exact / n);
    printf("  среднее совпадение: %.3f из %d экспертов\n",
           double(total_overlap) / n, top_k);
    printf("  счёт роутера      : %.3f мс на токен на слой (%.1f мс на 48 слоёв)\n",
           ms / n, 48.0 * ms / n);
    printf("\nВЕРДИКТ: %s\n",
           same_set == n ? "наш роутер выбирает то же, что модель"
                         : "расхождение — движок направлял бы токены не тем экспертам");
    return same_set == n ? 0 : 2;
}
