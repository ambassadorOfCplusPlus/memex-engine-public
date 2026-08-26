// memex-qerr - per-expert requantisation error probe.
//
// The question: inside one MoE layer, all 128 experts live stacked in a single 3D tensor and
// are all quantised to the same type. If we requantise that layer down to IQ4_KS, do all the
// experts suffer equally, or do some take far more damage than others? If the spread is flat
// there is no point keeping a subset of experts at higher precision; if it is wide, a mixed
// precision expert blob is worth building.
//
// Deliberately does NOT go through the llama model loader: that would allocate the whole
// ~17 GB file. We read the GGUF header with no_alloc, take the descriptors (shape, type,
// strides) and then pull only the byte range of one expert slab at a time with fread. An
// expert of ffn_up_exps is ~1.6M floats, so buffers are allocated once and reused.
//
// Per expert:  stored bytes -> to_float -> quantize_chunk(target) -> to_float -> compare.
// The number reported is the relative Frobenius error sqrt(sum (w-w2)^2 / sum w^2).
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"

namespace {

struct Options {
    std::string model = "D:\\Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf";
    std::string type_name = "iq4_ks";
    std::string tensor_sel = "all";
    std::vector<int> layers = {0, 9, 19, 29, 39, 47};
    int max_experts = -1; // -1 = all
};

struct TypeAlias {
    const char * name;
    ggml_type    type;
};

const TypeAlias k_type_aliases[] = {
    { "iq2_ks",  GGML_TYPE_IQ2_KS  },
    { "iq2_kl",  GGML_TYPE_IQ2_KL  },
    { "iq3_ks",  GGML_TYPE_IQ3_KS  },
    { "iq4_ks",  GGML_TYPE_IQ4_KS  },
    { "iq4_kss", GGML_TYPE_IQ4_KSS },
    { "iq5_ks",  GGML_TYPE_IQ5_KS  },
    { "iq4_xs",  GGML_TYPE_IQ4_XS  },
    { "q4_k",    GGML_TYPE_Q4_K    },
    { "q5_k",    GGML_TYPE_Q5_K    },
    { "q6_k",    GGML_TYPE_Q6_K    },
    { "q8_0",    GGML_TYPE_Q8_0    },
};

std::string to_lower(std::string s) {
    for (char & c : s) {
        if (c >= 'A' && c <= 'Z') {
            c = char(c - 'A' + 'a');
        }
    }
    return s;
}

bool parse_type(const std::string & name, ggml_type * out) {
    const std::string want = to_lower(name);
    for (const TypeAlias & a : k_type_aliases) {
        if (want == a.name) {
            *out = a.type;
            return true;
        }
    }
    return false;
}

bool parse_int_list(const std::string & s, std::vector<int> * out) {
    out->clear();
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        const std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!tok.empty()) {
            char * end = nullptr;
            const long v = strtol(tok.c_str(), &end, 10);
            if (end == tok.c_str() || (end && *end != '\0')) {
                fprintf(stderr, "error: '%s' in --layers is not an integer\n", tok.c_str());
                return false;
            }
            out->push_back(int(v));
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return !out->empty();
}

void print_usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s [-m model.gguf] [--type iq4_ks] [--layers 0,9,19]\n"
            "          [--tensor up|gate|down|all] [--experts N]\n"
            "\n"
            "  --type      target quantisation type: iq2_ks, iq2_kl, iq3_ks, iq4_ks, iq4_kss,\n"
            "              iq5_ks, iq4_xs, q4_K, q5_K, q6_K, q8_0\n"
            "  --layers    comma separated layer indices (default 0,9,19,29,39,47)\n"
            "  --tensor    which expert tensor kind(s) to probe (default all)\n"
            "  --experts   only process the first N experts of each tensor (default all)\n",
            argv0);
}

bool parse_args(int argc, char ** argv, Options * o) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_next = (i + 1) < argc;
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "-m" || arg == "--model") {
            if (!has_next) { fprintf(stderr, "error: %s needs a value\n", arg.c_str()); return false; }
            o->model = argv[++i];
        } else if (arg == "--type") {
            if (!has_next) { fprintf(stderr, "error: %s needs a value\n", arg.c_str()); return false; }
            o->type_name = argv[++i];
        } else if (arg == "--tensor") {
            if (!has_next) { fprintf(stderr, "error: %s needs a value\n", arg.c_str()); return false; }
            o->tensor_sel = to_lower(argv[++i]);
            if (o->tensor_sel != "up" && o->tensor_sel != "gate" &&
                o->tensor_sel != "down" && o->tensor_sel != "all") {
                fprintf(stderr, "error: --tensor must be up, gate, down or all (got '%s')\n", o->tensor_sel.c_str());
                return false;
            }
        } else if (arg == "--layers") {
            if (!has_next) { fprintf(stderr, "error: %s needs a value\n", arg.c_str()); return false; }
            if (!parse_int_list(argv[++i], &o->layers)) {
                return false;
            }
        } else if (arg == "--experts") {
            if (!has_next) { fprintf(stderr, "error: %s needs a value\n", arg.c_str()); return false; }
            o->max_experts = atoi(argv[++i]);
            if (o->max_experts <= 0) {
                fprintf(stderr, "error: --experts must be positive\n");
                return false;
            }
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

// Every lookup that can fail prints what IS there. A tool that quietly reports zero error is
// worse than one that dies loudly - this project has already been fooled by a "0%" that was
// actually a broken reference.
void dump_tensor_names(gguf_context * gguf, const char * missing) {
    const int n = int(gguf_get_n_tensors(gguf));
    fprintf(stderr, "error: tensor '%s' not found. %d tensors present:\n", missing, n);
    for (int i = 0; i < n; ++i) {
        fprintf(stderr, "  %s\n", gguf_get_tensor_name(gguf, i));
    }
}

struct Stats {
    double min  = 0.0;
    double med  = 0.0;
    double mean = 0.0;
    double max  = 0.0;
    double ratio = 0.0;
};

Stats compute_stats(std::vector<double> v) {
    Stats s;
    if (v.empty()) {
        return s;
    }
    std::sort(v.begin(), v.end());
    s.min = v.front();
    s.max = v.back();
    s.med = (v.size() % 2 == 1) ? v[v.size() / 2]
                                : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    double sum = 0.0;
    for (double x : v) {
        sum += x;
    }
    s.mean  = sum / double(v.size());
    s.ratio = s.min > 0.0 ? s.max / s.min : 0.0;
    return s;
}

struct Buffers {
    std::vector<float>   src;   // dequantised stored weights
    std::vector<float>   round; // dequantised after requantisation
    std::vector<uint8_t> raw;   // stored bytes of one expert slab
    std::vector<uint8_t> qtmp;  // requantised bytes
};

// Returns the relative Frobenius error for one expert, or -1.0 on failure.
double expert_error(FILE * f,
                    const ggml_tensor * t,
                    size_t base_offset,
                    int expert,
                    ggml_type target,
                    Buffers * buf) {
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const size_t  slab = size_t(t->nb[2]);
    const size_t  off  = base_offset + size_t(expert) * slab;
    const int64_t n    = ne0 * ne1;

    // 64-bit seek is mandatory: the file is ~17 GB and a 32-bit long would silently wrap,
    // reading the wrong expert instead of failing.
#ifdef _WIN32
    const int seek_rc = _fseeki64(f, (__int64) off, SEEK_SET);
#else
    const int seek_rc = fseeko(f, (off_t) off, SEEK_SET);
#endif
    if (seek_rc != 0) {
        fprintf(stderr, "error: seek to %zu failed for expert %d\n", off, expert);
        return -1.0;
    }
    const size_t got = fread(buf->raw.data(), 1, slab, f);
    if (got != slab) {
        fprintf(stderr, "error: short read for expert %d: wanted %zu bytes, got %zu\n",
                expert, slab, got);
        return -1.0;
    }

    const ggml_type_traits_t src_traits = ggml_internal_get_type_traits(t->type);
    if (src_traits.to_float == nullptr) {
        fprintf(stderr, "error: no to_float for stored type %s\n", ggml_type_name(t->type));
        return -1.0;
    }
    // to_float must be called ONE ROW AT A TIME. Several of this fork's types (IQ4_KS and the
    // rest of the row-scale family) carry row_meta_size bytes of per-row metadata - a leading
    // float scale - ahead of the blocks. Their dequantiser reads that scale once at the start
    // of the pointer it is handed, so a single call spanning the whole slab decodes row 0 and
    // then reinterprets every later row's scale as quantised data. That produced a plausible
    // looking but meaningless ~117% error before this was fixed. Row-wise is also correct for
    // the plain block types, where row_meta_size is 0.
    const size_t src_row_stride = size_t(t->nb[1]);
    for (int64_t r = 0; r < ne1; ++r) {
        src_traits.to_float(buf->raw.data() + r * src_row_stride,
                            buf->src.data() + size_t(r * ne0), ne0);
    }

    const size_t written = ggml_quantize_chunk(target, buf->src.data(), buf->qtmp.data(),
                                               /*start=*/0, /*nrows=*/ne1, /*n_per_row=*/ne0,
                                               /*imatrix=*/nullptr, /*user_data=*/nullptr);
    const size_t dst_row_stride = ggml_row_size(target, ne0);
    const size_t expect = dst_row_stride * size_t(ne1);
    if (written != expect) {
        fprintf(stderr, "error: quantize_chunk wrote %zu bytes, expected %zu\n", written, expect);
        return -1.0;
    }

    const ggml_type_traits_t dst_traits = ggml_internal_get_type_traits(target);
    if (dst_traits.to_float == nullptr) {
        fprintf(stderr, "error: no to_float for target type %s\n", ggml_type_name(target));
        return -1.0;
    }
    for (int64_t r = 0; r < ne1; ++r) {
        dst_traits.to_float(buf->qtmp.data() + r * dst_row_stride,
                            buf->round.data() + size_t(r * ne0), ne0);
    }

    double num = 0.0;
    double den = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double w = double(buf->src[size_t(i)]);
        const double d = w - double(buf->round[size_t(i)]);
        num += d * d;
        den += w * w;
    }
    if (!(den > 0.0)) {
        fprintf(stderr, "error: expert %d has zero weight energy - refusing to report an error of 0\n",
                expert);
        return -1.0;
    }
    return std::sqrt(num / den);
}

} // namespace

int main(int argc, char ** argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) {
        return 1;
    }

    ggml_type target = GGML_TYPE_IQ4_KS;
    if (!parse_type(opt.type_name, &target)) {
        fprintf(stderr, "error: unknown --type '%s'. known:", opt.type_name.c_str());
        for (const TypeAlias & a : k_type_aliases) {
            fprintf(stderr, " %s", a.name);
        }
        fprintf(stderr, "\n");
        return 1;
    }

    // Not every type in the enum has a quantiser compiled into this build, and a few of the
    // low-bit IQ types will not quantise without an importance matrix. Refuse up front: a type
    // that silently produces nonsense is worse than one that declines to run at all.
    {
        const ggml_type_traits_t tt = ggml_internal_get_type_traits(target);
        if (tt.from_float == nullptr) {
            fprintf(stderr, "error: type %s has no from_float in this build - cannot requantise to it\n",
                    ggml_type_name(target));
            return 1;
        }
        if (tt.to_float == nullptr) {
            fprintf(stderr, "error: type %s has no to_float in this build - cannot measure its error\n",
                    ggml_type_name(target));
            return 1;
        }
        if (ggml_quantize_requires_imatrix(target)) {
            fprintf(stderr, "error: type %s requires an importance matrix, which this tool does not"
                            " supply - refusing rather than reporting a bogus error\n",
                    ggml_type_name(target));
            return 1;
        }
    }

    std::vector<std::string> kinds;
    if (opt.tensor_sel == "all") {
        kinds = { "up", "gate", "down" };
    } else {
        kinds = { opt.tensor_sel };
    }

    ggml_context *  meta_ctx = nullptr;
    gguf_init_params gp = { /*no_alloc=*/true, /*ctx=*/&meta_ctx };
    gguf_context * gguf = gguf_init_from_file(opt.model.c_str(), gp);
    if (gguf == nullptr || meta_ctx == nullptr) {
        fprintf(stderr, "error: gguf_init_from_file failed for '%s'\n", opt.model.c_str());
        return 1;
    }
    const size_t data_offset = gguf_get_data_offset(gguf);

    FILE * f = fopen(opt.model.c_str(), "rb");
    if (f == nullptr) {
        fprintf(stderr, "error: cannot open '%s' for reading\n", opt.model.c_str());
        gguf_free(gguf);
        return 1;
    }

    printf("model      : %s\n", opt.model.c_str());
    printf("target type: %s\n", ggml_type_name(target));
    printf("data offset: %zu\n", data_offset);
    printf("tensors    : %d\n", int(gguf_get_n_tensors(gguf)));
    printf("\n");

    Buffers buf;
    // Per (kind) accumulation across all sampled layers, for the final summary.
    std::vector<std::vector<double>> per_kind(kinds.size());
    // Per-expert error of the first sampled layer of each kind, for the best/worst listing.
    std::vector<std::vector<double>> first_layer(kinds.size());
    std::vector<int>                 first_layer_idx(kinds.size(), -1);

    bool any_ok = false;

    for (size_t ki = 0; ki < kinds.size(); ++ki) {
        const std::string & kind = kinds[ki];
        for (int layer : opt.layers) {
            char name[128];
            snprintf(name, sizeof(name), "blk.%d.ffn_%s_exps.weight", layer, kind.c_str());

            const int tid = gguf_find_tensor(gguf, name);
            if (tid < 0) {
                dump_tensor_names(gguf, name);
                fclose(f);
                gguf_free(gguf);
                return 1;
            }
            ggml_tensor * t = ggml_get_tensor(meta_ctx, name);
            if (t == nullptr) {
                fprintf(stderr, "error: '%s' is in the gguf index but ggml_get_tensor returned null\n", name);
                fclose(f);
                gguf_free(gguf);
                return 1;
            }
            if (t->ne[2] <= 0 || t->ne[3] != 1) {
                fprintf(stderr, "error: '%s' has unexpected shape [%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]"
                                " - expected a 3D stacked expert tensor\n",
                        name, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
                fclose(f);
                gguf_free(gguf);
                return 1;
            }

            const int64_t ne0 = t->ne[0];
            const int64_t ne1 = t->ne[1];
            const int     n_expert = int(t->ne[2]);
            const int     n_do = (opt.max_experts > 0) ? std::min(opt.max_experts, n_expert) : n_expert;
            const int64_t n_elem = ne0 * ne1;

            // If nb[1] is not the row size for the stored type our row-wise walk of the slab
            // would drift. Check rather than assume.
            const size_t stored_row_size = ggml_row_size(t->type, ne0);
            if (size_t(t->nb[1]) != stored_row_size) {
                fprintf(stderr, "error: '%s' nb[1]=%zu but ggml_row_size(%s, %" PRId64 ")=%zu\n",
                        name, size_t(t->nb[1]), ggml_type_name(t->type), ne0, stored_row_size);
                fclose(f);
                gguf_free(gguf);
                return 1;
            }
            if (size_t(t->nb[2]) != stored_row_size * size_t(ne1)) {
                fprintf(stderr, "error: '%s' nb[2]=%zu but expected %zu for %" PRId64 " rows\n",
                        name, size_t(t->nb[2]), stored_row_size * size_t(ne1), ne1);
                fclose(f);
                gguf_free(gguf);
                return 1;
            }
            if (ne0 % ggml_blck_size(target) != 0) {
                fprintf(stderr, "error: row length %" PRId64 " of '%s' is not a multiple of the %s block size %" PRId64 "\n",
                        ne0, name, ggml_type_name(target), int64_t(ggml_blck_size(target)));
                fclose(f);
                gguf_free(gguf);
                return 1;
            }

            // Allocate once per tensor shape, reuse across all experts.
            const size_t need_f   = size_t(n_elem);
            const size_t need_raw = size_t(t->nb[2]);
            const size_t need_q   = ggml_row_size(target, ne0) * size_t(ne1);
            if (buf.src.size()   < need_f)   { buf.src.assign(need_f, 0.0f); }
            if (buf.round.size() < need_f)   { buf.round.assign(need_f, 0.0f); }
            if (buf.raw.size()   < need_raw) { buf.raw.assign(need_raw, uint8_t(0)); }
            if (buf.qtmp.size()  < need_q)   { buf.qtmp.assign(need_q, uint8_t(0)); }

            const size_t base = data_offset + gguf_get_tensor_offset(gguf, tid);

            std::vector<double> errs;
            errs.reserve(size_t(n_do));
            for (int e = 0; e < n_do; ++e) {
                const double err = expert_error(f, t, base, e, target, &buf);
                if (err < 0.0) {
                    fprintf(stderr, "error: aborting at layer %d %s expert %d\n", layer, kind.c_str(), e);
                    fclose(f);
                    gguf_free(gguf);
                    return 1;
                }
                errs.push_back(err);
            }

            // Real footprint, not the nominal bit count: ggml_row_size adds row_meta_size (the
            // per-row float scale of the _KS family) on top of the block metadata, so two
            // nominally 4-bit types can differ measurably here.
            const double bpw = 8.0 * double(ggml_row_size(target, ne0)) / double(ne0);

            const Stats s = compute_stats(errs);
            printf("layer %2d  ffn_%-4s  stored %-8s  [%" PRId64 " x %" PRId64 " x %d]  experts %3d  "
                   "min %.6f  med %.6f  mean %.6f  max %.6f  max/min %.3f  target %.4f bpw (%.5f B/w)\n",
                   layer, kind.c_str(), ggml_type_name(t->type), ne0, ne1, n_expert, n_do,
                   s.min, s.med, s.mean, s.max, s.ratio, bpw, bpw / 8.0);
            fflush(stdout);

            per_kind[ki].insert(per_kind[ki].end(), errs.begin(), errs.end());
            if (first_layer_idx[ki] < 0) {
                first_layer_idx[ki] = layer;
                first_layer[ki]     = errs;
            }
            any_ok = true;
        }
        printf("\n");
    }

    fclose(f);

    if (!any_ok) {
        fprintf(stderr, "error: nothing was measured\n");
        gguf_free(gguf);
        return 1;
    }

    printf("=== summary across sampled layers (target %s) ===\n", ggml_type_name(target));
    for (size_t ki = 0; ki < kinds.size(); ++ki) {
        const Stats s = compute_stats(per_kind[ki]);
        printf("ffn_%-4s  n=%4zu  min %.6f  med %.6f  mean %.6f  max %.6f  max/min %.3f\n",
               kinds[ki].c_str(), per_kind[ki].size(), s.min, s.med, s.mean, s.max, s.ratio);
    }

    printf("\n=== best / worst experts, first sampled layer ===\n");
    for (size_t ki = 0; ki < kinds.size(); ++ki) {
        if (first_layer[ki].empty()) {
            continue;
        }
        std::vector<int> order(first_layer[ki].size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = int(i);
        }
        const std::vector<double> & v = first_layer[ki];
        std::sort(order.begin(), order.end(), [&v](int a, int b) { return v[size_t(a)] < v[size_t(b)]; });
        const size_t k = std::min<size_t>(5, order.size());

        printf("layer %d ffn_%s  worst %zu:", first_layer_idx[ki], kinds[ki].c_str(), k);
        for (size_t i = 0; i < k; ++i) {
            const int e = order[order.size() - 1 - i];
            printf("  e%d=%.6f", e, v[size_t(e)]);
        }
        printf("\n");
        printf("layer %d ffn_%s  best  %zu:", first_layer_idx[ki], kinds[ki].c_str(), k);
        for (size_t i = 0; i < k; ++i) {
            const int e = order[i];
            printf("  e%d=%.6f", e, v[size_t(e)]);
        }
        printf("\n");
    }

    ggml_quantize_free();
    gguf_free(gguf);
    return 0;
}
