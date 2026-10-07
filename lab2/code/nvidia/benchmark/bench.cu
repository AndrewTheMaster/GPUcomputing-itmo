#include "gpu_runner.cuh"
#include <chrono>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>

using namespace lab;
using Clock = std::chrono::steady_clock;
namespace {
struct Options {
    std::size_t n = 1u << 20, n2 = static_cast<std::size_t>(-1);
    int threads = 256, repeat = 10, warmup = 3, bins = 256;
    std::uint32_t seed = 42;
    std::string impl = "all", pattern = "random", csv;
};
std::uint64_t number(const std::string& s) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Expected non-negative integer, got: " + s);
    std::size_t end = 0;
    const auto v = std::stoull(s, &end);
    if (end != s.size()) throw std::invalid_argument("Invalid number: " + s);
    return v;
}
int bounded_int(const std::string& s, int lo, int hi) {
    const auto v = number(s);
    if (v < static_cast<unsigned>(lo) || v > static_cast<unsigned>(hi))
        throw std::invalid_argument("Value outside supported range: " + s);
    return static_cast<int>(v);
}
void help(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
        << "Selected algorithm: " << name(algorithm) << "\n"
        << "  --n N          input length (default 1048576; for merge: length A)\n"
        << "  --n2 N         length B for merge (default: same as --n)\n"
        << "  --threads B    32,64,128,256,512 (student only; default 256)\n"
        << "  --repeat R     measured repetitions, 1..10000 (default 10)\n"
        << "  --warmup W     warmups, 0..1000 (default 3)\n"
        << "  --bins M       histogram bins, 1..65536 (default 256)\n"
        << "  --seed S       deterministic input seed (default 42)\n"
        << "  --pattern P    random; variant-specific: zeros/ones,\n"
        << "                 one-segment/singletons, sorted/reverse, ties, one-bin\n"
        << "  --impl I       all|reference|cpu|student|thrust (default all)\n"
        << "  --csv PATH     append raw validated measurements to this CSV\n"
        << "                 default: results/<algorithm>.csv\n";
}
Options parse(int argc, char** argv) {
    Options o;
    o.csv = "results/" + std::string(name(algorithm)) + ".csv";
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i+1 == argc) throw std::invalid_argument("Missing value for " + key);
        const std::string v = argv[++i];
        if (key == "--n") o.n = number(v);
        else if (key == "--n2") o.n2 = number(v);
        else if (key == "--threads") o.threads = bounded_int(v,32,512);
        else if (key == "--repeat") o.repeat = bounded_int(v,1,10000);
        else if (key == "--warmup") o.warmup = bounded_int(v,0,1000);
        else if (key == "--bins") o.bins = bounded_int(v,1,65536);
        else if (key == "--seed") {
            auto s = number(v); if (s > 0xffffffffULL) throw std::invalid_argument("seed exceeds uint32");
            o.seed = static_cast<std::uint32_t>(s);
        }
        else if (key == "--impl") o.impl = v;
        else if (key == "--pattern") o.pattern = v;
        else if (key == "--csv") o.csv = v;
        else throw std::invalid_argument("Unknown option: " + key);
    }
    if (o.threads!=32 && o.threads!=64 && o.threads!=128 && o.threads!=256 && o.threads!=512)
        throw std::invalid_argument("threads must be 32,64,128,256,512");
    if (algorithm != Algorithm::group_by && o.n2 != static_cast<std::size_t>(-1))
        throw std::invalid_argument("--n2 is only available for group_by (merge)");
    if (o.csv.empty()) throw std::invalid_argument("CSV path must not be empty");
    return o;
}
std::vector<std::string> implementations(const std::string& i) {
    if (i == "all") return {"cpu", "student", "thrust"};
    if (i == "reference") return {"cpu", "thrust"};
    if (i == "cpu" || i == "student" || i == "thrust") return {i};
    throw std::invalid_argument("Unknown implementation: " + i);
}
double millis(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}
std::string quote(const std::string& s) {
    std::string r = "\""; for (char ch : s) { if (ch=='\"') r += '"'; r += ch; } return r+'"';
}
struct Sample {
    std::string impl;
    int repeat;
    double op_ms, device_ms, e2e_ms, error;
};
const char* header = "run_id,algorithm,implementation,n_a,n_b,elements,bins,threads,repeat,seed,pattern,operation_ms,device_ms,end_to_end_ms,max_abs_error,valid,gpu,compute_capability,cuda_runtime,driver";
void write_csv(const Options& o, const Input& x, const std::vector<Sample>& rows,
               const std::string& gpu, const std::string& cc, int runtime, int driver) {
    const std::filesystem::path path(o.csv);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    const bool exists = std::filesystem::exists(path) && std::filesystem::file_size(path) > 0;
    if (exists) {
        std::ifstream old(path); std::string first; std::getline(old,first);
        if (!first.empty() && first.back()=='\r') first.pop_back();
        if (first != header) throw std::runtime_error("Existing CSV has a different schema; choose a new path");
    }
    std::ofstream file(path,std::ios::app);
    if (!file) throw std::runtime_error("Cannot write CSV: " + o.csv);
    if (!exists) file << header << '\n';
    const auto run_id = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    file << std::setprecision(10);
    for (const auto& r : rows) {
        file << run_id << ',' << name(algorithm) << ',' << r.impl << ',' << x.a.size() << ','
             << x.b.size() << ',' << elements(x) << ',' << (algorithm==Algorithm::histogram ? x.bins : 0) << ','
             << (r.impl=="student" ? o.threads : 0) << ',' << r.repeat << ',' << o.seed << ',' << quote(o.pattern) << ','
             << r.op_ms << ',';
        if (r.impl != "cpu") file << r.device_ms;
        file << ',' << r.e2e_ms << ',' << r.error << ",1," << quote(gpu) << ',' << quote(cc) << ',' << runtime << ',' << driver << '\n';
    }
    file.flush(); if (!file) throw std::runtime_error("Failed while writing CSV");
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") { help(argv[0]); return 0; }
    try {
        const auto o = parse(argc,argv);
        const auto impls = implementations(o.impl);
        const bool gpu_needed = !(impls.size()==1 && impls[0]=="cpu");
        const auto input = make_input(algorithm,o.n,o.seed,o.bins,o.pattern,o.n2);
        auto expected = allocate_output(algorithm,input); run_cpu(algorithm,input,expected);
        auto output = allocate_output(algorithm,input);
        std::unique_ptr<GpuRunner> gpu;
        std::unique_ptr<Event> start,stop;
        std::string gpu_name = "not used", cc = ""; int runtime = 0,driver = 0;
        if (gpu_needed) {
            const auto p = device_properties(); check_block(o.threads);
            gpu_name = p.name; cc = std::to_string(p.major)+"."+std::to_string(p.minor);
            CUDA_CHECK(cudaRuntimeGetVersion(&runtime)); CUDA_CHECK(cudaDriverGetVersion(&driver));
            gpu = std::make_unique<GpuRunner>(input);
            start = std::make_unique<Event>(); stop = std::make_unique<Event>();
        }
        auto run_one = [&](const std::string& impl, int rep) -> Sample {
            Sample s{impl,rep,0,0,0,0};
            if (impl=="cpu") {
                const auto t0 = Clock::now();
                std::atomic_signal_fence(std::memory_order_seq_cst);
                run_cpu(algorithm,input,output);
                std::atomic_signal_fence(std::memory_order_seq_cst);
                const auto t1=Clock::now();
                s.op_ms = s.e2e_ms = millis(t0,t1);
            } else {
                const auto full0 = Clock::now();
                gpu->upload(); // Input reset and H2D are included only in the end-to-end interval.
                CUDA_CHECK(cudaEventRecord(start->get(),0));
                const auto op0 = Clock::now();
                if (impl=="student") gpu->run_student(o.threads); else gpu->run_thrust();
                CUDA_CHECK(cudaEventRecord(stop->get(),0));
                CUDA_CHECK(cudaEventSynchronize(stop->get()));
                const auto op1 = Clock::now();
                gpu->download(output,impl=="student");
                const auto full1 = Clock::now();
                float dev_ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&dev_ms,start->get(),stop->get()));
                s.op_ms = millis(op0,op1); s.e2e_ms = millis(full0,full1); s.device_ms = dev_ms;
                // Integrity checks are deliberately outside measured intervals.
                gpu->require_inputs_unchanged();
            }
            const auto check = verify(algorithm,input,expected,output);
            if (!check.ok) throw std::runtime_error(impl+": "+check.message+"; CSV not written for this run");
            s.error = check.max_abs_error;
            return s;
        };
        for (int w=0; w<o.warmup; ++w) for (const auto& i:impls) (void)run_one(i,-1);
        std::vector<Sample> rows; rows.reserve(o.repeat*impls.size());
        for (int r=0; r<o.repeat; ++r) for (const auto& i:impls) rows.push_back(run_one(i,r));
        write_csv(o,input,rows,gpu_name,cc,runtime,driver);
        std::cout << name(algorithm) << ": correctness OK; " << rows.size() << " measured rows -> " << o.csv << '\n';
        for (const auto& i:impls) {
            double a=0,b=0; int n=0;
            for (const auto& r:rows) if (r.impl==i) { a+=r.op_ms; b+=r.e2e_ms; ++n; }
            std::cout << i << ": mean operation=" << a/n << " ms, end_to_end=" << b/n << " ms\n";
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
}
