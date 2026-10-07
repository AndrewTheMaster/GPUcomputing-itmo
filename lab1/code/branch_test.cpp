#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <inttypes.h>
#include <string>
#include <vector>

/*
 * Тест предсказателя переходов.
 *
 * Массив data заполняется двумя способами:
 *   predictable - условия имеют регулярный характер (сначала все 0, затем все 1);
 *   random      - условия сформированы псевдослучайной последовательностью.
 *
 * Измеряемый цикл:
 *      if (data[i]) sum += data[i];
 *
 * Число переходов и доля ошибочных предсказаний измеряются утилитой perf.
 */

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s N [predictable|random] [repeats]\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t n        = std::atoll(argv[1]);
    std::string mode = argv[2];
    size_t repeats  = (argc > 3) ? std::atoll(argv[3]) : 1;

    std::vector<int> data(n);

    if (mode == "predictable") {
        for (size_t i = 0; i < n; i++) {
            data[i] = (i < n / 2) ? 0 : 1;
        }
    } else {
        /* xorshift64 - простая воспроизводимая псевдослучайная последовательность */
        uint64_t s = 88172645463325252ULL;
        for (size_t i = 0; i < n; i++) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            data[i] = static_cast<int>(s & 1ULL);
        }
    }

    volatile long long sum = 0;
    for (size_t r = 0; r < repeats; r++) {
        long long acc = 0;
        for (size_t i = 0; i < n; i++) {
            if (data[i]) {
                acc += data[i];
            }
        }
        sum += acc;
    }

    printf("mode : %s\n", mode.c_str());
    printf("n : %zu\n", n);
    printf("repeats : %zu\n", repeats);
    printf("sum = %lld\n", static_cast<long long>(sum));

    return EXIT_SUCCESS;
}
