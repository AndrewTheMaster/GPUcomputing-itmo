#include <cstdlib>
#include <cstdio>
#include <inttypes.h>
#include <string>
#include <vector>

/*
 * DAXPY-подобная вычислительная нагрузка:
 *
 *      y[i] = a * x[i] + y[i],   i = 0, stride, 2*stride, ...
 *
 * Параметры:
 *   n       - число элементов массивов;
 *   repeats - число повторов вычисления;
 *   stride  - шаг обращения к элементам массива;
 *   type    - тип элемента: char | float | double (по умолчанию double).
 *
 * Рабочий набор при stride = 1 составляет W = 2 * n * sizeof(T) байт
 * (массивы x и y).
 */

template <typename T>
void daxpy(size_t n, size_t stride, T a, T *x, T *y) {
    for (size_t i = 0; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

template <typename T>
void run_daxpy(size_t n, size_t stride, size_t repeats) {
    std::vector<T> x(n), y(n);

    printf("input size in bytes : %zu\n", n * sizeof(T));

    T a = static_cast<T>(3);

    /* Инициализация массивов, чтобы измеряемая нагрузка работала
     * с определёнными данными (исключаем чтение неинициализированной памяти). */
    for (size_t i = 0; i < n; i++) {
        x[i] = static_cast<T>((i % 17) + 1);
        y[i] = static_cast<T>((i % 13) + 1);
    }

    for (size_t i = 0; i < repeats; i++) {
        daxpy(n, stride, a, x.data(), y.data());
    }

    /* Контрольная сумма не позволяет компилятору удалить вычисления. */
    double sum = 0.0;
    for (size_t i = 0; i < n; i += stride) {
        sum += static_cast<double>(y[i]);
    }
    printf("checksum : %.6f\n", sum);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s N repeats stride [char|float|double]\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t n       = std::atoll(argv[1]);
    size_t repeats = std::atoll(argv[2]);
    size_t stride  = std::atoll(argv[3]);
    std::string type = (argc > 4) ? argv[4] : "double";

    printf("n : %zu\n", n);
    printf("repeats : %zu\n", repeats);
    printf("stride : %zu\n", stride);
    printf("type : %s\n", type.c_str());

    if (type == "char") {
        run_daxpy<char>(n, stride, repeats);
    } else if (type == "float") {
        run_daxpy<float>(n, stride, repeats);
    } else if (type == "double") {
        run_daxpy<double>(n, stride, repeats);
    } else {
        fprintf(stderr, "unknown type: %s\n", type.c_str());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
