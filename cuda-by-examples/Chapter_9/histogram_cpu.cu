#include "book.h"
#include <iostream>
#include <chrono>

constexpr int SIZE = 100 * 1024 * 1024;

int main () {
    unsigned char* buffer = (unsigned char*)big_random_block (SIZE);
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int histogram[256];
    memset (histogram, 0, sizeof (histogram));
    for (int i = 0; i < SIZE; i++) {
        histogram[buffer[i]]++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Elapsed Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " ms\n";
    long long histoCnt = 0;
    for (int i = 0; i < 256; i++) {
        histoCnt += histogram[i];
    }
    printf ("Histogram Sum: %lld\n", histoCnt);
    return 0;
}