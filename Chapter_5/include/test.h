#ifndef TEST_H_
#define TEST_H_
struct test {
    int a;
    int b;
    int c[64];
};

extern __shared__ test t;
#endif