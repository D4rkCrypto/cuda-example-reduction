#include "reduce1.cuh"
#include "reduce2.cuh"
#include "reduce3.cuh"
#include "reduce4.cuh"
#include "reduce5.cuh"
#include "reduce6.cuh"
#include "reduce7.cuh"

int main() {
    // print timer banner
    print_timer_banner();

    reduce1();
    reduce2();
    reduce3();
    reduce4();
    reduce5();
    reduce6();
    reduce7();

    return 0;
}
