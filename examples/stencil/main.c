#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "stencil.h"

int main(int argc, char *argv[]) {
    int PROBLEM_SIZE = atoi(argv[1]);

    int n = PROBLEM_SIZE;
    int m = PROBLEM_SIZE;

    int c = CHANNELS;
    int h = FILTER_HEIGHT;
    int w = FILTER_WIDTH;

    REAL *src    = (REAL *) malloc(sizeof(REAL) * c * m * n);
    REAL *dst    = (REAL *) malloc(sizeof(REAL) * c * m * n);

    // init filter
    const float filter[FILTER_HEIGHT][FILTER_WIDTH] = {
        { 0, 0, 1, 0, 0, },
        { 0, 0, 2, 0, 0, },
        { 3, 4, 5, 6, 7, },
        { 0, 0, 8, 0, 0, },
        { 0, 0, 9, 0, 0, },
    };

    // init src
    initialize(c, m, n, src);
    // test
    stencil_rtune(src, dst, c, m, n, filter[0], h, w);

    return 0;
}
