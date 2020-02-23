#include <iostream>
#include <cassert>

long factorial(long num)
{
    assert(num>=0);

    long fact = 1;
    for(long i = 1; i <= num; i++) {
        fact *= i;
    }
    return fact;
}

long combination(long n, long m)
{
    if (n<m) {
        return 0;
    } else {
        long fact = 1;
        for(long i = 0; i < m; i++) {
            fact *= (n-i);
        }
        return fact/factorial(m);
    }
}