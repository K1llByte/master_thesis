#ifndef UTILS_HPP
#define UTILS_HPP

#include <array>
#include <cmath>
#include <type_traits>

// Auxiliar definitions and constants
using vec2 = std::array<float, 2>;
constexpr float pi = 3.14159265358979323846;


// // Auxiliar function to check if a templated const
// // generic is power of 2
// template<typename T>
//     requires (std::is_integral_v<T> && std::is_unsigned_v<T>)
template<std::size_t N>
constexpr bool is_power_of_2()
{
    return N && !(N & (N-1));
}


// Prints an array of N printable elements
template<typename T, std::size_t N>
void print_array(const std::array<T,N>& arr)
{
    std::cout << "[ ";
    for(size_t i = 0 ; i < N ; ++i)
        std::cout << arr[i] << (i == N-1 ? " ]\n" : ", ");
}


// Prints an array of N complex vec2 elements
template<std::size_t N>
void print_complex_array(const std::array<vec2,N>& arr)
{
    std::cout << "[";
    for(size_t i = 0 ; i < N ; ++i)
    {
        if (arr[i][1] < 0) {
            std::cout << '(' << arr[i][0] << arr[i][1] << "*i" << ')' << (i == N-1 ? "]\n" : ", ");
        }
        else {
            std::cout << '(' << arr[i][0] << '+' << arr[i][1] << "*i" << ')' << (i == N-1 ? "]\n" : ", ");
        }
    }
}


// Applies euler's formula
constexpr vec2 euler(const float angle)
{
    return vec2{std::cos(angle), std::sin(angle)};
}


// Converts an array of N floating point elements to N complex elements
// Which sets the imaginary component to 0
template <std::size_t N>
constexpr std::array<vec2, N> to_complex(const std::array<float, N>& real)
{
    std::array<vec2, N> complex;
    for(std::size_t i = 0 ; i < N ; ++i)
        complex[i][0] = real[i];
    return complex;
}


// Complex multiplication
constexpr vec2 complex_mul(const vec2& a, const vec2& b)
{
    return vec2{
        a[0]*b[0] - a[1]*b[1],
        a[0]*b[1] + a[1]*b[0]
    };
}

// Complex sum
constexpr vec2 complex_add(const vec2& a, const vec2& b)
{
    return vec2{
        a[0] + b[0],
        a[1] + b[1],
    };
}

// Complex subtraction
constexpr vec2 complex_sub(const vec2& a, const vec2& b)
{
    return vec2{
        a[0] - b[0],
        a[1] - b[1],
    };
}

// Bit reversal funciton
// note: if N is the size of the sequence
// then num_bits will be log2(N)
std::size_t bit_reverse(std::size_t v, std::size_t num_bits)
{
    // NOTE: This is an iterative implementation,
    // if the input value is just 8 bits then
    // change to lookup table implementation
    std::size_t r = 0;
    for (std::size_t i = 0 ; i < num_bits ; ++i) {
        if(v & (1 << i))
            r |= (1 << ((num_bits - 1) - i));
    }
    return r;
}

template<typename T>
void swap(T& a, T& b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

// Inplace Array Bit reversal for any T array
template<std::size_t N, typename T>
void inplace_bit_reversal(std::array<T,N>& arr)
{
    constexpr auto num_bits = static_cast<std::size_t>(std::log2(N));
    constexpr auto HALF_N = N/2;
    for(std::size_t i = 1; i < HALF_N; ++i)
    {
        const auto br_i = bit_reverse(i, num_bits);
        swap(arr[br_i], arr[i]);
    }
}

#endif // UTILS_HPP