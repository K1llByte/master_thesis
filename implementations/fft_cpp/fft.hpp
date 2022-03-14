#ifndef FFT_HPP
#define FFT_HPP

#include <array>
#include <type_traits>
#include <concepts>
#include <cmath>

#include "utils.hpp"

namespace recursive
{

    ///////////////// Forward Recursive FFT /////////////////

    template <std::size_t N>
    std::array<vec2, N> fft(const std::array<float, N>& p)
        // requires (is_power_of_2<N>())
    {
        // 1.
        if(N == 1)
            return to_complex(p);
        // 2.
        const size_t HALF_N = N/2;
        std::array<float, HALF_N> pe;
        std::array<float, HALF_N> po;
        for(std::size_t i = 0 ; i < N ; ++i)
        {
            
            if(i % 2 == 0) // Even
            // if((i & 1) == 0) // Even
                pe[i/2] = p[i];
            else // Odd
                po[(i-1)/2] = p[i];
        }
        // 3.
        auto ye = fft(pe);
        auto yo = fft(po);
        // 4.
        std::array<vec2, N> y;
        // 5.
        for (std::size_t j = 0 ; j < HALF_N ; ++j)
        {
            const auto wj = euler(-2 * pi / N * j);
            y[j] = vec2{
                ye[j][0] + wj[0]*yo[j][0] - wj[1]*yo[j][1],
                ye[j][1] + wj[0]*yo[j][1] + wj[1]*yo[j][0]
            };
            y[j + HALF_N] = vec2{
                ye[j][0] - wj[0]*yo[j][0] + wj[1]*yo[j][1],
                ye[j][1] - wj[0]*yo[j][1] - wj[1]*yo[j][0]
            };
        }

        return y;
    }

    ///////////////// Inverse Recursive FFT /////////////////

    template <std::size_t N>
    std::array<vec2, N> ifft_aux(const std::array<vec2, N>& p)
    {
        // 1.
        if(N == 1)
            return p;
        // 2.
        const size_t HALF_N = N/2;
        std::array<vec2, HALF_N> pe;
        std::array<vec2, HALF_N> po;
        for(std::size_t i = 0 ; i < N ; ++i)
        {
            if(i % 2 == 0) // Even
            // if((i & 1) == 0) // Even
                pe[i/2] = p[i];
            else // Odd
                po[(i-1)/2] = p[i];
        }
        // 3.
        auto ye = ifft_aux(pe);
        auto yo = ifft_aux(po);
        // 4.
        std::array<vec2, N> y;
        // 5.
        for (std::size_t j = 0 ; j < HALF_N ; ++j)
        {
            const auto wj = euler(2 * pi / N * j);
            y[j] = vec2{
                ye[j][0] + wj[0]*yo[j][0] - wj[1]*yo[j][1],
                ye[j][1] + wj[0]*yo[j][1] + wj[1]*yo[j][0]
            };
            y[j + HALF_N] = vec2{
                ye[j][0] - wj[0]*yo[j][0] + wj[1]*yo[j][1],
                ye[j][1] - wj[0]*yo[j][1] - wj[1]*yo[j][0]
            };
        }
        return y;
    }

    template <std::size_t N>
    std::array<vec2, N> ifft(const std::array<vec2, N>& input)
        // requires (is_power_of_2<N>())
    {
        auto output = ifft_aux(input);
        // Structured bindings for real and imaginary part
        for(auto& [r, i] : output)
        {
            // Only real part needs the division
            // since the imaginary component will be 0
            r /= N;
        }
        return output;
    }

}

/////////////////////////////////////////////////////////

#endif // FFT_HPP