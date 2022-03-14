#ifndef DFT_HPP
#define DFT_HPP

#include <cstdint>
#include <vector>
#include "utils.hpp"

namespace dft
{
    ////////////////////// Forward DFT //////////////////////

    template<std::size_t N>
    std::array<vec2, N> fft(const std::array<float, N>& in)
    {
        // 1. Initialize twiddle matrix
        std::array<vec2, N> res{};
        constexpr auto twiddle_matrix = [&]{
            std::vector<vec2> mat(N*N);
            for(std::size_t i = 0; i < N; ++i)
            {
                for(std::size_t j = 0; j < N; ++j)
                {
                    mat[i*N + j] = euler(-2 * pi * i * j / N);
                }
            }
            return mat;
        }();
        
        // 2. Dot product
        for(std::size_t i = 0; i < N; ++i)
        {
            for(std::size_t j = 0; j < N; ++j)
            {
                auto tmp = complex_mul(twiddle_matrix[i*N + j], vec2{in[j], 0});
                res[i] = complex_add(res[i], tmp);
            }
        }
        return res;
    }

    ////////////////////// Inverse DFT //////////////////////

    template<std::size_t N>
    std::array<vec2, N> ifft(const std::array<vec2, N>& in)
    {
        // 1. Initialize twiddle matrix
        std::array<vec2, N> res{};
        auto twiddle_matrix = [&]{
            std::vector<vec2> mat(N*N);
            for(std::size_t i = 0; i < N; ++i)
            {
                for(std::size_t j = 0; j < N; ++j)
                {
                    mat[i*N + j] = euler(2 * pi * i * j / N);
                }
            }
            return mat;
        }();
        
        // 2. Dot product
        for(std::size_t i = 0; i < N; ++i)
        {
            for(std::size_t j = 0; j < N; ++j)
            {
                auto tmp = complex_mul(twiddle_matrix[i*N + j], in[j]);
                res[i] = complex_add(res[i], tmp);
            }
        }

        // Structured bindings for real and imaginary part
        for(auto& [r, _] : res)
        {
            // Only real part needs the division
            // since the imaginary component will be 0
            r /= N;
        }

        return res;
    }

}

/////////////////////////////////////////////////////////

#endif // FFT_DIF_ITER_CPP