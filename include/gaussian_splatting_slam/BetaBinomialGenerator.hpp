#pragma once

#include <random>
#include <cmath>

namespace gaussian_splatting_slam
{
    /**
     * @brief Generator for Beta-Binomial distribution
     *
     * This class generates random integers following a Beta-Binomial distribution
     * using the composition method: first sample from Beta(alpha, beta), then
     * sample from Binomial(n, p) where p is the beta sample.
     */
    class BetaBinomialGenerator
    {
    private:
        std::mt19937 gen; // Mersenne Twister random number generator

        /**
         * @brief Sample from Beta distribution using gamma method
         * @param alpha Alpha parameter of beta distribution
         * @param beta Beta parameter of beta distribution
         * @return Random sample from Beta(alpha, beta)
         */
        double sampleBeta(double alpha, double beta);

    public:
        /**
         * @brief Default constructor with random seed
         */
        BetaBinomialGenerator();

        /**
         * @brief Constructor with specified seed
         * @param seed Seed for random number generator
         */
        BetaBinomialGenerator(unsigned int seed);

        /**
         * @brief Generate an integer according to beta-binomial distribution
         * @param n Number of trials
         * @param alpha Alpha parameter of underlying beta distribution
         * @param beta Beta parameter of underlying beta distribution
         * @return Random integer from BetaBinomial(n, alpha, beta)
         */
        int sampleBetaBinomial(int n, double alpha, double beta);

        /**
         * @brief Version with seed for reproducibility
         * @param n Number of trials
         * @param alpha Alpha parameter of underlying beta distribution
         * @param beta Beta parameter of underlying beta distribution
         * @param seed Seed for this specific sample
         * @return Random integer from BetaBinomial(n, alpha, beta)
         */
        int sampleBetaBinomial(int n, double alpha, double beta, unsigned int seed);
    };
} // namespace gaussian_splatting_slam