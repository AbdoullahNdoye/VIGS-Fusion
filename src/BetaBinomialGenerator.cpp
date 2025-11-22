#include <gaussian_splatting_slam/BetaBinomialGenerator.hpp>
#include <random>

namespace gaussian_splatting_slam
{
    // Default constructor with random seed
    BetaBinomialGenerator::BetaBinomialGenerator() : gen(std::random_device{}())
    {
    }

    // Constructor with specified seed
    BetaBinomialGenerator::BetaBinomialGenerator(unsigned int seed) : gen(seed)
    {
    }

    // Sample from Beta distribution using gamma method
    double BetaBinomialGenerator::sampleBeta(double alpha, double beta)
    {
        std::gamma_distribution<double> gamma_a(alpha, 1.0);
        std::gamma_distribution<double> gamma_b(beta, 1.0);

        double x = gamma_a(gen);
        double y = gamma_b(gen);

        // If X ~ Gamma(alpha) and Y ~ Gamma(beta), then X/(X+Y) ~ Beta(alpha, beta)
        return x / (x + y);
    }

    // Generate an integer according to beta-binomial distribution
    int BetaBinomialGenerator::sampleBetaBinomial(int n, double alpha, double beta)
    {
        // Step 1: Draw p from Beta(alpha, beta)
        double p = sampleBeta(alpha, beta);

        // Step 2: Draw X from Binomial(n, p)
        std::binomial_distribution<int> binomial(n, p);
        return binomial(gen);
    }

    // Version with seed for reproducibility
    int BetaBinomialGenerator::sampleBetaBinomial(int n, double alpha, double beta, unsigned int seed)
    {
        gen.seed(seed);
        return sampleBetaBinomial(n, alpha, beta);
    }

} // namespace gaussian_splatting_slam