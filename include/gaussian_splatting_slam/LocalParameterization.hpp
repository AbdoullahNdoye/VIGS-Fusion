#pragma once
#include <ceres/ceres.h>
#include <Eigen/Core>

namespace gaussian_splatting_slam
{
	class LocalParameterizationBase : public ceres::LocalParameterization
	{
	public:
		/**
		 * @brief Computes the distance between two parameter values in local (tangent) space.
		 * @param xi Left-hand-side parameter
		 * @param xj Right-hand-side parameter
		 * @param xi_minus_xj LHS [boxminus] RHS
		 */
		virtual void boxMinus(const double *xi, const double *xj,
							  double *xi_minus_xj) const = 0;

		/**
		 * @brief Computes the derivative of the boxminus operation with respect to the left-hand-side parameter
		 * @param xi Left-hand-side parameter
		 * @param xj Right-hand-side parameter
		 * @return The derivative of LHS [boxminus] RHS with repsect to LHS parameter
		 */
		virtual Eigen::MatrixXd boxMinusJacobianLeft(double const *xi, double const *xj) const = 0;

		/**
		 * @brief Computes the derivative of the boxminus operation with respect to the right-hand-side parameter
		 * @param xi Left-hand-side parameter
		 * @param xj Right-hand-side parameter
		 * @return The derivative of LHS [boxminus] RHS with repsect to RHS parameter
		 */
		virtual Eigen::MatrixXd boxMinusJacobianRight(double const *xi, double const *xj) const = 0;
	};

}; // namespace gaussian_splatting_slam
