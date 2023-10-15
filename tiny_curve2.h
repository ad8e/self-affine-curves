#include "console.h"
#include <ceres/tiny_solver.h>
#include <ceres/tiny_solver_autodiff_function.h>
#include <numbers>
#include <utility>

#if !_EMSCRIPTEN
#define EMSCRIPTEN_KEEPALIVE
#endif

using namespace ceres;
using Eigen::Matrix;
using std::abs;
using std::array;
using std::cos;
using std::exp;
using std::sin;
using std::sqrt;

auto sq(auto i) { return i * i; }
using std::numbers::pi;

constexpr double zero_threshold = 1e-8; //arbitrary!

template <typename T>
Matrix<T, 2, 1> apply_curve(const Matrix<T, 2, 1>& a, const Matrix<T, 2, 1>& b, const T& p, const T& q, double t, int mode) {
	if (mode == 0) {
		//(-a + b) + a e^(p t) + b e^(q t)
		return a * (-1 + exp(p * t)) + b * (-1 + exp(q * t));
	}
	else if (mode == 1) {
		//-a + e^(p t)(a cos (q t) + b sin(q t))
		return -a + exp(p * t) * (a * cos(q * t) + b * sin(q * t));
	}
	else if (mode == 2) {
		//-a + e^(p t)(a + b t)
		return -a + exp(p * t) * (a + b * t);
	}
	else if (mode == 3) {
		//a t + b t^2
		return t * (a + b * t);
	}
	std::unreachable();
}

//returns dt, dt^2
template <typename T>
array<Matrix<T, 2, 1>, 2> differentials(const Matrix<T, 2, 1>& a, const Matrix<T, 2, 1>& b, const T& p, const T& q, double t, int mode) {
	Matrix<T, 2, 1> dt, dt2;
	if (mode == 0) {
		//(-a + b) + a e^(p t) + b e^(q t)
		dt = a * p * exp(p * t) + b * q * exp(q * t);
		dt2 = a * p * p * exp(p * t) + b * q * q * exp(q * t);
	}
	else if (mode == 1) {
		//-a + e^(p t)(a cos (q t) + b sin(q t))
		dt = exp(p * t) * ((a * p + b * q) * cos(q * t) + (b * p - a * q) * sin(q * t));
		dt2 = exp(p * t) * ((a * p * p + 2. * b * p * q - a * q * q) * cos(q * t) + (b * p * p - 2. * a * p * q - b * q * q) * sin(q * t));
	}
	else if (mode == 2) {
		dt = exp(p * t) * (b + a * p + b * p * t);
		dt2 = exp(p * t) * p * (a * p + b * (2. + p * t));
	}
	else if (mode == 3) {
		dt = a + 2. * b * t;
		dt2 = 2. * b;
	}
	return {dt, dt2};
}

//future: can speed this up by considering the two boundaries. where many of the terms become zero.
template <typename T>
T curvature_from_differentials(Matrix<T, 2, 1>& dt, Matrix<T, 2, 1>& dt2) {
	return (dt[0] * dt2[1] - dt[1] * dt2[0]) / pow(sq(dt[0]) + sq(dt[1]), 1.5);
}

template <typename T>
T calculate_curvatures(Matrix<T, 2, 1>& a, Matrix<T, 2, 1>& b, T& p, T& q, double t, int mode) {
	auto [dt, dt2] = differentials(a, b, p, q, t, mode);
	//if constexpr (std::is_same_v<T, double>)
	//	outc("differentials", dt[0], dt[1], dt2[0], dt2[1], "curvature", curvature_from_differentials(dt, dt2));
	return curvature_from_differentials(dt, dt2);
}

//a, b, p, q, mode
template <typename T>
std::tuple<Matrix<T, 2, 1>, Matrix<T, 2, 1>, T, T, int> decompress(const T& m, const T& r) {
	int mode;
	T p;
	T q;
	Matrix<T, 2, 1> a;
	Matrix<T, 2, 1> b;

	//if r > 0, use real. with p=m+r, q=m-r.
	//if r < 0, use imag
	//if r == 0 and m =/= 0, use down one.
	//if r == 0 and m == 0, use parabola.
	//if r == +-m, use half and half.
	//I expect the smoothness should work out.

	//todo: when is this reasonable?
	//todo: it's missing a mode. c + a t + b e^t
	//	if (abs(10000. * r) < abs(m) || abs(r) < zero_threshold) {
	//		if (abs(m) >= zero_threshold) {
	//			mode = 2;
	//			p = m;
	//			q = T(0.);
	//			//-a + e^(p t)(a + b t)
	//			a[0] = -(1. + p) / (1. - exp(p) + p);
	//			b[0] = -p / (-1. + exp(p) - p);
	//			a[1] = -2. / (2. - 2. * exp(p) + exp(p) * p);
	//			b[1] = p / (2. - 2. * exp(p) + exp(p) * p);
	//		}
	//		else {
	//			mode = 3;
	//			p = T(0.);
	//			q = T(0.);
	//			//a t + b t^2
	//			a[0] = T(2.);
	//			b[0] = T(-1.);
	//			a[1] = T(0.);
	//			b[1] = T(1.);
	//		}
	//	}
	//	else
	if (r > 0) {
		mode = 0;
		p = m + r;
		q = m - r;
		//(-a + b) + a e^(p t) + b e^(q t)
		T denomx = exp(p) * p - exp(q) * q + exp(p + q) * (-p + q);
		a[0] = exp(q) * q / denomx;
		b[0] = -exp(p) * p / denomx;
		T denomy = (p * (1. - exp(q)) + q * (-1. + exp(p)));
		a[1] = q / denomy;
		b[1] = -p / denomy;
	}
	else if (r < 0) {
		mode = 1;
		p = m;
		q = r;
		//-a + e^(p t)(a cos (q t) + b sin(q t))
		T denomx = -exp(p) * q + q * cos(q) + p * sin(q);
		a[0] = -(q * cos(q) + p * sin(q)) / denomx;
		b[0] = (p * cos(q) - q * sin(q)) / denomx;
		T denomy = q - exp(p) * q * cos(q) + exp(p) * p * sin(q);
		a[1] = -q / denomy;
		b[1] = p / denomy;
	}
	return {a, b, p, q, mode};
}

template <typename T>
void verify_correctness(const Matrix<T, 2, 1>& a, const Matrix<T, 2, 1>& b, const T& p, const T& q, int mode) {
	//position at 0
	{
		auto distance = Matrix<T, 2, 1>{0, 0} - apply_curve(a, b, p, q, 0, mode);
		check_warn(abs(distance[0]) < 0.0001, "x point at 0 failed", a, b, p, q, mode);
		check_warn(abs(distance[1]) < 0.0001, "y point at 0 failed", a, b, p, q, mode);
	}
	//position at 1
	{
		auto distance = Matrix<T, 2, 1>{1, 1} - apply_curve(a, b, p, q, 1, mode);
		check_warn(abs(distance[0]) < 0.0001, "x point at 1 failed", a, b, p, q, mode);
		check_warn(abs(distance[1]) < 0.0001, "y point at 1 failed", a, b, p, q, mode);
	}

	//direction is 0
	{
		auto [dt, dt2] = differentials<double>(a, b, p, q, 1., mode);
		check_warn(abs(dt[0]) < 0.0001, "x-derivative at 1 not zero", a, b, p, q, mode);
	}
	{
		auto [dt, dt2] = differentials<double>(a, b, p, q, 0., mode);
		check_warn(abs(dt[1]) < 0.0001, "y-derivative at 0 not zero", a, b, p, q, mode);
	}
}

struct MyFunctor {
	typedef double Scalar;
	enum {
		NUM_RESIDUALS = 2,
		NUM_PARAMETERS = 2,
	};

	double c1;
	double c2;
	template <typename T>
	bool operator()(const T* const parameters, T* residuals) const {
		const T& m = parameters[0];
		const T& r = parameters[1];
		auto [a, b, p, q, mode] = decompress(m, r);
		T curvature0 = calculate_curvatures(a, b, p, q, 0, mode);
		T curvature1 = calculate_curvatures(a, b, p, q, 1, mode);

		residuals[0] = curvature0 - c1;
		residuals[1] = curvature1 - c2;
		return true;
	}
};

using AutoDiffFunction = TinySolverAutoDiffFunction<MyFunctor, 2, 2>;
