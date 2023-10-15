#include "console.h"
#include "tiny_curve2.h"
#include <ceres/tiny_solver.h>
#include <ceres/tiny_solver_autodiff_function.h>
/*
cmake -DCMAKE_TOOLCHAIN_FILE=/home/a/Downloads/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake -DCMAKE_BUILD_TYPE=Debug -G "Unix Makefiles" .

*/

int main(int argc, char* argv[]) {
	double curvature0 = 0.7;
	double curvature1 = 1.2;
	if (argc == 3) {
		curvature0 = atof(argv[1]);
		curvature1 = atof(argv[2]);
	}

	MyFunctor my_functor{curvature0, curvature1};
	AutoDiffFunction f(my_functor);

	//Eigen::Vector2d x(solution);
	Eigen::Vector2d x(0, -pi / 2);
	if (curvature0 < 0.5 && curvature1 > 0.5 || curvature0 > 0.5 && curvature1 < 0.5) {
		//x = {pi / 2, pi / 2};
		x = {0, -5 * pi / 2};
	}
	TinySolver<AutoDiffFunction> solver;
	solver.options.max_num_iterations *= 10; //50 is too small. gets stuck. corresponding to summary.status == 3
	solver.Solve(f, &x);
	double solution[2];
	solution[0] = x[0];
	solution[1] = x[1];

	auto [a, b, p, q, mode] = decompress(solution[0], solution[1]);
	verify_correctness(a, b, p, q, mode);
	outc("coef", a, b, "exp", p, q, mode);
	outc("curvatures", curvature0, curvature1, "cost:", solver.summary.final_cost, "iterations", solver.summary.iterations, "status code", int(solver.summary.status)); //solver.summary.gradient_max_norm

	for (int i = 0; i <= 10; ++i) {
		double t = i / 10.0;
		auto result = apply_curve(a, b, p, q, mode, t);
		outc(result[0], result[1]);
	}

	//test if the circle produces the right number
	for (int i = 0; i <= 10; ++i) {
		double t = i / 10.0;
		auto [a, b, p, q, mode] = decompress(0., -pi / 2);
		auto result = apply_curve(a, b, p, q, mode, t);
		outc(result[0], result[1]);
	}

	//calc_derive_from_curvature<double>(0, 0, 1, -0.99, 1);
	return 0;
}