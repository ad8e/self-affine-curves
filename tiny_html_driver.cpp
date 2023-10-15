#include "tiny_curve2.h"
#include <emscripten.h>
#include "console.h"

//cd /home/a/stuff/2dcurve/curvesolver/build/
//emcmake cmake -DCMAKE_BUILD_TYPE=Release -DCeres_DIR="/usr/lib/x86_64-linux-gnu/cmake/Ceres/" ../
//emcmake cmake -DCMAKE_BUILD_TYPE=Release -DCeres_DIR="/home/a/Downloads/ceres-solver/cmake" ../

/*
build steps:
1. download eigen and ceres. put them in downloads folder
source /home/a/Downloads/emsdk/emsdk_env.sh
go to eigen. create the build/ folder.
	emcmake cmake ../
	emmake make
	emmake make install
go to ceres. create the build/ folder.
	emcmake cmake ../ -DMINIGLOG=on
	cd ..
	emmake make
	seems unnecessary: emmake make install
*/

double solution[2] = {2, 1};
//int solution_index = 0;
//double extract_doubles_one_by_one() { //very sad way to pass doubles to javascript
//	return solution[(solution_index++ % 4)];
//}

double y_too;
extern "C" {

//solves the problem when in standard form: points at (0, 0), (1, 0), (1, 1)
EMSCRIPTEN_KEEPALIVE void solve_problem(double curvature1, double curvature2) {
 	MyFunctor my_functor{curvature1, curvature2}; //some state is being kept, but I don't know which. let's just reset it.
	AutoDiffFunction f(my_functor);

	//Eigen::Vector2d x(solution);
	Eigen::Vector2d x(0, -pi/2);
	if (curvature1 < 0.5 && curvature2 > 0.5 ||curvature1 > 0.5 && curvature2 < 0.5)
		//x = {pi / 2, pi / 2};
		x = {0, -5 * pi / 2};
	TinySolver<AutoDiffFunction> solver;
	solver.options.max_num_iterations *= 10; //50 is too small. gets stuck. corresponding to summary.status == 3
	solver.Solve(f, &x);
	solution[0] = x[0];
	solution[1] = x[1];

	//double r = solution[1] * solution[1] + solution[2] * solution[3];

	//outc(solution[0], solution[1], "final cost", solver.summary.final_cost);

	//outc("results:", x[0], x[1]);

	auto [a, b, p, q, mode] = decompress(solution[0], solution[1]);
	verify_correctness(a, b, p, q, mode);
	outc("coef", a, b, "exp", p, q, mode);
	outc("curvatures", curvature1, curvature2, "cost:", solver.summary.final_cost, "iterations", solver.summary.iterations, "status code", int(solver.summary.status)); //solver.summary.gradient_max_norm
	//	{
	//		auto [a, b, o, k] = calc_derive_coefficients_from_curvature(x[0], x[1], c1, c2);
	//		//outc("old curvatures", c1, c2, "new curvatures", curvature_from_coefficients<double>(x[0], x[1], a, b, o, k));
	//		//curvature_from_diffeq_solution(x[0], x[1], c1, c2);
	//
	//	}
	//outc("closeness to endpoint:", calc_derive_from_curvature<double>(x[0],x[1], c1,c2, 1));
}


EMSCRIPTEN_KEEPALIVE void set_m_and_r_directly(double m, double r) {
	solution[0] = m;
	solution[1] = r;
	//solution[1] = r < 0 ? -sqrt(-r) : sqrt(r); //r itself slows down to 0 near 0. this fixes that...except it's only the part near 0 which needs this fix
	solution[1] = r + 2 * (r < 0 ? -sqrt(-r) : sqrt(r)); //works better
	//solution[1] = m + r;

	auto [a, b, p, q, mode] = decompress(m, solution[1]);
	verify_correctness(a, b, p, q, mode);
	double curvature0 = calculate_curvatures(a, b, p, q, mode, 0);
	double curvature1 = calculate_curvatures(a, b, p, q, mode, 1);
	outc("curvatures", curvature0, curvature1);
	outc("coef", a, b, "exp", p, q, mode);
}

EMSCRIPTEN_KEEPALIVE double get_x(double t) {
	auto [a, b, p, q, mode] = decompress(solution[0], solution[1]);
	Matrix<double, 2, 1> point = apply_curve(a, b, p, q, mode, t);
	y_too = point[1];
	return point[0];
}
EMSCRIPTEN_KEEPALIVE double get_also_y() {
	return y_too;
}
}

int main() {
	EM_ASM({g9(final2_data, final2).insertInto('#final2-playground')});
	EM_ASM({g9(final1_data, final1).insertInto('#final1-playground')});
}
//to use: call solve_problem() to set parameters. to draw the curve, call get_x(t), then get_also_y()