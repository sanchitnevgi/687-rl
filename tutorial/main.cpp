#include <iostream>
#include <vector>
#include <string>
#include <random>

#include <Eigen/Dense>

#include "Circle.h"

using namespace std;
using namespace Eigen;

void vector_tutorial()
{
    VectorXd vec = VectorXd::Random(100);
    double vec_mean = vec.mean();
    cout << "Vector Mean is " << vec_mean;
}

void matrix_tutorial_1() {
    MatrixXd runs = MatrixXd::Ones(10, 20);
    cout << runs.row(0) << endl;
}

int main() {
    // vector_tutorial();
    matrix_tutorial_1();
    return 0;
}
