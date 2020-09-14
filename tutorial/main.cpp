#include <iostream>
#include <vector>
#include <string>
#include <random>

#include <Eigen/Dense>

#include "Circle.h"

using namespace std;
using namespace Eigen;

double sum(double& x, double& y)
{
    return x + y;
}

int sum(const int& x, const int& y)
{
    return x + y;
}

int sum(int* x, int* y)
{
    return *x + *y;
}

int main() {
    cout << "Hello, World!" << endl;
    double x, y;
    x = 10;
    y = 2.5;
    cout << sum(x, y) << endl ;

    int b = 5;
    int* a = &b;
    cout << a << endl;

    VectorXd ones = VectorXd::Ones(5);
    VectorXd zeros = VectorXd::Zero(2);

    cout << "Zero Col vector" << endl;
    cout << zeros << endl << endl;

    MatrixXd mat(3, 3);
    MatrixXd id_matrix = MatrixXd::Identity(10, 10);
    cout << id_matrix << endl << endl;

    vector<int> feat(10);

    for (int i = 0; i < feat.size(); ++i)
        feat[i] = i;

    for (int i = 0; i < feat.size(); ++i) {
        cout << feat[i] << " ";
    }
    cout << endl;

    string hello = "Hello World!";
    cout << hello << endl;

    vector<vector<int>> data(10);

    int e = 3;
    int* to_e = &e;

    cout << to_e << " " << &to_e << endl;

    int* f = &e;
    int* g = &e;

    cout << sum(f, g) << endl;

    Circle circle(3);
    cout << "The area of circle is " << circle.area() << endl;

    uniform_int_distribution<int> uni_int(0, 10);
    mt19937_64 generator;

    for (int i = 0; i < 10; ++i) {
        cout << uni_int(generator) << " ";
    }
    cout << endl;

    return 0;
}
