//
// Created by Sanchit Nevgi on 12/09/20.
//
#include <cmath>
#include "Circle.h"

Circle::Circle() {
    radius = 0;
}

Circle::Circle(double radius) {
    this->radius = radius;
}

double Circle::area() {
    return 2.0 * M_PI * radius * radius;
}

