//
// Created by Sanchit Nevgi on 12/09/20.
//

#include "Rectangle.h"

Rectangle::Rectangle() {
    width = height = 0;
}

Rectangle::Rectangle(double width, double height) {
    this->width = width;
    this->height = height;
}

double Rectangle::area() {
    return width * height;
}