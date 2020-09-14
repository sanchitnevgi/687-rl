//
// Created by Sanchit Nevgi on 12/09/20.
//

#ifndef TUTORIAL_CIRCLE_H
#define TUTORIAL_CIRCLE_H

#include "Shape.h"

class Circle : public Shape {
public:
    Circle();
    Circle(double radius);

    double area() override;

private:
    double radius;
};


#endif //TUTORIAL_CIRCLE_H
