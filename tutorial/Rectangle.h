//
// Created by Sanchit Nevgi on 12/09/20.
//

#ifndef TUTORIAL_RECTANGLE_H
#define TUTORIAL_RECTANGLE_H

class Rectangle {
public:
    Rectangle();
    Rectangle(double width, double height);

    double area();
protected:
private:
    double width;
    double height;
};

#endif //TUTORIAL_RECTANGLE_H
