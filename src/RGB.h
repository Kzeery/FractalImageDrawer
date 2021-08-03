#pragma once
namespace FractalImages {
    struct RGB
    {
        double r;
        double g;
        double b;

        RGB(double r, double g, double b);
        RGB operator-(const RGB& other);
    };

}

