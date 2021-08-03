#include "RGB.h"
namespace FractalImages {
    RGB::RGB(double r, double g, double b) : r(r), g(g), b(b) {};
    

    RGB RGB::operator-(const RGB& other)
    {
        return RGB(r - other.r, g - other.g, b - other.b);
    }
}
