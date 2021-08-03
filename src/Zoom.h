#pragma once
namespace FractalImages{
    struct Zoom {
        long long X_{ 0 };
        long long Y_{ 0 };
        double Scale_{ 0.0 };
        Zoom(long long x, long long y, double scale) : X_(x), Y_(y), Scale_(scale) {};
};

}