#pragma once
#include "Zoom.h"
#include <vector>
#include <utility>
namespace FractalImages {
    class ZoomList
    {
    private:
        std::vector<Zoom> Zooms_;
        double XCenter_{ 0 };
        double YCenter_{ 0 };
        double Scale_{ 1 };
    public:
        
        void add(const Zoom& zoom);
        bool pop();
        std::pair<double, double> doZoom(long long x, long long y);
        double getXCenter() const;
        double getYCenter() const;
        double getScale() const;
    };
}


