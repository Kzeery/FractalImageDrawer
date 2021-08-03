#include "ZoomList.h"
#include "Constants.h"
namespace FractalImages {

    void ZoomList::add(const Zoom& zoom)
    {
        Zooms_.push_back(zoom);
        XCenter_ += (zoom.X_ - WIDTH / 2) * Scale_;
        YCenter_ += (zoom.Y_ - HEIGHT / 2) * Scale_;
        Scale_ *= zoom.Scale_;
    }
    bool ZoomList::pop()
    {
        if (Zooms_.size() <= 2)
            return false;
        Zoom lastZoom = Zooms_.back();
        Scale_ /= lastZoom.Scale_;
        YCenter_ -= (lastZoom.Y_ - HEIGHT / 2) * Scale_;
        XCenter_ -= (lastZoom.X_ - WIDTH / 2) * Scale_;
        Zooms_.pop_back();
        return true;
    }

    std::pair<double, double> ZoomList::doZoom(long long x, long long y)
    {
        double xFractal = (x - WIDTH / 2) * Scale_ + XCenter_;
        double yFractal = (y - HEIGHT / 2) * Scale_ + YCenter_;
        return std::pair<double, double>(xFractal, yFractal);
    }


    double ZoomList::getXCenter() const 
    {
        return XCenter_;
    }
    double ZoomList::getYCenter() const
    {
        return YCenter_;
    }
    double ZoomList::getScale() const
    {
        return Scale_;
    }
}