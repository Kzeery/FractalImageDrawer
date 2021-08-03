#pragma once
#include <string>
#include <cstdint>
#include <memory>
#include "RGB.h"
namespace FractalImages {

    class Bitmap
    {
    private:
        std::unique_ptr<uint8_t[]> Pixels_{ nullptr };
    public:
        Bitmap();
        void setPixel(int x, int y, const RGB &rgb);
        bool write(std::string filename);
        virtual ~Bitmap();
    };

}

