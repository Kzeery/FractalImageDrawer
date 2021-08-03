#include "Bitmap.h"
#include "BitmapInfoHeader.h"
#include "BitmapFileHeader.h"
#include "Constants.h"
#include <fstream>
using namespace FractalImages;
namespace FractalImages {

    Bitmap::Bitmap() : Pixels_(new uint8_t[WIDTH * HEIGHT * 3]{})
    {

    }

    Bitmap::~Bitmap()
    {
    }

    bool Bitmap::write(std::string filename)
    {
        BitmapFileHeader fileHeader;
        BitmapInfoHeader infoHeader;
        
        fileHeader.fileSize = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader) + (WIDTH * HEIGHT * 3);
        fileHeader.dataOffset = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader);

        infoHeader.width = WIDTH;
        infoHeader.height = HEIGHT;

        std::ofstream file;
        file.open(filename.c_str(), std::ios::out | std::ios::binary);
        if (!file) return false;
        file.write((char*)&fileHeader, sizeof(fileHeader));
        file.write((char*)&infoHeader, sizeof(infoHeader));
        file.write((char*)Pixels_.get(), WIDTH * HEIGHT * 3);

        file.close();
        if (!file) return false;
        return true;
    }

    void Bitmap::setPixel(int x, int y, const RGB& colour)
    {
        uint8_t* pixel = Pixels_.get();
        pixel += 3 * (y * WIDTH + x);
        pixel[0] = colour.b;
        pixel[1] = colour.g;
        pixel[2] = colour.r;
    }

}