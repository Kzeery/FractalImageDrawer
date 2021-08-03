#pragma once
#include "ZoomList.h"
#include "RGB.h"
#include "Screen.h"
#include <mutex>
#include <vector>
#include <cuda_runtime.h>

namespace FractalImages {
    class ZoomList;
    class FractalCreator
    {
        friend void CalculateIterationsCuda(FractalCreator* fractalCreator);
    public:
        static bool CudaChecked_;
        FractalCreator(Screen* screen);
        void run();
        void update();
        void addRange(double rangeEnd, const RGB& rgb);
        void addZoom(const Zoom& zoom);
        bool popZoom();
        virtual ~FractalCreator();
    private:
        
        void init();
        void calcIterations();
        void calculateIterations();
        void calculateIterationsMT(long long start, long long threadcount);
        void drawFractal();
        void drawFractalMT(long long start, long long threadcount);
        void calculateRangeTotals();
        int getRange(int iterations) const;
        

    private:
        bool GotFirstRange_{ false };
        bool CudaAvailable_;
        int SMCount_;
        int* Histogram_ = nullptr;
        int* Fractal_ = nullptr;
        ZoomList ZoomList_;
        Screen* Screen_;
        std::mutex MyMutex_;
        std::vector<int> Ranges_;
        std::vector<RGB> Colors_;
        std::vector<int> RangeTotals_;
    };
    __global__ void kernel(double scale, double xcenter, double ycenter, int SMCOUNT, int* Histogram_, int* Fractal_);

    


    


}

