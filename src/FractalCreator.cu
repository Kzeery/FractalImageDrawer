#include "FractalCreator.cuh"
#include "Constants.h"
#include "ZoomList.cuh"
#include "Mandelbrot.cuh"
#include "device_launch_parameters.h"
#include <thread>

namespace FractalImages {
    bool FractalCreator::CudaChecked_ = false;
    FractalCreator::FractalCreator(Screen* screen) : ZoomList_(ZoomList()), Screen_(screen) {
        Histogram_ = new int[MAX_ITERATIONS] { 0 };
        Fractal_ = new int[WIDTH * HEIGHT]{ 0 };
        addZoom((Zoom(WIDTH / 2, HEIGHT / 2, 4.0 / WIDTH)));
        addZoom((Zoom(WIDTH / 4, HEIGHT / 2, 1)));
        int deviceCount = 0;
        if (!CudaChecked_)
        {
            CudaChecked_ = true;
            cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
            if (error_id == cudaSuccess)
            {
                CudaAvailable_ = true;
                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, 0);
                SMCount_ = deviceProp.multiProcessorCount * 0.9;
            }
        }
        
    };

    void FractalCreator::run()
    {
        calcIterations();
        calculateRangeTotals();
        drawFractal();
        Screen_->update();
    }

    void FractalCreator::update()
    {
        init();
        run();
    }

    void FractalCreator::addRange(double rangeEnd, const RGB& rgb)
    {
        Ranges_.push_back(rangeEnd * MAX_ITERATIONS);
        Colors_.push_back(rgb);

        if (GotFirstRange_)
        {
            RangeTotals_.push_back(0);
        }
        GotFirstRange_ = true;
    }

    void FractalCreator::addZoom(const Zoom& zoom)
    {
        ZoomList_.add(zoom);
    }

    bool FractalCreator::popZoom()
    {
        return ZoomList_.pop();
    }

    void FractalCreator::init()
    {
        std::fill(RangeTotals_.begin(), RangeTotals_.end(), 0);
        memset(Histogram_, 0, MAX_ITERATIONS * sizeof(int));
        memset(Fractal_, 0, WIDTH * HEIGHT * sizeof(int));
    }

    void FractalCreator::calcIterations()
    {
        if (CudaAvailable_)
        {
            CalculateIterationsCuda(this);
        }
        else
        {
            calculateIterations();
        }
    }

    void FractalCreator::calculateIterations()
    {
        const long long NUMOFTHREADS = std::thread::hardware_concurrency() - 6;
        if (NUMOFTHREADS < 2) return drawFractalMT(0, 1);
        std::vector<std::thread> threads(NUMOFTHREADS);
        auto fn = [&](long long start, long long threadcount)
        {
            calculateIterationsMT(start, threadcount);
        };
        for (int i = 0; i < NUMOFTHREADS; i++)
        {
            threads[i] = std::thread(fn, i, NUMOFTHREADS);
        }
        for (int i = 0; i < NUMOFTHREADS; i++)
        {
            threads[i].join();
        }
    }

    void FractalCreator::calculateIterationsMT(long long start, long long threadcount)
    {
        for (long long x = start; x < WIDTH; x += threadcount)
        {
            for (long long y = 0; y < HEIGHT; y++)
            {
                auto coords = ZoomList_.doZoom(x, y);
                int iterations = getIterations(coords.first, coords.second);
                Fractal_[y * WIDTH + x] = iterations;
                if (iterations != MAX_ITERATIONS)
                {
                    MyMutex_.lock();
                    Histogram_[iterations]++;
                    MyMutex_.unlock();
                }
            }
        }
    }

    void FractalCreator::drawFractal()
    {
        const long long NUMOFTHREADS = std::thread::hardware_concurrency() - 4;
        if (NUMOFTHREADS < 2) return drawFractalMT(0, 1);
        std::vector<std::thread> threads(NUMOFTHREADS);
        auto fn = [&](long long start, long long threadcount)
        {
            drawFractalMT(start, threadcount);
        };
        for (int i = 0; i < NUMOFTHREADS; i++)
        {
            threads[i] = std::thread(fn, i, NUMOFTHREADS);
        }
        for (int i = 0; i < NUMOFTHREADS; i++)
        {
            threads[i].join();
        }
    }
    void FractalCreator::drawFractalMT(long long start, long long threadcount)
    {
        for (long long x = start; x < WIDTH; x += threadcount)
        {
            for (long long y = 0; y < HEIGHT; y++)
            {
                int iterations = Fractal_[y * WIDTH + x];
                RGB color(0, 0, 0);

                if (iterations != MAX_ITERATIONS)
                {
                    int range = getRange(iterations);
                    int rangeTotal = RangeTotals_[range];
                    int rangeStart = Ranges_[range];
                    RGB& startColor = Colors_[range];
                    RGB& endColor = Colors_[range + 1];
                    RGB colorDiff = endColor - startColor;
                    int totalPixels = 0;
                    for (int i = rangeStart; i <= iterations; i++)
                    {
                        totalPixels += Histogram_[i];
                    }
                    color.r = startColor.r + colorDiff.r * (double)totalPixels / rangeTotal;
                    color.g = startColor.g + colorDiff.g * (double)totalPixels / rangeTotal;
                    color.b = startColor.b + colorDiff.b * (double)totalPixels / rangeTotal;
                }
                Screen_->setPixel(x, y, color.r, color.g, color.b);
            }
        }
    }
   
    void FractalCreator::calculateRangeTotals()
    {
        long long currentIndex = 0;
        for (int i = 0; i < MAX_ITERATIONS; i++)
        {
            if (i >= Ranges_[currentIndex + 1])
            {
                currentIndex++;
            }
            RangeTotals_[currentIndex] += Histogram_[i];
        }

    }

    int FractalCreator::getRange(int iterations) const
    {
        for (int i = 1; i < Ranges_.size(); i++)
        {
            if (Ranges_[i] > iterations)
                return i - 1;
        }
        return -1;
    }

    FractalCreator::~FractalCreator()
    {
        delete[] Histogram_;
        delete[] Fractal_;
    }

    void CalculateIterationsCuda(FractalCreator* fractalCreator)
    {
        int* d_Histogram;
        int* d_Fractal;
        cudaMalloc(&d_Histogram, sizeof(int) * MAX_ITERATIONS);
        cudaMalloc(&d_Fractal, sizeof(int) * WIDTH * HEIGHT);
        cudaMemcpy(d_Histogram, fractalCreator->Histogram_, sizeof(int) * MAX_ITERATIONS, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Fractal, fractalCreator->Fractal_, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
        
        double scale = fractalCreator->ZoomList_.getScale();
        double XCenter = fractalCreator->ZoomList_.getXCenter();
        double YCenter = fractalCreator->ZoomList_.getYCenter();
        int SMCOUNT = fractalCreator->SMCount_;
        kernel<<<SMCOUNT, THREADCOUNT>>>(scale, XCenter, YCenter, SMCOUNT, d_Histogram, d_Fractal);
        cudaDeviceSynchronize();

        cudaMemcpy(fractalCreator->Histogram_, d_Histogram, sizeof(int) * MAX_ITERATIONS, cudaMemcpyDeviceToHost);
        cudaMemcpy(fractalCreator->Fractal_, d_Fractal, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
        cudaFree(d_Histogram);
        cudaFree(d_Fractal);
    }
    
    __global__ void kernel(double scale, double xcenter, double ycenter, int SMCOUNT, int* Histogram_, int* Fractal_)
    {
        for (long long x = threadIdx.x; x < WIDTH; x += THREADCOUNT)
            for (long long y = blockIdx.x; y < HEIGHT; y += SMCOUNT)
            {
                double xFractal, yFractal = 0;
                doZoom(x, y, scale, xcenter, ycenter, xFractal, yFractal);

                int iterations = getIterations(xFractal, yFractal);
                Fractal_[y * WIDTH + x] = iterations;
                if (iterations != MAX_ITERATIONS)
                {
                    atomicAdd(Histogram_ + iterations, 1);
                }
            }
    }
}
