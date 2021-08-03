#define SDL_MAIN_HANDLED
#include "FractalCreator.cuh"
#include "RGB.h"

using namespace FractalImages;

int main(int argc, char** argv)
{
    Screen screen;
    if (screen.init() == false) {
        return 1;
    }
    SDL_SetMainReady();
    
    std::unique_ptr<FractalCreator> fractalCreator(new FractalCreator(&screen));
    fractalCreator->addRange(0.0, RGB(0, 0, 0));
    fractalCreator->addRange(0.001, RGB(0, 50, 255));
    fractalCreator->addRange(0.04, RGB(255, 0, 255));
    fractalCreator->addRange(1.0, RGB(255, 255, 255));
    fractalCreator->run();
    
    while (true)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type)
            {
            case SDL_QUIT:
                screen.close();
                return 0;
                break;
            case SDL_MOUSEBUTTONUP:
                switch (event.button.button)
                {
                case SDL_BUTTON_LEFT:
                    fractalCreator->addZoom(Zoom(event.button.x, event.button.y, 0.5));
                    fractalCreator->update();
                    break;
                case SDL_BUTTON_RIGHT:
                    if (fractalCreator->popZoom())
                    {
                        fractalCreator->update();
                    }
                    break;
                }
                break;
            }
        }
    }
    screen.close();
    printf("Finished!\n");
    return 0;
}