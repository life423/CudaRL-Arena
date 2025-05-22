#include <SDL.h>
#include <iostream>

int main() {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        return 1;
    }
    SDL_Window* win = SDL_CreateWindow("SDL2 Test", 100, 100, 640, 480, SDL_WINDOW_SHOWN);
    if (win == nullptr) {
        std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }
    SDL_Delay(1000); // Show window for 1 second
    SDL_DestroyWindow(win);
    SDL_Quit();
    std::cout << "SDL2 is working correctly!" << std::endl;
    return 0;
}