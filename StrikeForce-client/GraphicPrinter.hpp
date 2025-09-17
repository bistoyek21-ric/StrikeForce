/*
MIT License

Copyright (c) 2025 bistoyek21 R.I.C.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <queue>
#include <optional>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

sf::ConvexShape createTriangle(int direction, sf::FloatRect bounds, sf::Color fillColor, float x, float y){
    sf::ConvexShape triangle(3);
    triangle.setFillColor(fillColor);
    float w = bounds.width, h = bounds.height;
    switch(direction){
        case 1:
            triangle.setPoint(0, sf::Vector2f(0, 0));
            triangle.setPoint(1, sf::Vector2f(w, 0));
            triangle.setPoint(2, sf::Vector2f(w/2, h));
            break;
        case 2:
            triangle.setPoint(0, sf::Vector2f(0, 0));
            triangle.setPoint(1, sf::Vector2f(0, h));
            triangle.setPoint(2, sf::Vector2f(w, h/2));
            break;
        case 3:
            triangle.setPoint(0, sf::Vector2f(0, h));
            triangle.setPoint(1, sf::Vector2f(w, h));
            triangle.setPoint(2, sf::Vector2f(w/2, 0));
            break;
        case 4:
            triangle.setPoint(0, sf::Vector2f(w, 0));
            triangle.setPoint(1, sf::Vector2f(w, h));
            triangle.setPoint(2, sf::Vector2f(0, h/2));
            break;
        default:
            break;
    }
    triangle.setPosition(x + bounds.left, y + bounds.top);
    return triangle;
}

class GraphicPrinter{
private:
    struct TextBufferEntry{
        std::string text;
        std::vector<sf::Color> textColor;
        std::vector<sf::Color> bgColor;
    };

    unsigned int charSize, H, W;
    sf::Font font;
    std::vector<TextBufferEntry> buffer;
    std::queue<std::string> printQueue;

    sf::Color currentTextColor = sf::Color::White;
    sf::Color currentBgColor = sf::Color::Black;

    std::mutex mtx;
    std::condition_variable cv;
    bool running = false;
    std::thread renderThread;
    bool clearRequested = false;

    float lineSpacing = 1.2f;
    float startX = 5.f;
    float startY = 5.f;

    sf::RenderWindow* window;

    sf::Color ansiColor(int code, bool isForeground){
        switch(code){
            case 30: return sf::Color(0, 0, 0);
            case 31: return sf::Color::Red;
            case 32: return sf::Color::Green;
            case 33: return sf::Color::Yellow;
            case 34: return sf::Color::Blue;
            case 35: return sf::Color::Magenta;
            case 36: return sf::Color::Cyan;
            case 37: return sf::Color::White;
            case 40: return sf::Color(0, 0, 0);
            case 41: return sf::Color(255, 0, 0);
            case 42: return sf::Color(0, 255, 0);
            case 43: return sf::Color(255, 255, 0);
            case 44: return sf::Color(0, 0, 255);
            case 45: return sf::Color(255, 0, 255);
            case 46: return sf::Color(0, 255, 255);
            case 47: return sf::Color(255, 255, 255);
            default: return isForeground ? sf::Color::White : sf::Color::Black;
        }
    }

    void renderLoop(){
        sf::RenderWindow w(sf::VideoMode(W, H), "StrikeForce");
        window = &w;
        window->setFramerateLimit(100);
        while(window->isOpen() && running){
            sf::Event event;
            while(window->pollEvent(event))
                if(event.type == sf::Event::Closed)
                    window->close();
            std::unique_lock lock(mtx);
            cv.wait_for(lock, std::chrono::milliseconds(25));
            if(clearRequested){
                buffer.clear();
                clearRequested = false;
            }
            while(!printQueue.empty()){
                parsePrint(printQueue.front());
                printQueue.pop();
            }
            window->clear(sf::Color::Black);
            float yOffset = startY;
            for(const auto& entry: buffer){
                float xOffset = startX;
                for(int i = 0; i < entry.text.size(); ++i){
                    char c = entry.text[i];
                    std::string s; s.push_back(c);
                    sf::Text text(s, font, charSize);
                    text.setFillColor(entry.textColor[i]);
                    text.setPosition(xOffset, yOffset);
                    sf::FloatRect bounds = text.getLocalBounds();
                    sf::RectangleShape bgRect(sf::Vector2f(bounds.width, bounds.height));
                    bgRect.setFillColor(entry.bgColor[i]);
                    bgRect.setPosition(xOffset + bounds.left, yOffset + bounds.top);
                    if(1 <= s[0] && s[0] <= 4){
                        sf::ConvexShape triangle = createTriangle(s[0], bounds, entry.textColor[i], xOffset, yOffset);
                        window->draw(bgRect);
                        window->draw(triangle);
                    }
                    else{
                        window->draw(bgRect);
                        window->draw(text);
                    }
                    xOffset += charSize * lineSpacing / 2;
                }
                yOffset += charSize * lineSpacing;
            }
            window->display();
        }
    }

    void parsePrint(const std::string& str){
        std::string line;
        std::vector<sf::Color> textColor, BgColor;
        size_t i = 0;
        while(i < str.size()){
            if(i + 9 < str.size() && str[i] == '\033' && str[i + 1] == '[' && str[i + 4] == 'm'
                && str[i + 5] == '\033' && str[i + 6] == '[' && str[i + 9] == 'm'){
                int code1 = 10 * (str[i + 2] - '0') + str[i + 3] - '0';
                int code2 = 10 * (str[i + 7] - '0') + str[i + 8] - '0';
                currentTextColor = ansiColor(code1, true);
                currentBgColor = ansiColor(code2, false);
                i += 10;
            }
            else if (str[i] == '\n'){
                buffer.push_back({line, textColor, BgColor});
                line.clear(), textColor.clear(), BgColor.clear();
                ++i;
            }
            else{
                line += str[i++];
                textColor.push_back(currentTextColor);
                BgColor.push_back(currentBgColor);
            }
        }
        if(!line.empty()){
            buffer.push_back({line, textColor, BgColor});
            line.clear(), textColor.clear(), BgColor.clear();
        }
    }

public:
    GraphicPrinter(unsigned int characterSize = 14, unsigned int H = 800, unsigned int W = 800) : charSize(characterSize), H(H), W(W){
        if(!font.loadFromFile("./DejaVuSansMono.ttf"))
            throw std::runtime_error("Failed to load font");
    }

    void start(){
        running = true;
        renderThread = std::thread(&GraphicPrinter::renderLoop, this);
    }

    void stop(){
        running = false;
        cv.notify_all();
        if(renderThread.joinable())
            renderThread.join();
        window->close();
    }

    void print(const std::string& str){
        std::lock_guard lock(mtx);
        printQueue.push(str);
        cv.notify_all();
    }

    void cls(){
        std::lock_guard lock(mtx);
        clearRequested = true;
        cv.notify_all();
    }

    void render(const std::string& str){
        std::lock_guard lock(mtx);
        clearRequested = true;
        while(!printQueue.empty())
            printQueue.pop();
        printQueue.push(str);
        cv.notify_all();
    }
} printer;