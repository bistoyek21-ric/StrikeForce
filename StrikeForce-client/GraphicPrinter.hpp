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

// ---------------- Graphic Printer ----------------------
class GraphicPrinter {
private:
    struct TextBufferEntry {
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

    sf::Color ansiColor(int code, bool isForeground) {
        switch (code) {
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

    void renderLoop() {
        sf::RenderWindow w(sf::VideoMode(H, W), "StrikeForce");
        window = &w;
        window->setFramerateLimit(4000);

        while (window->isOpen() && running) {
            sf::Event event;
            while (window->pollEvent(event)) {
                if (event.type == sf::Event::Closed) {
                    window->close();
                }
            }

            std::unique_lock lock(mtx);
            cv.wait_for(lock, std::chrono::milliseconds(1));

            if (clearRequested) {
                buffer.clear();
                clearRequested = false;
            }

            while (!printQueue.empty()) {
                parsePrint(printQueue.front());
                printQueue.pop();
            }

            window->clear(sf::Color::Black);
            float yOffset = startY;
            for (const auto& entry : buffer) {
                float xOffset = startX;
                for (int i = 0; i < entry.text.size(); ++i) {
                    std::string s; s.push_back(entry.text[i]);
                    sf::Text text(s, font, charSize);
                    text.setFillColor(entry.textColor[i]);
                    text.setPosition(xOffset, yOffset);

                    sf::FloatRect bounds = text.getLocalBounds();
                    sf::RectangleShape bgRect(sf::Vector2f(bounds.width, bounds.height));
                    bgRect.setFillColor(entry.bgColor[i]);
                    bgRect.setPosition(xOffset + bounds.left, yOffset + bounds.top);

                    xOffset += charSize * lineSpacing / 2;

                    window->draw(bgRect);
                    window->draw(text);
                }
                yOffset += charSize * lineSpacing;
            }
            window->display();
        }
    }

    void parsePrint(const std::string& str) {
        std::string line;
        std::vector<sf::Color> textColor, BgColor;
        size_t i = 0;
        while (i < str.size()) {
            if (i + 9 < str.size() && str[i] == '\033' && str[i + 1] == '[' && str[i + 4] == 'm'
                && str[i + 5] == '\033' && str[i + 6] == '[' && str[i + 9] == 'm') {
                int code1 = 10 * (str[i + 2] - '0') + str[i + 3] - '0';
                int code2 = 10 * (str[i + 7] - '0') + str[i + 8] - '0';
                currentTextColor = ansiColor(code1, true);
                currentBgColor = ansiColor(code2, false);
                i += 10;
            } else if (str[i] == '\n') {
                buffer.push_back({line, textColor, BgColor});
                line.clear(), textColor.clear(), BgColor.clear();
                ++i;
            } else {
                line += str[i++];
                textColor.push_back(currentTextColor);
                BgColor.push_back(currentBgColor);
            }
        }
        if (!line.empty()) {
            buffer.push_back({line, textColor, BgColor});
            line.clear(), textColor.clear(), BgColor.clear();
        }
    }

public:
    GraphicPrinter(unsigned int characterSize = 14, unsigned int H = 1200, unsigned int W = 1200) : charSize(characterSize), H(H), W(W) {
        if (!font.loadFromFile("./DejaVuSansMono.ttf")) {
            throw std::runtime_error("Failed to load font");
        }
    }

    void start() {
        running = true;
        renderThread = std::thread(&GraphicPrinter::renderLoop, this);
    }

    void stop() {
        running = false;
        cv.notify_all();
        if (renderThread.joinable())
            renderThread.join();
        window->close();
    }

    void print(const std::string& str) {
        std::lock_guard lock(mtx);
        printQueue.push(str);
        cv.notify_all();
    }

    void cls() {
        std::lock_guard lock(mtx);
        clearRequested = true;
        cv.notify_all();
    }
} printer;