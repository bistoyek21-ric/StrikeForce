# StrikeForce
## Created by 21

## Phase 2 (Developing the structures)
#### Project Manager: Kasra Fouladi
#### Author: Kasra Fouladi

##### Requirements: C++ (17 at least), SMFL framework, curl and 7z (these two are terminal gadgets)

**StrikeForce** is 21's open-source FPS-like Research testbed to develope FPS-bots, and also a Battle Royale game available on Windows, Unix-based operating systems, and Apple devices.

This project also has a scientific aspect. You can create AI bots and have them compete against each other to see who wins. Additionally, you can use this project to train single or multi agent or Collective Intelligence AI models. This open-source environment for ML and AI development can serve as the basis for many research studies on learning methods.

We organize events focusing on different parts of this game, especially the AI Battle Royale section. For more information about AI contests and to participate in the development of this project or other open-source projects directed by [@bistoyek21-ric](https://github.com/bistoyek21-ric), join our Telegram [@StrikeForce21](https://t.me/StrikeForce21).

---

## Installation

To install and run this project, you need a C++ compiler. Follow these steps:

1. Install dependencies using 'INSTALLDEP.md'.

2. Clone the repository (Or you can download one of the versions):
    ```sh
    git clone https://github.com/bistoyek21-ric/StrikeForce.git
    ```
3. Navigate to the project directory:
    ```sh
    cd StrikeForce/StrikeForce-client
    ```
4. Compile the project:
    ```sh
    g++ -std=c++17 main.cpp -o app -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lsfml-graphics -lsfml-window -lsfml-system && ./app
    ```
5. Run the executable:
    ```sh
    ./app
    ```

You should do almost the same thing (but not linking sfml and libtorch linkers) to run the server the only difference is between the name of folders and source files. (There is a builtin account with user=1, pass=1)

Attention:\
    1. If you using windows you have to enter the linker -lws2_32 otherwise you shouldn't.\
    2. If you using windows you souldn't enter "./" at the first.
    
---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Since this project is open-source, you can make your desired changes and create your own versions of it. For example, you can add a portal gun, the ability to build temporary barriers, change the game map, create new items or game modes, and many other modifications. However, please do not change the name of the project or the original creator. (You can create your version and add yourself to the list of creators in the head function and name your version like StrikeForce-kasra-version-1.2.4.)

Thank you! 😊

**All material and intellectual rights of this environment belong to author ([@kasrafouladi](https://github.com/kasrafouladi) and [@bistoyek21-ric](https://github.com/bistoyek21-ric). Any plagiarism will be prosecuted.**

---

#### In case of any issues like errors, bugs, or questions about the code, you can contact the author at:
- k04sr405@gmail.com

