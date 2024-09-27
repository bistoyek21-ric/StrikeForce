/*
MIT License

Copyright (c) 2024 bistoyek(21)

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
#include "basic.hpp"

namespace Enviorment::Random{

	long long jomle, mod = (1 << 16) + 1;

	long long random[18], seed[18], us[18], p[(1 << 16) + 1][11];

	void make_p(){
		for(int i = 0; i < mod; ++i){
			p[i][0] = 1;
			for(int j = 1; j < 11; ++j)
				p[i][j] = (i * p[i][j - 1]) % mod;
		}
		return;
	}

	long long binpow(long long a, long long b){
		long long res = 1;
		b %= mod - 1;
		while(b){
			if(b & 1)
				res = (res * a) % mod;
			a = (a * a) % mod;
			b >>= 1;
		}
		return res;
	}

	int _rand(){
		long long sum = 1;
		for(int i = 0; i < 18; ++i)
			sum = (sum + us[i] * p[random[i]][seed[i]]) % mod;
		random[0] = binpow(sum + (int)(sum == 0), ++jomle);
		for(int i = 0; i < 17; ++i)
			std::swap(random[i], random[i + 1]);
		return random[17] & 1023;
	}

	void _srand(long long tb, long long u_s){
		for(int i = 0; i < 18; ++i){
			us[i] = u_s % 10 + 1;
			seed[i] = tb % 10 + 1;
			u_s /= 10;
			tb /= 10;
			random[i] = 0;
		}
		jomle = 18;
		for(int i = 0; i < 1024; ++i)
			_rand();
		return;
	}
}
