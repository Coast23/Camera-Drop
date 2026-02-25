// Author: Coast23
// Date: 2026-02-25
/*
生成最小 Hamming Distance 尽量大的图案集。
生成的图案集需经过实验验证方能用于实际使用。
*/

#include <cstdio>
#include <vector>
#include <string>
#include <cstdint>
#include <iostream>

const int INF = 1145141919;
const int NUM = 32;  // 图案数

int main(){
    // 把 4x4 的 mask 拓展为 8x8 的 mask
    auto expand = [&](uint16_t mask) -> uint64_t {
        uint64_t res = 0;
        for(int r = 0; r < 4; ++r){
            for(int c = 0; c < 4; ++c){
                if((mask >> (r * 4 + c)) & 1){
                    uint32_t r8 = r << 1;
                    uint32_t c8 = c << 1;
                    res |= (1ULL << (r8 * 8 + c8));            // 左上
                    res |= (1ULL << (r8 * 8 + c8 + 1));        // 右上
                    res |= (1ULL << ((r8 + 1) * 8 + c8));      // 左下
                    res |= (1ULL << ((r8 + 1) * 8 + c8 + 1));  // 右下
                }
            }
        }
        return res;
    };

    auto popcnt = [&](uint64_t x) -> int {
        return __builtin_popcountll(x);
    };

    // 获取图案集的最小 Hamming Distance
    auto get_dist = [&](std::vector<uint16_t>& pick) -> int {
        int res = INF;
        int n = pick.size();
        for(int i = 0; i < n; ++i){
            for(int j = i + 1; j < n; ++j){
                res = std::min(res, popcnt(pick[i] ^ pick[j]));
            }
        }
        return res;
    };

    // 打印 patterns, col 为每行的图案数
    auto show_patterns = [&](std::vector<uint64_t>& Dict, int col = 4) -> void {
        int n = Dict.size();
        int row = (n + col - 1) / col;
        int R = 9 * row + 1;
        int C = 17 * col + 1;
        std::vector<std::string> canva(R, std::string(C, ' '));
        for(int i = 0; i < n; ++i){
            int rx = i / col;
            int cx = i % col;
            int x = rx * 9, y = cx * 17; // (x, y) 是左上角 * 的画布坐标
            // 绘制边框
            canva[x][y] = '*';
            canva[x][y + 17] = '*';
            canva[x + 9][y] = '*';
            canva[x + 9][y + 17] = '*';
            for(int c = y + 1; c <= y + 16; ++c) canva[x][c] = '-', canva[x + 9][c] = '-';
            for(int r = x + 1; r <= x + 8; ++r) canva[r][y] = '|', canva[r][y + 17] = '|';
            // 绘制 mask 对应的图案
            for(int dr = 0; dr < 8; ++dr){
                for(int dc = 0; dc < 8; ++dc){
                    if((Dict[i] >> (dr * 8 + dc)) & 1){
                        canva[x + dr + 1][y + (dc << 1) + 1] = '#';
                        canva[x + dr + 1][y + (dc << 1) + 2] = '#';
                    }
                }
            }
        }
        for(auto& line : canva){
            for(auto& c : line){
                if(c == '#') std::cout << "█"; // Fuck UTF-8 Characters
                else putchar(c);
            }
            putchar('\n');
        }
    };

    // 4x4 图案中，只选白色个数在 6~10 之间的图案
    std::vector<uint16_t> cand;
    for(int i = 0; i < (1 << 16); ++i){
        int p = popcnt(i);
        if(p >= 6 and p <= 10) cand.push_back(i);
    }
  //  printf("Candidate size: %u\n", cand.size());

    // 钦定第一个为黑白上下对半的图案。
    std::vector<uint16_t> pick; pick.push_back(0x00FF);
    std::vector<int> dist(cand.size(), INF);

    for(int i = 0; i < cand.size(); ++i) dist[i] = popcnt(cand[i] ^ pick[0]);

    for(int k = 1; k < NUM; ++k){
        int best = -1, maxDist = -1;
        for(int i = 0; i < cand.size(); ++i){
            if(dist[i] > maxDist){
                maxDist = dist[i];
                best = i;
            }
        }
        uint16_t picked = cand[best];
        pick.push_back(picked);
        for(int i = 0; i < cand.size(); ++i) dist[i] = std::min(dist[i], popcnt(cand[i] ^ picked));
    }
    
    int HammingDist = get_dist(pick) << 2; // 8x8 的 Hamming Distance 是 4x4 的 4 倍
    std::vector<uint64_t> Dict;
    for(auto& mask4x4 : pick) Dict.push_back(expand(mask4x4));

    puts("======================================");
    printf("Pattern Count: %d\n", NUM);
    printf("Min Hamming Distance: %d\n", HammingDist);
    printf("Error Tolerance: %d flips\n", (HammingDist - 1) >> 1);
    puts("======================================");

    printf("const uint64_t Dict[%u] = {\n", NUM);
    for(auto& mask8x8 : Dict){
        printf("    ");
        printf("0x%016llXULL,\n", mask8x8);
    }
    printf("};\n");
    show_patterns(Dict, 8);
}

/*
======================================
Pattern Count: 64
Min Hamming Distance: 20
Error Tolerance: 9 flips
======================================
*/