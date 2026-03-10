#pragma once

#include "tile.hpp"
#include "constants.hpp"

using namespace cv;

class Painter {
public:
    Painter(Mat& img_, const bool draw_anchor = true) : img_(img_) {
        if(draw_anchor) draw_anchors();
    }

    void draw_normal_anchor(int x0, int y0){
        rectangle(img_, Rect(x0, y0, ANCHOR_L1_SIZE, ANCHOR_L1_SIZE),
                        Scalar(255, 255, 255), FILLED);
        rectangle(img_, Rect(x0 + ANCHOR_L2_INSET, y0 + ANCHOR_L2_INSET,
                        ANCHOR_L2_SIZE, ANCHOR_L2_SIZE), Scalar(0, 0, 0), FILLED);
        rectangle(img_, Rect(x0 + ANCHOR_L3_INSET, y0 + ANCHOR_L3_INSET,
                        ANCHOR_L3_SIZE, ANCHOR_L3_SIZE), Scalar(255, 255, 255), FILLED);
        rectangle(img_, Rect(x0 + ANCHOR_L4_INSET, y0 + ANCHOR_L4_INSET,
                        ANCHOR_L4_SIZE, ANCHOR_L4_SIZE), Scalar(0, 0, 0), FILLED);
    }

    void draw_br_anchor(int x0, int y0){
        int h1 = ANCHOR_L1_SIZE >> 1;
        rectangle(img_, Rect(x0,      y0,      h1, h1), Vec3b(0, 255, 255),  FILLED); // Yellow
        rectangle(img_, Rect(x0+h1,   y0,      h1, h1), Vec3b(0, 255, 0),    FILLED); // Green
        rectangle(img_, Rect(x0,      y0+h1,   h1, h1), Vec3b(255, 0, 255),  FILLED); // Magenta
        rectangle(img_, Rect(x0+h1,   y0+h1,   h1, h1), Vec3b(255, 255, 0), FILLED); // Cyan
        // L2: 42×42 黑色
        rectangle(img_, Rect(x0 + ANCHOR_L2_INSET, y0 + ANCHOR_L2_INSET,
                            ANCHOR_L2_SIZE, ANCHOR_L2_SIZE), Scalar(0, 0, 0), FILLED);
        // L3: 28×28 四象限彩色
        int h3 = ANCHOR_L3_SIZE >> 1;
        int ix = x0 + ANCHOR_L3_INSET, iy = y0 + ANCHOR_L3_INSET;
        rectangle(img_, Rect(ix,    iy,    h3, h3), Vec3b(0, 255, 255),  FILLED); // Yellow
        rectangle(img_, Rect(ix+h3, iy,    h3, h3), Vec3b(0, 255, 0),    FILLED); // Green
        rectangle(img_, Rect(ix,    iy+h3, h3, h3), Vec3b(255, 0, 255),  FILLED); // Magenta
        rectangle(img_, Rect(ix+h3, iy+h3, h3, h3), Vec3b(255, 255, 0), FILLED); // Cyan
        // L4: 14×14 黑色
        rectangle(img_, Rect(x0 + ANCHOR_L4_INSET, y0 + ANCHOR_L4_INSET,
                            ANCHOR_L4_SIZE, ANCHOR_L4_SIZE), Scalar(0, 0, 0), FILLED);
    }

    void draw_anchors(){
        constexpr int tl_x = ANCHOR_OUT_START;
        constexpr int tl_y = ANCHOR_OUT_START;
        constexpr int tr_x = IMG_WIDTH - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
        constexpr int tr_y = ANCHOR_OUT_START;
        constexpr int bl_x = ANCHOR_OUT_START;
        constexpr int bl_y = IMG_HEIGHT - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
        constexpr int br_x = IMG_WIDTH - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
        constexpr int br_y = IMG_HEIGHT - ANCHOR_OUT_START - ANCHOR_L1_SIZE;

        draw_normal_anchor(tl_x, tl_y);  // TL: 白黑白黑
        draw_normal_anchor(tr_x, tr_y);  // TR: 白黑白黑
        draw_normal_anchor(bl_x, bl_y);  // BL: 白黑白黑
        draw_br_anchor    (br_x, br_y);  // BR: 彩黑彩黑（方向标记）
    }

    bool draw_tile(const Tile& tile, const int grid_r, const int grid_c){
        if(is_reserved(grid_r, grid_c)) return false;

        int startX = MARGIN + grid_c * STRIDE;
        int startY = MARGIN + grid_r * STRIDE;

        const auto mask = tile.get_pattern();
        const auto color = tile.get_color();

        for(uint8_t pr = 0; pr < TILE_SIZE; ++pr)
            for(uint8_t pc = 0; pc < TILE_SIZE; ++pc)
                if((mask >> (pr * 8 + pc)) & 1)
                    img_.at<Vec3b>(startY + pr, startX + pc) = color;
        return true;
    }

    bool draw_tile(const uint8_t data, const int grid_r, const int grid_c){
        if(is_reserved(grid_r, grid_c)) return false;

        int startX = MARGIN + grid_c * STRIDE;
        int startY = MARGIN + grid_r * STRIDE;

        const auto tile = Tile(data);
        const auto mask = tile.get_pattern();
        const auto color = tile.get_color();

        for(uint8_t pr = 0; pr < TILE_SIZE; ++pr)
            for(uint8_t pc = 0; pc < TILE_SIZE; ++pc)
                if((mask >> (pr * 8 + pc)) & 1)
                    img_.at<Vec3b>(startY + pr, startX + pc) = color;
        return true;
    }

    static bool is_reserved(int grid_r, int grid_c){
        if(grid_r < Config::ANCHOR_RESERVED_CELLS and grid_c < Config::ANCHOR_RESERVED_CELLS) return true;  // TL
        if(grid_r < Config::ANCHOR_RESERVED_CELLS and grid_c >= GRID_C - Config::ANCHOR_RESERVED_CELLS) return true;  // TR
        if(grid_r >= GRID_R - Config::ANCHOR_RESERVED_CELLS and grid_c < Config::ANCHOR_RESERVED_CELLS) return true;  // BL
        if(grid_r >= GRID_R - Config::ANCHOR_RESERVED_CELLS and grid_c >= GRID_C - Config::ANCHOR_RESERVED_CELLS) return true;  // BR
        return false;
    }

private:
    Mat& img_;
};