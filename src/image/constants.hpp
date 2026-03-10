#pragma once

#include "util/config.hpp"

// 图像常量
static const int IMG_WIDTH  = Config::IMG_WIDTH;
static const int IMG_HEIGHT = Config::IMG_HEIGHT;
static const int STRIDE     = Config::STRIDE;
static const int MARGIN     = Config::MARGIN;
static const int TILE_SIZE  = Config::TILE_SIZE;

static constexpr int GRID_R = (IMG_HEIGHT - MARGIN * 2) / STRIDE;
static constexpr int GRID_C = (IMG_WIDTH - MARGIN * 2) / STRIDE;

// Anchor 层级常量（TL/TR/BL 和 BR 共用相同尺寸，仅颜色不同）
const int ANCHOR_OUT_START = Config::ANCHOR_OUT_START;
const int ANCHOR_L1_SIZE   = Config::ANCHOR_L1_SIZE;
const int ANCHOR_L2_INSET  = Config::ANCHOR_L2_INSET;
const int ANCHOR_L2_SIZE   = Config::ANCHOR_L2_SIZE;
const int ANCHOR_L3_INSET  = Config::ANCHOR_L3_INSET;
const int ANCHOR_L3_SIZE   = Config::ANCHOR_L3_SIZE;
const int ANCHOR_L4_INSET  = Config::ANCHOR_L4_INSET;
const int ANCHOR_L4_SIZE   = Config::ANCHOR_L4_SIZE;