#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <avrt.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_3.h>
#include <dwmapi.h>
#include <mmsystem.h>
#include <objbase.h>
#endif

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string input_dir;
    double fps = 20.0;
    bool loop = true;
    bool pause_at_cycle = true;
    bool fullscreen = true;
    std::string window_name = "CameraDrop Player";
};

struct ScreenSize {
    int width = 0;
    int height = 0;
};

struct ViewState {
    ScreenSize screen;
    double zoom_scale = 1.0;
    bool fullscreen = true;
    bool redraw = true;
};

constexpr double kMinZoomScale = 0.10;
constexpr double kMaxZoomScale = 8.0;
constexpr double kWheelZoomFactor = 1.02;
constexpr double kCtrlWheelZoomFactor = 1.01;
constexpr double kShiftWheelZoomFactor = 1.05;
constexpr double kKeyZoomFactor = 1.05;

bool has_image_extension(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp";
}

void print_usage() {
    std::cout
        << "Usage: fullscreen_player --input <frames-dir> [--fps <n>]\n"
        << "                         [--once] [--continuous] [--windowed] [--window-name <name>]\n"
        << "Keys: Esc/q exit, Space pause, r restart, ',' prev, '.' next, +/- zoom, [/] integer zoom, 0 reset zoom\n"
        << "Mouse: wheel fine zoom, Ctrl+wheel ultra-fine, Shift+wheel coarse\n";
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            opts.input_dir = argv[++i];
        } else if (arg == "--fps" && i + 1 < argc) {
            opts.fps = std::stod(argv[++i]);
        } else if (arg == "--once") {
            opts.loop = false;
        } else if (arg == "--continuous") {
            opts.pause_at_cycle = false;
        } else if (arg == "--windowed") {
            opts.fullscreen = false;
        } else if (arg == "--window-name" && i + 1 < argc) {
            opts.window_name = argv[++i];
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (opts.input_dir.empty()) {
        throw std::runtime_error("Missing --input");
    }
    if (opts.fps <= 0.0) {
        throw std::runtime_error("FPS must be > 0");
    }
    return opts;
}

std::vector<fs::path> collect_frames(const fs::path& dir) {
    if (!fs::exists(dir)) {
        throw std::runtime_error("Input path does not exist: " + dir.string());
    }
    if (!fs::is_directory(dir)) {
        throw std::runtime_error("Input must be a directory of frames: " + dir.string());
    }

    std::vector<fs::path> frames;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && has_image_extension(entry.path())) {
            frames.push_back(entry.path());
        }
    }
    std::sort(frames.begin(), frames.end());
    if (frames.empty()) {
        throw std::runtime_error("No image frames found in: " + dir.string());
    }
    return frames;
}

std::vector<cv::Mat> load_frames(const std::vector<fs::path>& frame_paths) {
    std::vector<cv::Mat> frames;
    frames.reserve(frame_paths.size());

    int expected_cols = 0;
    int expected_rows = 0;
    for (const auto& path : frame_paths) {
        cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Failed to load frame: " + path.string());
        }
        if (expected_cols == 0) {
            expected_cols = image.cols;
            expected_rows = image.rows;
        } else if (image.cols != expected_cols || image.rows != expected_rows) {
            throw std::runtime_error("Frame size mismatch: " + path.string());
        }
        cv::Mat bgra;
        cv::cvtColor(image, bgra, cv::COLOR_BGR2BGRA);
        frames.push_back(std::move(bgra));
    }
    return frames;
}

int compute_fit_integer_scale(const cv::Mat& frame, const ScreenSize& screen) {
    if (screen.width <= 0 || screen.height <= 0 || frame.cols <= 0 || frame.rows <= 0) {
        return 1;
    }
    return std::max(1, std::min(screen.width / frame.cols, screen.height / frame.rows));
}

double clamp_zoom_scale(double zoom_scale) {
    return std::clamp(zoom_scale, kMinZoomScale, kMaxZoomScale);
}

double apply_zoom_factor(double current_scale, double factor, int steps) {
    if (steps <= 0) {
        return clamp_zoom_scale(current_scale);
    }
    return clamp_zoom_scale(current_scale * std::pow(factor, steps));
}

#ifdef _WIN32

template <typename T>
class ComPtr {
public:
    ComPtr() = default;
    ~ComPtr() { reset(); }

    ComPtr(const ComPtr&) = delete;
    ComPtr& operator=(const ComPtr&) = delete;

    ComPtr(ComPtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    ComPtr& operator=(ComPtr&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T* get() const { return ptr_; }
    T* operator->() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }

    T** put() {
        reset();
        return &ptr_;
    }

    void reset(T* value = nullptr) {
        if (ptr_) {
            ptr_->Release();
        }
        ptr_ = value;
    }

private:
    T* ptr_ = nullptr;
};

struct ScopedHandle {
    HANDLE handle = nullptr;

    ~ScopedHandle() { reset(); }

    void reset(HANDLE next = nullptr) {
        if (handle) {
            CloseHandle(handle);
        }
        handle = next;
    }
};

class TimerResolutionGuard {
public:
    TimerResolutionGuard() {
        active_ = (timeBeginPeriod(1) == TIMERR_NOERROR);
    }
    ~TimerResolutionGuard() {
        if (active_) {
            timeEndPeriod(1);
        }
    }

private:
    bool active_ = false;
};

class MmcssGuard {
public:
    MmcssGuard() {
        task_handle_ = AvSetMmThreadCharacteristics(TEXT("Games"), &task_index_);
        if (task_handle_) {
            AvSetMmThreadPriority(task_handle_, AVRT_PRIORITY_HIGH);
        }
        original_priority_ = GetThreadPriority(GetCurrentThread());
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    }

    ~MmcssGuard() {
        SetThreadPriority(GetCurrentThread(), original_priority_);
        if (task_handle_) {
            AvRevertMmThreadCharacteristics(task_handle_);
        }
    }

private:
    DWORD task_index_ = 0;
    HANDLE task_handle_ = nullptr;
    int original_priority_ = THREAD_PRIORITY_NORMAL;
};

class ComInitGuard {
public:
    ComInitGuard() {
        const HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
        initialized_ = SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE;
    }

    ~ComInitGuard() {
        if (initialized_) {
            CoUninitialize();
        }
    }

private:
    bool initialized_ = false;
};

struct MonitorInfo {
    RECT rect{0, 0, 0, 0};
    ScreenSize size;
    double refresh_hz = 60.0;
    uint64_t refresh_scaled = 60000000;
    std::wstring device_name;
};

struct QuadParams {
    float dst_rect[4];
    float src_rect[4];
};

void throw_hresult(const char* what, HRESULT hr) {
    std::ostringstream oss;
    oss << what << " (HRESULT=0x" << std::hex << std::uppercase
        << static_cast<unsigned long>(hr) << ")";
    throw std::runtime_error(oss.str());
}

std::wstring utf8_to_wide(const std::string& input) {
    if (input.empty()) {
        return {};
    }
    const int size = MultiByteToWideChar(CP_UTF8, 0, input.c_str(), -1, nullptr, 0);
    if (size <= 0) {
        return std::wstring(input.begin(), input.end());
    }
    std::wstring wide(static_cast<size_t>(size - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, input.c_str(), -1, wide.data(), size);
    return wide;
}

void enable_windows_dpi_awareness() {
    HMODULE user32 = LoadLibraryA("user32.dll");
    if (user32) {
        using SetProcessDpiAwarenessContextFn = BOOL (WINAPI*)(HANDLE);
        const auto set_context = reinterpret_cast<SetProcessDpiAwarenessContextFn>(
            GetProcAddress(user32, "SetProcessDpiAwarenessContext"));
        if (set_context) {
            if (set_context(reinterpret_cast<HANDLE>(-4))) {
                FreeLibrary(user32);
                return;
            }
            if (set_context(reinterpret_cast<HANDLE>(-3))) {
                FreeLibrary(user32);
                return;
            }
        }

        using SetProcessDPIAwareFn = BOOL (WINAPI*)();
        const auto set_aware = reinterpret_cast<SetProcessDPIAwareFn>(
            GetProcAddress(user32, "SetProcessDPIAware"));
        if (set_aware) {
            set_aware();
        }
        FreeLibrary(user32);
    }
}

MonitorInfo detect_primary_monitor() {
    MonitorInfo info;
    const POINT origin{0, 0};
    const HMONITOR monitor = MonitorFromPoint(origin, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFOEXW monitor_info{};
    monitor_info.cbSize = sizeof(monitor_info);
    if (!GetMonitorInfoW(monitor, &monitor_info)) {
        throw std::runtime_error("GetMonitorInfoW failed");
    }

    info.rect = monitor_info.rcMonitor;
    info.size.width = monitor_info.rcMonitor.right - monitor_info.rcMonitor.left;
    info.size.height = monitor_info.rcMonitor.bottom - monitor_info.rcMonitor.top;
    info.device_name = monitor_info.szDevice;

    DWM_TIMING_INFO timing{};
    timing.cbSize = sizeof(timing);
    if (SUCCEEDED(DwmGetCompositionTimingInfo(nullptr, &timing))
        && timing.rateRefresh.uiNumerator > 0
        && timing.rateRefresh.uiDenominator > 0) {
        info.refresh_hz = static_cast<double>(timing.rateRefresh.uiNumerator)
                        / static_cast<double>(timing.rateRefresh.uiDenominator);
    } else {
        DEVMODEW mode{};
        mode.dmSize = sizeof(mode);
        if (EnumDisplaySettingsW(monitor_info.szDevice, ENUM_CURRENT_SETTINGS, &mode)
            && mode.dmDisplayFrequency > 1) {
            info.refresh_hz = static_cast<double>(mode.dmDisplayFrequency);
        }
    }
    info.refresh_scaled = static_cast<uint64_t>(std::llround(info.refresh_hz * 1000000.0));
    return info;
}

bool handle_zoom_key(ViewState& view, int key, double default_scale) {
    auto set_zoom_scale = [&view](double next_scale) -> bool {
        next_scale = clamp_zoom_scale(next_scale);
        if (std::abs(next_scale - view.zoom_scale) < 1e-6) {
            return false;
        }
        view.zoom_scale = next_scale;
        view.redraw = true;
        std::cout << "zoom_scale=" << view.zoom_scale << std::endl;
        return true;
    };

    auto step_zoom_scale = [&view, &set_zoom_scale](double factor, int direction) -> bool {
        if (direction == 0 || factor <= 0.0 || std::abs(factor - 1.0) < 1e-9) {
            return false;
        }
        const double base = direction > 0 ? factor : (1.0 / factor);
        return set_zoom_scale(apply_zoom_factor(view.zoom_scale, base, 1));
    };

    auto snap_zoom_to_integer = [&view, &set_zoom_scale](int direction) -> bool {
        if (direction == 0) {
            return false;
        }
        if (direction > 0) {
            const int next_integer = std::max(1, static_cast<int>(std::floor(view.zoom_scale + 1e-6)) + 1);
            return set_zoom_scale(static_cast<double>(next_integer));
        }
        const int previous_integer = std::max(1, static_cast<int>(std::ceil(view.zoom_scale - 1e-6)) - 1);
        return set_zoom_scale(static_cast<double>(previous_integer));
    };

    switch (key) {
    case '+':
    case '=':
        return step_zoom_scale(kKeyZoomFactor, 1);
    case '-':
    case '_':
        return step_zoom_scale(kKeyZoomFactor, -1);
    case '[':
    case '{':
        return snap_zoom_to_integer(-1);
    case ']':
    case '}':
        return snap_zoom_to_integer(1);
    case '0':
        return set_zoom_scale(default_scale);
    default:
        return false;
    }
}

class DxgiPlayer {
public:
    DxgiPlayer(const Options& opts, std::vector<cv::Mat>&& frames)
        : opts_(opts), frames_(std::move(frames)) {
        monitor_ = detect_primary_monitor();
        default_scale_ = static_cast<double>(compute_fit_integer_scale(frames_.front(), monitor_.size));
        view_.screen = opts_.fullscreen ? monitor_.size : ScreenSize{frames_.front().cols, frames_.front().rows};
        view_.zoom_scale = default_scale_;
        view_.fullscreen = opts_.fullscreen;
        fps_scaled_ = static_cast<uint64_t>(std::llround(opts_.fps * 1000000.0));
        refresh_scaled_ = monitor_.refresh_scaled;
        refresh_hz_ = monitor_.refresh_hz;
        if (fps_scaled_ == 0) {
            throw std::runtime_error("FPS must be > 0");
        }
        if (fps_scaled_ > refresh_scaled_) {
            std::ostringstream oss;
            oss << "Requested fps " << opts_.fps << " exceeds display refresh " << std::fixed
                << std::setprecision(3) << refresh_hz_ << " Hz";
            throw std::runtime_error(oss.str());
        }
        const double ratio = refresh_hz_ / opts_.fps;
        const double nearest = std::round(ratio);
        exact_divisor_ = (nearest >= 1.0 && std::abs(ratio - nearest) < 1e-4);
        integer_vsync_divisor_ = exact_divisor_ ? static_cast<int>(nearest) : 0;
    }

    int run() {
        create_window();
        create_device_and_swapchain();
        create_pipeline();
        print_startup_banner();

        while (running_) {
            if (paused_) {
                if (view_.redraw) {
                    if (!wait_for_present_slot()) {
                        continue;
                    }
                    present_current_frame();
                } else {
                    wait_for_messages();
                }
                continue;
            }

            if (!wait_for_present_slot()) {
                continue;
            }
            present_current_frame();
            advance_after_present();
        }

        return 0;
    }

private:
    static LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
        if (message == WM_NCCREATE) {
            const auto* create = reinterpret_cast<CREATESTRUCTW*>(lparam);
            auto* self = static_cast<DxgiPlayer*>(create->lpCreateParams);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
            self->hwnd_ = hwnd;
        }

        auto* self = reinterpret_cast<DxgiPlayer*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        if (!self) {
            return DefWindowProcW(hwnd, message, wparam, lparam);
        }
        return self->handle_message(message, wparam, lparam);
    }

    LRESULT handle_message(UINT message, WPARAM wparam, LPARAM lparam) {
        switch (message) {
        case WM_CLOSE:
            running_ = false;
            DestroyWindow(hwnd_);
            return 0;
        case WM_DESTROY:
            running_ = false;
            PostQuitMessage(0);
            return 0;
        case WM_SIZE: {
            const int width = LOWORD(lparam);
            const int height = HIWORD(lparam);
            on_resize(width, height);
            return 0;
        }
        case WM_MOUSEWHEEL:
            on_mouse_wheel(GET_WHEEL_DELTA_WPARAM(wparam), GET_KEYSTATE_WPARAM(wparam));
            return 0;
        case WM_KEYDOWN:
            on_key(static_cast<int>(wparam));
            return 0;
        default:
            return DefWindowProcW(hwnd_, message, wparam, lparam);
        }
    }

    void create_window() {
        const HINSTANCE instance = GetModuleHandleW(nullptr);
        const wchar_t* class_name = L"CameraDropStrictPlayerWindow";

        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = &DxgiPlayer::WndProc;
        wc.hInstance = instance;
        wc.lpszClassName = class_name;
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.style = CS_HREDRAW | CS_VREDRAW;
        if (!RegisterClassExW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
            throw std::runtime_error("RegisterClassExW failed");
        }

        DWORD style = WS_VISIBLE;
        DWORD ex_style = WS_EX_APPWINDOW;
        RECT window_rect{};
        if (opts_.fullscreen) {
            style |= WS_POPUP;
            ex_style |= WS_EX_TOPMOST;
            window_rect = monitor_.rect;
        } else {
            style |= WS_OVERLAPPEDWINDOW;
            window_rect.left = 0;
            window_rect.top = 0;
            window_rect.right = frames_.front().cols;
            window_rect.bottom = frames_.front().rows;
            AdjustWindowRectEx(&window_rect, style, FALSE, ex_style);
            const int width = window_rect.right - window_rect.left;
            const int height = window_rect.bottom - window_rect.top;
            const int x = monitor_.rect.left + std::max(0, (monitor_.size.width - width) / 2);
            const int y = monitor_.rect.top + std::max(0, (monitor_.size.height - height) / 2);
            window_rect.left = x;
            window_rect.top = y;
            window_rect.right = x + width;
            window_rect.bottom = y + height;
        }

        const std::wstring title = utf8_to_wide(opts_.window_name);
        hwnd_ = CreateWindowExW(
            ex_style,
            class_name,
            title.c_str(),
            style,
            window_rect.left,
            window_rect.top,
            window_rect.right - window_rect.left,
            window_rect.bottom - window_rect.top,
            nullptr,
            nullptr,
            instance,
            this);
        if (!hwnd_) {
            throw std::runtime_error("CreateWindowExW failed");
        }

        ShowWindow(hwnd_, SW_SHOW);
        UpdateWindow(hwnd_);
        if (opts_.fullscreen) {
            SetWindowPos(hwnd_,
                         HWND_TOPMOST,
                         monitor_.rect.left,
                         monitor_.rect.top,
                         monitor_.size.width,
                         monitor_.size.height,
                         SWP_SHOWWINDOW);
        }
    }

    void create_device_and_swapchain() {
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        const D3D_FEATURE_LEVEL levels[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0,
        };
        D3D_FEATURE_LEVEL level{};
        HRESULT hr = D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            flags,
            levels,
            static_cast<UINT>(std::size(levels)),
            D3D11_SDK_VERSION,
            device_.put(),
            &level,
            context_.put());
        if (FAILED(hr)) {
            hr = D3D11CreateDevice(
                nullptr,
                D3D_DRIVER_TYPE_WARP,
                nullptr,
                flags,
                levels,
                static_cast<UINT>(std::size(levels)),
                D3D11_SDK_VERSION,
                device_.put(),
                &level,
                context_.put());
        }
        if (FAILED(hr)) {
            throw_hresult("D3D11CreateDevice failed", hr);
        }

        ComPtr<IDXGIDevice> dxgi_device;
        hr = device_->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(dxgi_device.put()));
        if (FAILED(hr)) {
            throw_hresult("QueryInterface IDXGIDevice failed", hr);
        }

        ComPtr<IDXGIAdapter> adapter;
        hr = dxgi_device->GetAdapter(adapter.put());
        if (FAILED(hr)) {
            throw_hresult("GetAdapter failed", hr);
        }

        ComPtr<IDXGIFactory2> factory;
        hr = adapter->GetParent(__uuidof(IDXGIFactory2), reinterpret_cast<void**>(factory.put()));
        if (FAILED(hr)) {
            throw_hresult("GetParent IDXGIFactory2 failed", hr);
        }
        factory_ = std::move(factory);
        factory_->MakeWindowAssociation(hwnd_, DXGI_MWA_NO_ALT_ENTER);

        recreate_swapchain();
    }

    void recreate_swapchain() {
        if (!device_ || !factory_) {
            throw std::runtime_error("DXGI device not initialized");
        }

        const ScreenSize size = query_client_size();
        if (size.width <= 0 || size.height <= 0) {
            return;
        }
        view_.screen = size;

        rtv_.reset();
        frame_latency_waitable_.reset();
        if (swap_chain_) {
            swap_chain_->ResizeBuffers(0, static_cast<UINT>(size.width), static_cast<UINT>(size.height),
                                       DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);
        } else {
            DXGI_SWAP_CHAIN_DESC1 desc{};
            desc.Width = static_cast<UINT>(size.width);
            desc.Height = static_cast<UINT>(size.height);
            desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            desc.SampleDesc.Count = 1;
            desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            desc.BufferCount = 2;
            desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
            desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
            desc.Scaling = DXGI_SCALING_NONE;
            desc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

            ComPtr<IDXGISwapChain1> swap_chain1;
            HRESULT hr = factory_->CreateSwapChainForHwnd(
                device_.get(),
                hwnd_,
                &desc,
                nullptr,
                nullptr,
                swap_chain1.put());
            if (FAILED(hr)) {
                desc.Scaling = DXGI_SCALING_STRETCH;
                hr = factory_->CreateSwapChainForHwnd(
                    device_.get(),
                    hwnd_,
                    &desc,
                    nullptr,
                    nullptr,
                    swap_chain1.put());
            }
            if (FAILED(hr)) {
                throw_hresult("CreateSwapChainForHwnd failed", hr);
            }

            ComPtr<IDXGISwapChain2> swap_chain2;
            hr = swap_chain1->QueryInterface(__uuidof(IDXGISwapChain2), reinterpret_cast<void**>(swap_chain2.put()));
            if (FAILED(hr)) {
                throw_hresult("QueryInterface IDXGISwapChain2 failed", hr);
            }
            swap_chain_ = std::move(swap_chain2);
        }

        HRESULT hr = swap_chain_->SetMaximumFrameLatency(1);
        if (FAILED(hr)) {
            throw_hresult("SetMaximumFrameLatency failed", hr);
        }
        frame_latency_waitable_.reset(swap_chain_->GetFrameLatencyWaitableObject());
        if (!frame_latency_waitable_.handle) {
            throw std::runtime_error("GetFrameLatencyWaitableObject failed");
        }

        ComPtr<ID3D11Texture2D> back_buffer;
        hr = swap_chain_->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(back_buffer.put()));
        if (FAILED(hr)) {
            throw_hresult("GetBuffer failed", hr);
        }
        hr = device_->CreateRenderTargetView(back_buffer.get(), nullptr, rtv_.put());
        if (FAILED(hr)) {
            throw_hresult("CreateRenderTargetView failed", hr);
        }

        D3D11_VIEWPORT viewport{};
        viewport.TopLeftX = 0.0f;
        viewport.TopLeftY = 0.0f;
        viewport.Width = static_cast<float>(size.width);
        viewport.Height = static_cast<float>(size.height);
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 1.0f;
        context_->RSSetViewports(1, &viewport);
        view_.redraw = true;
    }

    void create_pipeline() {
        static constexpr const char* kVertexShader = R"(
cbuffer QuadParams : register(b0) {
    float4 dst_rect;
    float4 src_rect;
};

struct VSOut {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

VSOut main(uint vertex_id : SV_VertexID) {
    float2 pos[4] = {
        float2(dst_rect.x, dst_rect.y),
        float2(dst_rect.z, dst_rect.y),
        float2(dst_rect.x, dst_rect.w),
        float2(dst_rect.z, dst_rect.w)
    };
    float2 uv[4] = {
        float2(src_rect.x, src_rect.y),
        float2(src_rect.z, src_rect.y),
        float2(src_rect.x, src_rect.w),
        float2(src_rect.z, src_rect.w)
    };

    VSOut outv;
    outv.pos = float4(pos[vertex_id], 0.0, 1.0);
    outv.uv = uv[vertex_id];
    return outv;
}
)";

        static constexpr const char* kPixelShader = R"(
Texture2D frame_tex : register(t0);
SamplerState point_sampler : register(s0);

float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
    return frame_tex.Sample(point_sampler, uv);
}
)";

        UINT compile_flags = D3DCOMPILE_ENABLE_STRICTNESS;
        ComPtr<ID3DBlob> vs_blob;
        ComPtr<ID3DBlob> ps_blob;
        ComPtr<ID3DBlob> errors;

        HRESULT hr = D3DCompile(
            kVertexShader,
            std::strlen(kVertexShader),
            nullptr,
            nullptr,
            nullptr,
            "main",
            "vs_4_0",
            compile_flags,
            0,
            vs_blob.put(),
            errors.put());
        if (FAILED(hr)) {
            const char* msg = errors ? static_cast<const char*>(errors->GetBufferPointer()) : "D3DCompile VS failed";
            throw std::runtime_error(msg);
        }
        errors.reset();

        hr = D3DCompile(
            kPixelShader,
            std::strlen(kPixelShader),
            nullptr,
            nullptr,
            nullptr,
            "main",
            "ps_4_0",
            compile_flags,
            0,
            ps_blob.put(),
            errors.put());
        if (FAILED(hr)) {
            const char* msg = errors ? static_cast<const char*>(errors->GetBufferPointer()) : "D3DCompile PS failed";
            throw std::runtime_error(msg);
        }

        hr = device_->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, vs_.put());
        if (FAILED(hr)) {
            throw_hresult("CreateVertexShader failed", hr);
        }
        hr = device_->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, ps_.put());
        if (FAILED(hr)) {
            throw_hresult("CreatePixelShader failed", hr);
        }

        D3D11_BUFFER_DESC quad_desc{};
        quad_desc.ByteWidth = sizeof(QuadParams);
        quad_desc.Usage = D3D11_USAGE_DEFAULT;
        quad_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        hr = device_->CreateBuffer(&quad_desc, nullptr, quad_buffer_.put());
        if (FAILED(hr)) {
            throw_hresult("CreateBuffer failed", hr);
        }

        D3D11_SAMPLER_DESC sampler_desc{};
        sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
        sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
        sampler_desc.MinLOD = 0;
        sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
        hr = device_->CreateSamplerState(&sampler_desc, sampler_.put());
        if (FAILED(hr)) {
            throw_hresult("CreateSamplerState failed", hr);
        }

        create_frame_texture();
    }

    void create_frame_texture() {
        const cv::Mat& frame = frames_.front();

        D3D11_TEXTURE2D_DESC tex_desc{};
        tex_desc.Width = static_cast<UINT>(frame.cols);
        tex_desc.Height = static_cast<UINT>(frame.rows);
        tex_desc.MipLevels = 1;
        tex_desc.ArraySize = 1;
        tex_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        tex_desc.SampleDesc.Count = 1;
        tex_desc.Usage = D3D11_USAGE_DYNAMIC;
        tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        tex_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        HRESULT hr = device_->CreateTexture2D(&tex_desc, nullptr, frame_texture_.put());
        if (FAILED(hr)) {
            throw_hresult("CreateTexture2D failed", hr);
        }
        hr = device_->CreateShaderResourceView(frame_texture_.get(), nullptr, frame_srv_.put());
        if (FAILED(hr)) {
            throw_hresult("CreateShaderResourceView failed", hr);
        }
        uploaded_index_ = std::numeric_limits<size_t>::max();
        view_.redraw = true;
    }

    ScreenSize query_client_size() const {
        RECT rect{};
        GetClientRect(hwnd_, &rect);
        return {rect.right - rect.left, rect.bottom - rect.top};
    }

    void print_startup_banner() const {
        bool clipped = false;
        compute_quad_params(frames_.front(), &clipped);
        std::cout << "loaded_frames=" << frames_.size()
                  << " frame_size=" << frames_.front().cols << "x" << frames_.front().rows
                  << " screen_size=" << view_.screen.width << "x" << view_.screen.height
                  << " zoom_scale=" << view_.zoom_scale
                  << " fps=" << opts_.fps
                  << " refresh_hz=" << std::fixed << std::setprecision(3) << refresh_hz_
                  << " loop=" << (opts_.loop ? "true" : "false")
                  << " pause_at_cycle=" << (opts_.pause_at_cycle ? "true" : "false")
                  << std::defaultfloat << std::endl;
        if (clipped) {
            std::cout << "warning: frame exceeds screen; fullscreen playback is center-cropped to preserve raw pixel geometry" << std::endl;
        }
        if (exact_divisor_) {
            std::cout << "vsync_cadence: exact " << integer_vsync_divisor_
                      << " vertical blank(s) per frame" << std::endl;
        } else {
            std::cout << "warning: requested fps is not an integer divisor of display refresh;"
                      << " cadence jitter is mathematically unavoidable on this monitor" << std::endl;
        }
        std::cout << "display_backend: DXGI flip-model swap chain + frame-latency waitable object + point-sampled scaling" << std::endl;
        std::cout << "controls: Esc/q exit, Space pause, r restart, ',' prev, '.' next, +/- coarse zoom, [/] integer zoom, 0 reset, mouse wheel fine zoom, Ctrl+wheel ultra-fine, Shift+wheel coarse" << std::endl;
    }

    bool wait_for_present_slot() {
        while (running_) {
            HANDLE wait_handle = frame_latency_waitable_.handle;
            const DWORD result = MsgWaitForMultipleObjectsEx(
                1,
                &wait_handle,
                INFINITE,
                QS_ALLINPUT,
                MWMO_INPUTAVAILABLE);

            if (result == WAIT_OBJECT_0) {
                return true;
            }
            if (result == WAIT_OBJECT_0 + 1) {
                pump_messages();
                if (!running_) {
                    return false;
                }
                if (paused_ && !view_.redraw) {
                    return false;
                }
                continue;
            }
            if (result == WAIT_FAILED) {
                throw std::runtime_error("MsgWaitForMultipleObjectsEx failed");
            }
        }
        return false;
    }

    void wait_for_messages() {
        const DWORD result = MsgWaitForMultipleObjectsEx(
            0,
            nullptr,
            INFINITE,
            QS_ALLINPUT,
            MWMO_INPUTAVAILABLE);
        if (result == WAIT_OBJECT_0) {
            pump_messages();
        } else if (result == WAIT_FAILED) {
            throw std::runtime_error("MsgWaitForMultipleObjectsEx failed while paused");
        }
    }

    void pump_messages() {
        MSG msg{};
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                running_ = false;
                return;
            }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }

    void on_resize(int width, int height) {
        if (width <= 0 || height <= 0 || !swap_chain_) {
            return;
        }
        recreate_swapchain();
    }

    void on_mouse_wheel(short delta, WORD key_flags) {
        if (!view_.fullscreen || delta == 0) {
            return;
        }
        const int steps = std::max(1, static_cast<int>(std::abs(delta) / WHEEL_DELTA));
        double step_factor = kWheelZoomFactor;
        if ((key_flags & MK_CONTROL) != 0) {
            step_factor = kCtrlWheelZoomFactor;
        } else if ((key_flags & MK_SHIFT) != 0) {
            step_factor = kShiftWheelZoomFactor;
        }
        const double base = delta > 0 ? step_factor : (1.0 / step_factor);
        const double next = apply_zoom_factor(view_.zoom_scale, base, steps);
        if (std::abs(next - view_.zoom_scale) >= 1e-6) {
            view_.zoom_scale = next;
            view_.redraw = true;
            std::cout << "zoom_scale=" << view_.zoom_scale << std::endl;
        }
    }

    void on_key(int key) {
        if (handle_zoom_key(view_, key, default_scale_)) {
            return;
        }

        switch (key) {
        case VK_ESCAPE:
        case 'Q':
            running_ = false;
            DestroyWindow(hwnd_);
            return;
        case VK_SPACE:
            if (paused_) {
                playback_resync_ = true;
            }
            paused_ = !paused_;
            view_.redraw = true;
            return;
        case 'R':
            index_ = 0;
            view_.redraw = true;
            playback_resync_ = true;
            return;
        case VK_OEM_COMMA:
            paused_ = true;
            index_ = (index_ + frames_.size() - 1) % frames_.size();
            view_.redraw = true;
            playback_resync_ = true;
            return;
        case VK_OEM_PERIOD:
            paused_ = true;
            index_ = (index_ + 1) % frames_.size();
            view_.redraw = true;
            playback_resync_ = true;
            return;
        default:
            return;
        }
    }

    QuadParams compute_quad_params(const cv::Mat& frame, bool* clipped_out = nullptr) const {
        const ScreenSize screen = view_.screen;
        if (screen.width <= 0 || screen.height <= 0) {
            throw std::runtime_error("Invalid client size");
        }

        const double scale = view_.fullscreen ? clamp_zoom_scale(view_.zoom_scale) : 1.0;
        const int src_w = std::min(frame.cols, std::max(1, static_cast<int>(std::ceil(screen.width / scale))));
        const int src_h = std::min(frame.rows, std::max(1, static_cast<int>(std::ceil(screen.height / scale))));
        const int src_x = std::max(0, (frame.cols - src_w) / 2);
        const int src_y = std::max(0, (frame.rows - src_h) / 2);
        const int dst_w = std::min(screen.width, std::max(1, static_cast<int>(std::lround(src_w * scale))));
        const int dst_h = std::min(screen.height, std::max(1, static_cast<int>(std::lround(src_h * scale))));
        const int dst_x = std::max(0, (screen.width - dst_w) / 2);
        const int dst_y = std::max(0, (screen.height - dst_h) / 2);

        const bool clipped = (src_w < frame.cols) || (src_h < frame.rows);
        if (clipped_out) {
            *clipped_out = clipped;
        }

        const float left = -1.0f + 2.0f * static_cast<float>(dst_x) / static_cast<float>(screen.width);
        const float right = -1.0f + 2.0f * static_cast<float>(dst_x + dst_w) / static_cast<float>(screen.width);
        const float top = 1.0f - 2.0f * static_cast<float>(dst_y) / static_cast<float>(screen.height);
        const float bottom = 1.0f - 2.0f * static_cast<float>(dst_y + dst_h) / static_cast<float>(screen.height);

        QuadParams params{};
        params.dst_rect[0] = left;
        params.dst_rect[1] = top;
        params.dst_rect[2] = right;
        params.dst_rect[3] = bottom;
        params.src_rect[0] = static_cast<float>(src_x) / static_cast<float>(frame.cols);
        params.src_rect[1] = static_cast<float>(src_y) / static_cast<float>(frame.rows);
        params.src_rect[2] = static_cast<float>(src_x + src_w) / static_cast<float>(frame.cols);
        params.src_rect[3] = static_cast<float>(src_y + src_h) / static_cast<float>(frame.rows);
        return params;
    }

    void upload_current_frame_if_needed() {
        if (uploaded_index_ == index_ && !view_.redraw) {
            return;
        }

        ID3D11ShaderResourceView* null_srv = nullptr;
        context_->PSSetShaderResources(0, 1, &null_srv);

        D3D11_MAPPED_SUBRESOURCE mapped{};
        HRESULT hr = context_->Map(frame_texture_.get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
        if (FAILED(hr)) {
            throw_hresult("Map frame texture failed", hr);
        }

        const cv::Mat& frame = frames_[index_];
        for (int y = 0; y < frame.rows; ++y) {
            std::memcpy(
                static_cast<std::uint8_t*>(mapped.pData) + static_cast<size_t>(y) * mapped.RowPitch,
                frame.ptr(y),
                static_cast<size_t>(frame.cols) * 4);
        }
        context_->Unmap(frame_texture_.get(), 0);
        uploaded_index_ = index_;
    }

    void present_current_frame() {
        upload_current_frame_if_needed();

        const QuadParams params = compute_quad_params(frames_[index_]);
        context_->UpdateSubresource(quad_buffer_.get(), 0, nullptr, &params, 0, 0);

        static const float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        ID3D11RenderTargetView* rtv = rtv_.get();
        ID3D11ShaderResourceView* srv = frame_srv_.get();
        ID3D11SamplerState* sampler = sampler_.get();
        ID3D11Buffer* cbuffer = quad_buffer_.get();

        context_->OMSetRenderTargets(1, &rtv, nullptr);
        context_->ClearRenderTargetView(rtv_.get(), clear_color);
        context_->IASetInputLayout(nullptr);
        context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        context_->VSSetShader(vs_.get(), nullptr, 0);
        context_->VSSetConstantBuffers(0, 1, &cbuffer);
        context_->PSSetShader(ps_.get(), nullptr, 0);
        context_->PSSetConstantBuffers(0, 1, &cbuffer);
        context_->PSSetSamplers(0, 1, &sampler);
        context_->PSSetShaderResources(0, 1, &srv);
        context_->Draw(4, 0);

        HRESULT hr = swap_chain_->Present(1, 0);
        if (FAILED(hr)) {
            throw_hresult("Present failed", hr);
        }

        view_.redraw = false;
    }

    void advance_after_present() {
        if (playback_resync_) {
            cadence_accum_ = 0;
            playback_resync_ = false;
        }

        cadence_accum_ += fps_scaled_;
        while (cadence_accum_ >= refresh_scaled_) {
            cadence_accum_ -= refresh_scaled_;
            ++index_;
            if (index_ >= frames_.size()) {
                if (!opts_.loop) {
                    running_ = false;
                    DestroyWindow(hwnd_);
                    return;
                }
                index_ = 0;
                if (opts_.pause_at_cycle) {
                    paused_ = true;
                    playback_resync_ = true;
                    std::cout << "cycle complete, paused at frame 0; press Space to continue" << std::endl;
                }
            }
            view_.redraw = true;
            if (paused_) {
                return;
            }
        }
    }

    Options opts_;
    std::vector<cv::Mat> frames_;
    MonitorInfo monitor_;
    ViewState view_;
    double default_scale_ = 1.0;

    ComInitGuard com_init_;
    TimerResolutionGuard timer_resolution_;
    MmcssGuard mmcss_;

    HWND hwnd_ = nullptr;
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<IDXGIFactory2> factory_;
    ComPtr<IDXGISwapChain2> swap_chain_;
    ComPtr<ID3D11RenderTargetView> rtv_;
    ComPtr<ID3D11VertexShader> vs_;
    ComPtr<ID3D11PixelShader> ps_;
    ComPtr<ID3D11Buffer> quad_buffer_;
    ComPtr<ID3D11SamplerState> sampler_;
    ComPtr<ID3D11Texture2D> frame_texture_;
    ComPtr<ID3D11ShaderResourceView> frame_srv_;
    ScopedHandle frame_latency_waitable_;

    size_t index_ = 0;
    size_t uploaded_index_ = std::numeric_limits<size_t>::max();
    bool running_ = true;
    bool paused_ = false;
    bool playback_resync_ = true;
    uint64_t cadence_accum_ = 0;
    uint64_t fps_scaled_ = 0;
    uint64_t refresh_scaled_ = 0;
    double refresh_hz_ = 60.0;
    bool exact_divisor_ = false;
    int integer_vsync_divisor_ = 0;
};

#endif

}  // namespace

int main(int argc, char** argv) {
    try {
#ifdef _WIN32
        enable_windows_dpi_awareness();
        const Options opts = parse_args(argc, argv);
        const std::vector<fs::path> frame_paths = collect_frames(fs::absolute(opts.input_dir));
        std::vector<cv::Mat> frames = load_frames(frame_paths);
        DxgiPlayer player(opts, std::move(frames));
        return player.run();
#else
        (void)argc;
        (void)argv;
        throw std::runtime_error("fullscreen_player currently requires Windows");
#endif
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        print_usage();
        return 1;
    }
}
