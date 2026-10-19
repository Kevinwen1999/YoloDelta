#pragma once
#include "delta/capture.hpp"
#include <windows.h>
#include <dxgi1_2.h>
#include <dwmapi.h>
#include <wrl/client.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace delta::bench {
inline std::string displayUtf8(const std::wstring& value) {
    const int bytes=WideCharToMultiByte(CP_UTF8,0,value.data(),static_cast<int>(value.size()),nullptr,0,nullptr,nullptr);
    std::string out(bytes,'\0');
    WideCharToMultiByte(CP_UTF8,0,value.data(),static_cast<int>(value.size()),out.data(),bytes,nullptr,nullptr);
    return out;
}
struct CaptureDisplay {
    RECT bounds{};
    std::wstring adapter,output;
};
inline CaptureDisplay captureDisplay(const StaticConfig& config) {
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    Microsoft::WRL::ComPtr<IDXGIOutput> output;
    DXGI_ADAPTER_DESC1 ad{};DXGI_OUTPUT_DESC od{};
    if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))
        || FAILED(factory->EnumAdapters1(config.capture_device_idx,&adapter))
        || FAILED(adapter->EnumOutputs(config.capture_output_idx,&output))
        || FAILED(adapter->GetDesc1(&ad)) || FAILED(output->GetDesc(&od)) || !od.AttachedToDesktop)
        throw std::runtime_error("Configured capture display is unavailable");
    return {od.DesktopCoordinates,ad.Description,od.DeviceName};
}

// A small provenance marker, not evaluator truth. Both rows must agree and the
// complement row detects corruption/tearing. Coordinates are display-local and
// inside the production minimum centered crop (320 px).
inline constexpr int markerWidth=96,markerHeight=8;
inline void writeSceneMarker(std::vector<std::uint8_t>& bgra,int width,int height,std::uint32_t id) {
    const int left=width/2-markerWidth/2,top=height/2+140;
    for(int y=0;y<markerHeight;++y)for(int x=0;x<markerWidth;++x) {
        const bool bit=((id>>(x/3))&1U) ^ (y>=markerHeight/2);
        const auto at=(static_cast<std::size_t>(top+y)*width+left+x)*4;
        bgra[at]=bgra[at+1]=bgra[at+2]=bit?240:16;bgra[at+3]=255;
    }
}
inline std::optional<std::uint32_t> readSceneMarker(const FramePacket& frame,int width,int height) {
    const int left=width/2-markerWidth/2-frame.capture.left;
    const int top=height/2+140-frame.capture.top;
    if(left<0 || top<0 || left+markerWidth>frame.width || top+markerHeight>frame.height
        || frame.bgr.size()!=static_cast<std::size_t>(frame.width)*frame.height*3)return {};
    std::uint32_t id=0;
    for(int bit=0;bit<32;++bit) {
        std::array<bool,2> value{};
        for(int row=0;row<2;++row) {
            const auto at=(static_cast<std::size_t>(top+row*4+1)*frame.width+left+bit*3+1)*3;
            const auto v=frame.bgr[at];
            if(v>64 && v<192)return {};
            if(std::abs(int(frame.bgr[at+1])-v)>16 || std::abs(int(frame.bgr[at+2])-v)>16)return {};
            value[row]=v>=192;
        }
        if(value[0]==value[1])return {};
        if(value[0])id|=std::uint32_t{1}<<bit;
    }
    return id;
}

class VisibleSceneWindow {
public:
    explicit VisibleSceneWindow(const CaptureDisplay& display):bounds_(display.bounds) {
        WNDCLASSW wc{};wc.lpfnWndProc=DefWindowProcW;wc.hInstance=GetModuleHandleW(nullptr);
        wc.lpszClassName=L"DeltaNoDeviceSimulation";
        RegisterClassW(&wc);
        window_=CreateWindowExW(WS_EX_TOPMOST,wc.lpszClassName,
            L"DELTA VISIBLE SIMULATION - NO MOUSE OR SERIAL OUTPUT",WS_POPUP|WS_VISIBLE,
            bounds_.left,bounds_.top,bounds_.right-bounds_.left,bounds_.bottom-bounds_.top,
            nullptr,nullptr,wc.hInstance,nullptr);
        if(!window_)throw std::runtime_error("Cannot create scene window");
        ShowWindow(window_,SW_SHOW);SetWindowPos(window_,HWND_TOPMOST,bounds_.left,bounds_.top,
            bounds_.right-bounds_.left,bounds_.bottom-bounds_.top,SWP_SHOWWINDOW);
        dc_=GetDC(window_);
        if(!dc_)throw std::runtime_error("Cannot acquire scene window DC");
    }
    ~VisibleSceneWindow(){if(dc_)ReleaseDC(window_,dc_);if(IsWindow(window_))DestroyWindow(window_);}
    HWND handle()const{return window_;}
    void present(const std::vector<std::uint8_t>& bgra,int width,int height) {
        MSG message{};
        while(PeekMessageW(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessageW(&message);}
        if(!IsWindow(window_))throw std::runtime_error("Scene window was closed");
        BITMAPINFO info{};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
        info.bmiHeader.biWidth=width;info.bmiHeader.biHeight=-height;
        info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;info.bmiHeader.biCompression=BI_RGB;
        if(StretchDIBits(dc_,0,0,width,height,0,0,width,height,bgra.data(),&info,DIB_RGB_COLORS,SRCCOPY)==GDI_ERROR)
            throw std::runtime_error("Scene presentation failed");
        GdiFlush();
        if(FAILED(DwmFlush()))throw std::runtime_error("DWM presentation fence failed");
    }
    static bool alignedAndUnobscured(HWND window,const CaptureDisplay& display,const CaptureRegion& crop) {
        RECT client{};POINT origin{};
        if(!IsWindowVisible(window) || IsIconic(window) || !GetClientRect(window,&client)
            || !ClientToScreen(window,&origin) || origin.x!=display.bounds.left || origin.y!=display.bounds.top
            || client.right!=display.bounds.right-display.bounds.left
            || client.bottom!=display.bounds.bottom-display.bounds.top)return false;
        // Enumerate higher windows: even a narrow overlay must invalidate the
        // captured rectangle, rather than relying on a few sample points.
        RECT region{origin.x+crop.left,origin.y+crop.top,origin.x+crop.left+crop.width,origin.y+crop.top+crop.height};
        for(HWND above=GetWindow(window,GW_HWNDPREV);above;above=GetWindow(above,GW_HWNDPREV)) {
            if(!IsWindowVisible(above) || IsIconic(above))continue;
            BOOL cloaked=FALSE;DwmGetWindowAttribute(above,DWMWA_CLOAKED,&cloaked,sizeof(cloaked));
            if(cloaked)continue;
            RECT other{},intersection{};
            if(GetWindowRect(above,&other) && IntersectRect(&intersection,&region,&other))return false;
        }
        return true;
    }
private:
    HWND window_=nullptr;HDC dc_=nullptr;RECT bounds_{};
};
}
