#pragma once
#if defined(_WIN32)
#include <windows.h>
#include <mmsystem.h>
#endif

namespace delta {
// Live control and visible simulation require the same Windows wait resolution.
class ScopedTimerResolution {
public:
    explicit ScopedTimerResolution(unsigned int period_ms):period_ms_(period_ms) {
#if defined(_WIN32)
        active_=timeBeginPeriod(period_ms_)==TIMERR_NOERROR;
#endif
    }
    ScopedTimerResolution(const ScopedTimerResolution&)=delete;
    ScopedTimerResolution& operator=(const ScopedTimerResolution&)=delete;
    ~ScopedTimerResolution() {
#if defined(_WIN32)
        if(active_)timeEndPeriod(period_ms_);
#endif
    }
    bool active()const{return active_;}
private:
    unsigned int period_ms_;
    bool active_=false;
};
}
