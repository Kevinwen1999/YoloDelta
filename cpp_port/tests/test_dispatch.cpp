#include "delta/dispatch.hpp"
#include <iostream>
#include <stdexcept>
#if defined(_MSC_VER)
#include <crtdbg.h>
#include <cstdlib>
#endif

int main() {
#if defined(_MSC_VER)
    _CrtSetReportMode(_CRT_ASSERT,_CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT,_CRTDBG_FILE_STDERR);
    _set_abort_behavior(0,_WRITE_ABORT_MSG|_CALL_REPORTFAULT);
#endif
    try {
        delta::RuntimeConfig runtime;
        runtime.tracking_strategy=delta::TrackingStrategy::RawDelta;
        runtime.display_rate_servo_enable=true;
        const auto now=delta::SteadyClock::time_point(std::chrono::seconds(1));
        delta::DisplayRateServoState servo;
        servo.valid=true;servo.updated_at=now;servo.velocity_x=100;
        servo.observation_time=now;
        for(const double hz:{1.0,10.0,19.0,20.0}) {
            delta::AimCommandScheduler scheduler;
            const auto result=scheduler.step(now,{},std::nullopt,servo,runtime,hz,true,true);
            if(!result.servo || !result.command || result.command->dx!=5)
                throw std::runtime_error("Low-rate servo must cap integration to 50 ms without invalid bounds");
        }
        std::cout<<"Checked low-rate dispatch tests passed\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
