#include "delta/dispatch.hpp"
#include "delta/recoil_output.hpp"
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
        {
            delta::RuntimeConfig config;
            config.calibrated_control_enable = true;
            const auto click_counts = [](const delta::RuntimeConfig& cfg) {
                delta::LegacyRecoilOutput recoil;
                const auto start = delta::SteadyClock::time_point(std::chrono::seconds(1));
                int total = 0;
                for (int tick = 0; tick < 20; ++tick)
                    total += recoil.step(cfg, false, true, tick >= 5, false,
                        start + std::chrono::milliseconds(tick * 10));
                return total;
            };
            const int downward_counts = click_counts(config);
            auto no_compensation = config;
            no_compensation.recoil_compensation_y_px = 0;
            auto no_calibrated = config;
            no_calibrated.calibrated_control_enable = false;
            std::cout << "LMB recoil off: " << downward_counts
                << "; compensation zero: " << click_counts(no_compensation)
                << "; calibrated off: " << click_counts(no_calibrated) << '\n';
            if (downward_counts != 0)
                throw std::runtime_error("Left-click with recoil toggles off must not add downward movement");
            if (click_counts(no_calibrated) != 0)
                throw std::runtime_error("Disabled recoil must remain off in the legacy controller too");
            config.legacy_recoil_output_enable = true;
            if (click_counts(config) != 134)
                throw std::runtime_error("Explicit legacy recoil must retain its integrated rate");
            config.recoil_compensation_y_rate_px_s = 100;
            if (click_counts(config) != 14)
                throw std::runtime_error("Explicit recoil rate must override the legacy per-command value");
            delta::LegacyRecoilOutput recoil;
            const auto start = delta::SteadyClock::time_point(std::chrono::seconds(1));
            recoil.step(config, false, true, true, false, start);
            config.legacy_recoil_output_enable = false;
            if (recoil.step(config, false, true, true, false, start + std::chrono::milliseconds(10)) != 0)
                throw std::runtime_error("Turning recoil off during a hold must immediately clear its output");
            config.legacy_recoil_output_enable = true;
            if (recoil.step(config, false, true, true, false, start + std::chrono::milliseconds(100)) != 0)
                throw std::runtime_error("Reenabling recoil must not replay an old held interval");
            if (recoil.step(config, true, true, true, false, start + std::chrono::milliseconds(110)) != 0)
                throw std::runtime_error("Virtual recoil must suppress direct legacy recoil");
        }
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
        {
            delta::AimCommandScheduler scheduler;
            servo.expires_at=now+std::chrono::milliseconds(2);
            const auto result=scheduler.step(now,{},std::nullopt,servo,runtime,165,true,true);
            if(result.command)throw std::runtime_error("Prediction expiring before the next dispatch interval must not emit servo movement");
            delta::CommandPacket incoming;incoming.dx=10;incoming.target_detected=true;
            incoming.cmd_generated=now;incoming.expires_at=servo.expires_at;
            delta::AimCommandScheduler fresh;
            if(fresh.step(now,{},incoming,servo,runtime,165,true,true).command)
                throw std::runtime_error("Fresh commands must honor the same prediction delivery budget");
            incoming.expires_at=now+std::chrono::milliseconds(20);
            delta::AimCommandScheduler timely;
            if(!timely.step(now,{},incoming,servo,runtime,165,true,true).command)
                throw std::runtime_error("Prediction with sufficient delivery budget must remain eligible");
        }
        {
            delta::RuntimeConfig calibrated;
            calibrated.calibrated_control_enable=true;
            delta::AimCommandScheduler scheduler;
            delta::DisplayRateServoState rate;
            rate.valid=true;rate.updated_at=now;rate.observation_time=now;rate.velocity_x=1000;
            delta::CommandPacket incoming;incoming.dx=100;incoming.target_detected=true;incoming.cmd_generated=now;
            const auto first=scheduler.step(now,{},incoming,rate,calibrated,100,true,true);
            if(!first.command || first.command->dx!=10)
                throw std::runtime_error("Calibrated rate must integrate once, without adding a frame correction");
            calibrated.calibrated_pixels_per_count_x=.5F;
            if(scheduler.step(now+std::chrono::milliseconds(10),{},incoming,rate,calibrated,100,true,true).command)
                throw std::runtime_error("Calibration changes must invalidate the previous servo rate");
            rate.updated_at=now+std::chrono::milliseconds(20);
            if(!scheduler.step(rate.updated_at,{},std::nullopt,rate,calibrated,100,true,true).command)
                throw std::runtime_error("A new observation must resume the recalibrated rate");
        }
        std::cout<<"Checked low-rate dispatch tests passed\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
