#include "delta/frontend.hpp"
#include "delta/recoil.hpp"

#include <array>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

namespace delta {

namespace {

using json = nlohmann::json;

std::string jsonEscape(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char c : text) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += c; break;
        }
    }
    return out;
}

const char* engageButtonName(const LeftHoldEngageButton button) {
    switch (button) {
    case LeftHoldEngageButton::Left: return "leftkey";
    case LeftHoldEngageButton::X1: return "x1";
    case LeftHoldEngageButton::Both: return "both";
    case LeftHoldEngageButton::Right:
    default: return "rightkey";
    }
}

LeftHoldEngageButton parseEngageButton(const std::string& value) {
    if (value == "leftkey") return LeftHoldEngageButton::Left;
    if (value == "x1") return LeftHoldEngageButton::X1;
    if (value == "both") return LeftHoldEngageButton::Both;
    return LeftHoldEngageButton::Right;
}

bool parseBoolLiteral(const std::string& value) {
    return value == "true" || value == "1" || value == "TRUE";
}

bool pathMatches(std::string_view path, std::string_view route) {
    if (path == route) {
        return true;
    }
    if (const size_t scheme = path.find("://"); scheme != std::string_view::npos) {
        const size_t first_slash = path.find('/', scheme + 3);
        if (first_slash != std::string_view::npos) {
            return pathMatches(path.substr(first_slash), route);
        }
    }
    if (path.size() > route.size() && path.substr(0, route.size()) == route) {
        const char next = path[route.size()];
        return next == '?' || next == '#';
    }
    return false;
}

bool rootPathMatches(std::string_view path) {
    if (path == "/") {
        return true;
    }
    if (const size_t scheme = path.find("://"); scheme != std::string_view::npos) {
        const size_t first_slash = path.find('/', scheme + 3);
        return first_slash == std::string_view::npos || path.substr(first_slash) == "/";
    }
    return false;
}

std::optional<std::string> extractJsonString(const std::string& body, const char* key) {
    const std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*\"([^\"]*)\"", std::regex::icase);
    std::smatch match;
    if (std::regex_search(body, match, pattern) && match.size() > 1) {
        return match[1].str();
    }
    return std::nullopt;
}

std::optional<double> extractJsonNumber(const std::string& body, const char* key) {
    const std::regex pattern(
        std::string("\"") + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        std::regex::icase);
    std::smatch match;
    if (std::regex_search(body, match, pattern) && match.size() > 1) {
        return std::stod(match[1].str());
    }
    return std::nullopt;
}

std::optional<bool> extractJsonBool(const std::string& body, const char* key) {
    const std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*(true|false|1|0)", std::regex::icase);
    std::smatch match;
    if (std::regex_search(body, match, pattern) && match.size() > 1) {
        return parseBoolLiteral(match[1].str());
    }
    return std::nullopt;
}

json buildRecoilProfilesJson(const StaticConfig& config) {
    json profiles = json::array();
    for (const auto& profile : listRecoilProfiles(config)) {
        profiles.push_back(json{
            {"id", profile.id},
            {"name", profile.name},
            {"updated_at", profile.updated_at},
            {"shot_count", profile.shot_count},
            {"scale_factor", profile.scale_factor},
            {"horizontal_scale_factor", profile.horizontal_scale_factor},
            {"fire_interval_ms", profile.fire_interval_ms},
            {"valid", profile.valid},
            {"error", profile.error},
        });
    }
    return profiles;
}

json buildRecoilConfigJson(const RuntimeConfig& cfg) {
    return json{
        {"recoil_mode", recoilModeName(cfg.recoil_mode)},
        {"selected_recoil_profile_id", cfg.selected_recoil_profile_id},
    };
}

json buildRecoilStatusObject(const SharedState& shared_state) {
    std::lock_guard<std::mutex> lock(shared_state.mutex);
    return json{
        {"recoil_mode", recoilModeName(shared_state.recoil.mode)},
        {"recoil_enabled", shared_state.recoil.enabled},
        {"recoil_ignore_mode_check", shared_state.recoil.ignore_mode_check},
        {"recoil_mode_active", shared_state.recoil.mode_active},
        {"recoil_hold_engage_toggle", shared_state.recoil.hold_engage_toggle},
        {"recoil_left_pressed", shared_state.recoil.left_pressed},
        {"recoil_x1_pressed", shared_state.recoil.x1_pressed},
        {"recoil_trigger_pressed", shared_state.recoil.trigger_pressed},
        {"recoil_spray_active", shared_state.recoil.spray_active},
        {"selected_profile_id", shared_state.recoil.selected_profile_id},
        {"selected_profile_name", shared_state.recoil.selected_profile_name},
        {"recoil_profile_loaded", shared_state.recoil.profile_loaded},
        {"recoil_shot_index", shared_state.recoil.shot_index},
        {"recoil_shot_count", shared_state.recoil.shot_count},
        {"recoil_scale_factor", shared_state.recoil.scale_factor},
        {"recoil_horizontal_scale_factor", shared_state.recoil.horizontal_scale_factor},
        {"recoil_fire_interval_ms", shared_state.recoil.fire_interval_ms},
        {"recoil_scheduled_dx", shared_state.recoil.scheduled_dx},
        {"recoil_scheduled_dy", shared_state.recoil.scheduled_dy},
        {"recoil_last_applied_dx", shared_state.recoil.last_applied_dx},
        {"recoil_last_applied_dy", shared_state.recoil.last_applied_dy},
        {"recoil_last_applied_shot_index", shared_state.recoil.last_applied_shot_index},
        {"recoil_apply_count", shared_state.recoil.apply_count},
        {"recoil_debug_state", shared_state.recoil.debug_state},
        {"recoil_error", shared_state.recoil.error},
    };
}

std::string buildRecoilPayload(const StaticConfig& config, RuntimeConfigStore& store, SharedState& shared_state) {
    json payload{
        {"config", buildRecoilConfigJson(store.snapshot())},
        {"status", buildRecoilStatusObject(shared_state)},
        {"profiles", buildRecoilProfilesJson(config)},
    };
    return payload.dump();
}

std::string buildRecoilProfilesPayload(const StaticConfig& config) {
    return json{{"profiles", buildRecoilProfilesJson(config)}}.dump();
}

bool applyRecoilPatch(const std::string& body, RuntimeConfig& cfg, std::string& error) {
    error.clear();
    json payload = json::parse(body, nullptr, false);
    if (payload.is_discarded() || !payload.is_object()) {
        error = "Request body must be a JSON object.";
        return false;
    }

    if (const auto it = payload.find("recoil_mode"); it != payload.end()) {
        if (!it->is_string()) {
            error = "recoil_mode must be a string.";
            return false;
        }
        cfg.recoil_mode = parseRecoilMode(it->get<std::string>());
    }
    if (const auto it = payload.find("selected_recoil_profile_id"); it != payload.end()) {
        if (!it->is_string()) {
            error = "selected_recoil_profile_id must be a string.";
            return false;
        }
        cfg.selected_recoil_profile_id = it->get<std::string>();
    }
    return true;
}

std::string buildConfigJson(const RuntimeConfig& cfg, const std::uint64_t version, const std::uint64_t reset_token) {
    std::ostringstream oss;
    oss << "{"
        << "\"pid_enable\":" << (cfg.pid_enable ? "true" : "false") << ","
        << "\"tracking_enabled\":" << (cfg.tracking_enabled ? "true" : "false") << ","
        << "\"debug_preview_enable\":" << (cfg.debug_preview_enable ? "true" : "false") << ","
        << "\"debug_overlay_enable\":" << (cfg.debug_overlay_enable ? "true" : "false") << ","
        << "\"aim_mode\":\"" << aimModeName(cfg.aim_mode) << "\","
        << "\"capture_cached_timeout_ms\":" << cfg.capture_cached_timeout_ms << ","
        << "\"body_y_ratio\":" << cfg.body_y_ratio << ","
        << "\"head_y_ratio\":" << cfg.head_y_ratio << ","
        << "\"tracking_strategy\":\"" << trackingStrategyName(cfg.tracking_strategy) << "\","
        << "\"tracking_velocity_alpha\":" << cfg.tracking_velocity_alpha << ","
        << "\"kp\":" << cfg.kp << ","
        << "\"ki\":" << cfg.ki << ","
        << "\"kd\":" << cfg.kd << ","
        << "\"integral_limit\":" << cfg.integral_limit << ","
        << "\"anti_windup_gain\":" << cfg.anti_windup_gain << ","
        << "\"derivative_alpha\":" << cfg.derivative_alpha << ","
        << "\"output_limit\":" << cfg.output_limit << ","
        << "\"pid_settle_enable\":" << (cfg.pid_settle_enable ? "true" : "false") << ","
        << "\"pid_settle_error_px\":" << cfg.pid_settle_error_px << ","
        << "\"pid_settle_threshold_min_scale\":" << cfg.pid_settle_threshold_min_scale << ","
        << "\"pid_settle_threshold_max_scale\":" << cfg.pid_settle_threshold_max_scale << ","
        << "\"pid_settle_stable_frames\":" << cfg.pid_settle_stable_frames << ","
        << "\"pid_settle_error_delta_px\":" << cfg.pid_settle_error_delta_px << ","
        << "\"pid_settle_pre_output_scale\":" << cfg.pid_settle_pre_output_scale << ","
        << "\"legacy_pid_lock_error_px\":" << cfg.legacy_pid_lock_error_px << ","
        << "\"legacy_pid_speed_multiplier\":" << cfg.legacy_pid_speed_multiplier << ","
        << "\"legacy_pid_threshold_min_scale\":" << cfg.legacy_pid_threshold_min_scale << ","
        << "\"legacy_pid_threshold_max_scale\":" << cfg.legacy_pid_threshold_max_scale << ","
        << "\"legacy_pid_transition_sharpness\":" << cfg.legacy_pid_transition_sharpness << ","
        << "\"legacy_pid_transition_midpoint\":" << cfg.legacy_pid_transition_midpoint << ","
        << "\"legacy_pid_stable_frames\":" << cfg.legacy_pid_stable_frames << ","
        << "\"legacy_pid_error_delta_px\":" << cfg.legacy_pid_error_delta_px << ","
        << "\"legacy_pid_prelock_scale\":" << cfg.legacy_pid_prelock_scale << ","
        << "\"sticky_bias_px\":" << cfg.sticky_bias_px << ","
        << "\"target_guard_enable\":" << (cfg.target_guard_enable ? "true" : "false") << ","
        << "\"target_guard_commit_frames\":" << cfg.target_guard_commit_frames << ","
        << "\"target_guard_hold_frames\":" << cfg.target_guard_hold_frames << ","
        << "\"target_guard_window_scale\":" << cfg.target_guard_window_scale << ","
        << "\"target_guard_min_window_px\":" << cfg.target_guard_min_window_px << ","
        << "\"target_lead_enable\":" << (cfg.target_lead_enable ? "true" : "false") << ","
        << "\"target_lead_commit_frames\":" << cfg.target_lead_commit_frames << ","
        << "\"target_lead_auto_latency_enable\":" << (cfg.target_lead_auto_latency_enable ? "true" : "false") << ","
        << "\"target_lead_max_time_s\":" << cfg.target_lead_max_time_s << ","
        << "\"target_lead_min_speed_px_s\":" << cfg.target_lead_min_speed_px_s << ","
        << "\"target_lead_max_offset_box_scale\":" << cfg.target_lead_max_offset_box_scale << ","
        << "\"target_lead_smoothing_alpha\":" << cfg.target_lead_smoothing_alpha << ","
        << "\"prediction_time\":" << cfg.prediction_time << ","
        << "\"target_max_lost_frames\":" << cfg.target_max_lost_frames << ","
        << "\"model_conf\":" << cfg.model_conf << ","
        << "\"detection_min_conf\":" << cfg.detection_min_conf << ","
        << "\"ego_motion_comp_enable\":" << (cfg.ego_motion_comp_enable ? "true" : "false") << ","
        << "\"ego_motion_comp_gain_x\":" << cfg.ego_motion_comp_gain_x << ","
        << "\"ego_motion_comp_gain_y\":" << cfg.ego_motion_comp_gain_y << ","
        << "\"ego_motion_error_gate_enable\":" << (cfg.ego_motion_error_gate_enable ? "true" : "false") << ","
        << "\"ego_motion_error_gate_px\":" << cfg.ego_motion_error_gate_px << ","
        << "\"ego_motion_error_gate_normalize_by_box\":"
        << (cfg.ego_motion_error_gate_normalize_by_box ? "true" : "false") << ","
        << "\"ego_motion_error_gate_norm_threshold\":" << cfg.ego_motion_error_gate_norm_threshold << ","
        << "\"ego_motion_reset_on_switch\":" << (cfg.ego_motion_reset_on_switch ? "true" : "false") << ","
        << "\"recoil_mode\":\"" << recoilModeName(cfg.recoil_mode) << "\","
        << "\"selected_recoil_profile_id\":\"" << jsonEscape(cfg.selected_recoil_profile_id) << "\","
        << "\"triggerbot_enable\":" << (cfg.triggerbot_enable ? "true" : "false") << ","
        << "\"triggerbot_click_hold_s\":" << cfg.triggerbot_click_hold_s << ","
        << "\"triggerbot_click_cooldown_s\":" << cfg.triggerbot_click_cooldown_s << ","
        << "\"side_button_key_sequence_use_key3\":" << (cfg.side_button_key_sequence_use_key3 ? "true" : "false") << ","
        << "\"side_button_key_sequence_key3_press_time_ms\":" << cfg.side_button_key_sequence_key3_press_time_ms << ","
        << "\"side_button_key_sequence_use_key1\":" << (cfg.side_button_key_sequence_use_key1 ? "true" : "false") << ","
        << "\"side_button_key_sequence_key1_press_time_ms\":" << cfg.side_button_key_sequence_key1_press_time_ms << ","
        << "\"side_button_key_sequence_use_right_click\":" << (cfg.side_button_key_sequence_use_right_click ? "true" : "false") << ","
        << "\"side_button_key_sequence_right_click_hold_ms\":" << cfg.side_button_key_sequence_right_click_hold_ms << ","
        << "\"side_button_key_sequence_use_left_click\":" << (cfg.side_button_key_sequence_use_left_click ? "true" : "false") << ","
        << "\"side_button_key_sequence_left_click_hold_ms\":" << cfg.side_button_key_sequence_left_click_hold_ms << ","
        << "\"side_button_key_sequence_loop_delay_ms\":" << cfg.side_button_key_sequence_loop_delay_ms << ","
        << "\"recoil_compensation_y_rate_px_s\":" << cfg.recoil_compensation_y_rate_px_s << ","
        << "\"recoil_compensation_y_px\":" << cfg.recoil_compensation_y_px << ","
        << "\"left_hold_engage_button\":\"" << engageButtonName(cfg.left_hold_engage_button) << "\","
        << "\"recoil_tune_fallback_ignore_mode_check\":"
        << (cfg.recoil_tune_fallback_ignore_mode_check ? "true" : "false") << ","
        << "\"sendinput_gain_x\":" << cfg.sendinput_gain_x << ","
        << "\"sendinput_gain_y\":" << cfg.sendinput_gain_y << ","
        << "\"sendinput_max_step\":" << cfg.sendinput_max_step << ","
        << "\"raw_max_step_x\":" << cfg.raw_max_step_x << ","
        << "\"raw_max_step_y\":" << cfg.raw_max_step_y << ","
        << "\"version\":" << version << ","
        << "\"reset_token\":" << reset_token
        << "}";
    return oss.str();
}

std::string buildStatusJson(const RuntimeConfig& cfg, const SharedState& shared_state) {
    std::lock_guard<std::mutex> lock(shared_state.mutex);
    std::ostringstream oss;
    oss << "{"
        << "\"running\":" << (shared_state.running ? "true" : "false") << ","
        << "\"mode\":" << shared_state.toggles.mode << ","
        << "\"mode_active\":" << (shared_state.toggles.mode != 0 ? "true" : "false") << ","
        << "\"mode_label\":\"" << (shared_state.toggles.mode == 0 ? "OFF" : "ACTIVE") << "\","
        << "\"aimmode\":" << static_cast<int>(cfg.aim_mode) << ","
        << "\"aimmode_label\":\"" << aimModeLabel(cfg.aim_mode) << "\","
        << "\"aim_mode\":\"" << aimModeName(cfg.aim_mode) << "\","
        << "\"aim_mode_label\":\"" << aimModeLabel(cfg.aim_mode) << "\","
        << "\"side_button_key_sequence_enabled\":"
        << (shared_state.side_button_key_sequence_enabled ? "true" : "false") << ","
        << "\"left_hold_engage\":" << (shared_state.toggles.left_hold_engage ? "true" : "false") << ","
        << "\"recoil_tune_fallback\":" << (shared_state.toggles.recoil_tune_fallback ? "true" : "false") << ","
        << "\"triggerbot_enable\":" << (cfg.triggerbot_enable ? "true" : "false") << ","
        << "\"debug_preview_enable\":" << (cfg.debug_preview_enable ? "true" : "false") << ","
        << "\"debug_overlay_enable\":" << (cfg.debug_overlay_enable ? "true" : "false") << ","
        << "\"tracking_strategy\":\"" << jsonEscape(shared_state.tracking_strategy) << "\","
        << "\"target_found\":" << (shared_state.target_found ? "true" : "false") << ","
        << "\"target_speed\":" << shared_state.target_speed << ","
        << "\"pid_settled\":" << (shared_state.pid_settled ? "true" : "false") << ","
        << "\"pid_settle_error_metric_px\":" << shared_state.pid_settle_error_metric_px << ","
        << "\"pid_settle_threshold_px\":" << shared_state.pid_settle_threshold_px << ","
        << "\"lead_active\":" << (shared_state.lead_active ? "true" : "false") << ","
        << "\"lead_time_ms\":" << shared_state.lead_time_ms << ","
        << "\"target_cls\":" << shared_state.target_cls << ","
        << "\"aim_dx\":" << shared_state.aim_dx << ","
        << "\"aim_dy\":" << shared_state.aim_dy << ","
        << "\"recoil_mode\":\"" << recoilModeName(shared_state.recoil.mode) << "\","
        << "\"recoil_enabled\":" << (shared_state.recoil.enabled ? "true" : "false") << ","
        << "\"recoil_ignore_mode_check\":" << (shared_state.recoil.ignore_mode_check ? "true" : "false") << ","
        << "\"recoil_mode_active\":" << (shared_state.recoil.mode_active ? "true" : "false") << ","
        << "\"recoil_hold_engage_toggle\":" << (shared_state.recoil.hold_engage_toggle ? "true" : "false") << ","
        << "\"recoil_left_pressed\":" << (shared_state.recoil.left_pressed ? "true" : "false") << ","
        << "\"recoil_x1_pressed\":" << (shared_state.recoil.x1_pressed ? "true" : "false") << ","
        << "\"recoil_trigger_pressed\":" << (shared_state.recoil.trigger_pressed ? "true" : "false") << ","
        << "\"recoil_spray_active\":" << (shared_state.recoil.spray_active ? "true" : "false") << ","
        << "\"selected_profile_id\":\"" << jsonEscape(shared_state.recoil.selected_profile_id) << "\","
        << "\"selected_profile_name\":\"" << jsonEscape(shared_state.recoil.selected_profile_name) << "\","
        << "\"recoil_profile_loaded\":" << (shared_state.recoil.profile_loaded ? "true" : "false") << ","
        << "\"recoil_shot_index\":" << shared_state.recoil.shot_index << ","
        << "\"recoil_shot_count\":" << shared_state.recoil.shot_count << ","
        << "\"recoil_scale_factor\":" << shared_state.recoil.scale_factor << ","
        << "\"recoil_horizontal_scale_factor\":" << shared_state.recoil.horizontal_scale_factor << ","
        << "\"recoil_fire_interval_ms\":" << shared_state.recoil.fire_interval_ms << ","
        << "\"recoil_scheduled_dx\":" << shared_state.recoil.scheduled_dx << ","
        << "\"recoil_scheduled_dy\":" << shared_state.recoil.scheduled_dy << ","
        << "\"recoil_last_applied_dx\":" << shared_state.recoil.last_applied_dx << ","
        << "\"recoil_last_applied_dy\":" << shared_state.recoil.last_applied_dy << ","
        << "\"recoil_last_applied_shot_index\":" << shared_state.recoil.last_applied_shot_index << ","
        << "\"recoil_apply_count\":" << shared_state.recoil.apply_count << ","
        << "\"recoil_debug_state\":\"" << jsonEscape(shared_state.recoil.debug_state) << "\","
        << "\"recoil_error\":\"" << jsonEscape(shared_state.recoil.error) << "\""
        << "}";
    return oss.str();
}

std::string buildPageHtml() {
    return R"HTML(<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Delta Native Runtime</title>
  <style>
    :root { color-scheme: dark; --bg:#0f1518; --panel:#162126; --line:#2b3b42; --text:#eef6fa; --muted:#90a5af; --accent:#f0b35a; font-family:"Segoe UI",sans-serif; }
    body { margin:0; background:linear-gradient(135deg,#0b1114,#121c21 55%,#0d1418); color:var(--text); padding:20px; }
    .panel { max-width:960px; margin:0 auto; background:rgba(22,33,38,.94); border:1px solid var(--line); border-radius:20px; overflow:hidden; }
    .hero { padding:20px 22px 14px; border-bottom:1px solid var(--line); }
    .hero h1 { margin:0 0 6px; font-size:1.8rem; }
    .hero p { margin:0; color:var(--muted); }
    .status, form { padding:18px 22px 22px; }
    .bar { padding:12px 14px; border:1px solid var(--line); border-radius:14px; background:#11191d; color:var(--muted); margin-bottom:12px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }
    .field { border:1px solid var(--line); border-radius:14px; padding:12px; background:#10181c; display:grid; gap:8px; }
    .field label { color:var(--muted); font-size:.92rem; }
    .field input, .field select { width:100%; padding:10px 11px; border-radius:10px; border:1px solid #34464f; background:#0c1215; color:var(--text); }
    .toggle { display:flex; align-items:center; justify-content:space-between; }
    .actions { display:flex; flex-wrap:wrap; gap:10px; margin-top:16px; }
    button { border:0; border-radius:999px; padding:12px 16px; cursor:pointer; }
    .primary { background:linear-gradient(135deg,#f0b35a,#ff8a5b); color:#15100d; font-weight:700; }
    .secondary { background:#223038; color:var(--text); border:1px solid #34464f; }
  </style>
</head>
<body>
  <main class="panel">
    <section class="hero">
      <h1>Delta Native Runtime</h1>
      <p>Live tuning for the native tracking runtime, including raw/raw-delta and legacy PID control.</p>
    </section>
    <section class="status">
      <div id="runtime" class="bar">Connecting…</div>
      <div id="message" class="bar">Loading configuration…</div>
    </section>
    <form id="form">
      <div class="grid" id="grid"></div>
      <div class="actions">
        <button class="primary" type="submit">Apply Changes</button>
        <button class="secondary" type="button" id="reload">Reload</button>
        <button class="secondary" type="button" id="reset">Reset PID</button>
      </div>
    </form>
  </main>
  <script>
    const fields = [
      {key:"pid_enable",label:"PID Enabled",type:"bool"},
      {key:"tracking_enabled",label:"Tracking Enabled",type:"bool"},
      {key:"debug_preview_enable",label:"Debug Preview",type:"bool"},
      {key:"debug_overlay_enable",label:"Debug Overlay",type:"bool"},
      {key:"capture_cached_timeout_ms",label:"Cached Capture Timeout (ms)",type:"number",step:1,min:0},
      {key:"body_y_ratio",label:"Body Aim Y Ratio",type:"number",step:0.01,min:0,max:1},
      {key:"head_y_ratio",label:"Head Aim Y Ratio",type:"number",step:0.01,min:0,max:1},
      {key:"tracking_strategy",label:"Tracking Strategy",type:"select",options:[{value:"raw",label:"Raw Detection"},{value:"raw_delta",label:"Raw + Velocity"},{value:"legacy_pid",label:"Legacy PID"}]},
      {key:"tracking_velocity_alpha",label:"Velocity Beta",type:"number",step:0.001},
      {key:"kp",label:"Kp (X/Y)",type:"number",step:0.001},
      {key:"ki",label:"Ki (X/Y)",type:"number",step:0.001},
      {key:"kd",label:"Kd (X/Y)",type:"number",step:0.001},
      {key:"integral_limit",label:"Integral Limit",type:"number",step:1},
      {key:"anti_windup_gain",label:"Anti-Windup Gain",type:"number",step:0.001},
      {key:"derivative_alpha",label:"Derivative Alpha",type:"number",step:0.001},
      {key:"output_limit",label:"PID Output Limit",type:"number",step:1},
      {key:"pid_settle_enable",label:"PID Settle Gate",type:"bool"},
      {key:"pid_settle_error_px",label:"PID Settle Error (px)",type:"number",step:0.1,min:0},
      {key:"pid_settle_threshold_min_scale",label:"PID Settle Min Scale",type:"number",step:0.001,min:0},
      {key:"pid_settle_threshold_max_scale",label:"PID Settle Max Scale",type:"number",step:0.001,min:0},
      {key:"pid_settle_stable_frames",label:"PID Settle Stable Frames",type:"number",step:1,min:1},
      {key:"pid_settle_error_delta_px",label:"PID Settle Error Delta (px)",type:"number",step:0.1,min:0},
      {key:"pid_settle_pre_output_scale",label:"PID Pre-Settle Scale",type:"number",step:0.001,min:0,max:1},
      {key:"legacy_pid_lock_error_px",label:"Legacy PID Lock Error (px)",type:"number",step:0.1,min:0},
      {key:"legacy_pid_speed_multiplier",label:"Legacy PID Speed Multiplier",type:"number",step:0.001},
      {key:"legacy_pid_threshold_min_scale",label:"Legacy PID Min Scale",type:"number",step:0.001,min:0},
      {key:"legacy_pid_threshold_max_scale",label:"Legacy PID Max Scale",type:"number",step:0.001,min:0},
      {key:"legacy_pid_transition_sharpness",label:"Legacy PID Transition Sharpness",type:"number",step:0.001,min:0},
      {key:"legacy_pid_transition_midpoint",label:"Legacy PID Transition Midpoint",type:"number",step:0.001},
      {key:"legacy_pid_stable_frames",label:"Legacy PID Stable Frames",type:"number",step:1,min:1},
      {key:"legacy_pid_error_delta_px",label:"Legacy PID Error Delta (px)",type:"number",step:0.1,min:0},
      {key:"legacy_pid_prelock_scale",label:"Legacy PID Prelock Scale",type:"number",step:0.001,min:0,max:1},
      {key:"sticky_bias_px",label:"Sticky Bias (px)",type:"number",step:1},
      {key:"target_guard_enable",label:"Target Guard",type:"bool"},
      {key:"target_guard_commit_frames",label:"Target Guard Commit Frames",type:"number",step:1,min:1},
      {key:"target_guard_hold_frames",label:"Target Guard Hold Frames",type:"number",step:1,min:0},
      {key:"target_guard_window_scale",label:"Target Guard Window Scale",type:"number",step:0.01,min:0.1},
      {key:"target_guard_min_window_px",label:"Target Guard Min Window (px)",type:"number",step:1,min:1},
      {key:"prediction_time",label:"Lead Time (s)",type:"number",step:0.001},
      {key:"target_max_lost_frames",label:"Max Lost Frames",type:"number",step:1,min:1},
      {key:"model_conf",label:"Model Conf",type:"number",step:0.001,min:0,max:1},
      {key:"detection_min_conf",label:"Detection Min Conf",type:"number",step:0.001,min:0,max:1},
      {key:"ego_motion_comp_enable",label:"Ego Motion Comp",type:"bool"},
      {key:"ego_motion_comp_gain_x",label:"Ego Comp Gain X",type:"number",step:0.001},
      {key:"ego_motion_comp_gain_y",label:"Ego Comp Gain Y",type:"number",step:0.001},
      {key:"ego_motion_error_gate_enable",label:"Ego Error Gate",type:"bool"},
      {key:"ego_motion_error_gate_px",label:"Ego Gate Error (px)",type:"number",step:1,min:0},
      {key:"ego_motion_error_gate_normalize_by_box",label:"Ego Gate Normalize By Box",type:"bool"},
      {key:"ego_motion_error_gate_norm_threshold",label:"Ego Gate Norm Threshold",type:"number",step:0.01,min:0},
      {key:"ego_motion_reset_on_switch",label:"Reset Ego On Switch",type:"bool"},
      {key:"triggerbot_enable",label:"Trigger Bot",type:"bool"},
      {key:"triggerbot_click_hold_s",label:"Trigger Hold (s)",type:"number",step:0.001,min:0},
      {key:"triggerbot_click_cooldown_s",label:"Trigger Cooldown (s)",type:"number",step:0.001,min:0},
      {key:"side_button_key_sequence_use_right_click",label:"F5 X1 Step 1: Use Right Click",type:"bool"},
      {key:"side_button_key_sequence_right_click_hold_ms",label:"F5 X1 Step 1: Right Click Hold (ms)",type:"number",step:0.001,min:0},
      {key:"side_button_key_sequence_use_left_click",label:"F5 X1 Step 2: Use Left Click",type:"bool"},
      {key:"side_button_key_sequence_left_click_hold_ms",label:"F5 X1 Step 2: Left Click Hold (ms)",type:"number",step:0.001,min:0},
      {key:"side_button_key_sequence_use_key3",label:"F5 X1 Step 3: Use Key 3",type:"bool"},
      {key:"side_button_key_sequence_key3_press_time_ms",label:"F5 X1 Step 3: Key 3 Hold (ms)",type:"number",step:0.001,min:0},
      {key:"side_button_key_sequence_use_key1",label:"F5 X1 Step 4: Use Key 1",type:"bool"},
      {key:"side_button_key_sequence_key1_press_time_ms",label:"F5 X1 Step 4: Key 1 Hold (ms)",type:"number",step:0.001,min:0},
      {key:"side_button_key_sequence_loop_delay_ms",label:"F5 X1 Step 5: Wait Before Next Loop (ms)",type:"number",step:0.001,min:0},
      {key:"recoil_compensation_y_rate_px_s",label:"Recoil Comp Y Rate (px/s)",type:"number",step:1},
      {key:"recoil_compensation_y_px",label:"Legacy Recoil Comp Y (px/cmd)",type:"number",step:0.1},
      {key:"left_hold_engage_button",label:"F6 Engage Button",type:"select",options:[{value:"rightkey",label:"Right Key"},{value:"leftkey",label:"Left Key"},{value:"x1",label:"X1 Side Button"},{value:"both",label:"Left / Right / X1"}]},
      {key:"recoil_tune_fallback_ignore_mode_check",label:"F7 Ignore Mode Check",type:"bool"},
      {key:"sendinput_gain_x",label:"SendInput Gain X",type:"number",step:0.001},
      {key:"sendinput_gain_y",label:"SendInput Gain Y",type:"number",step:0.001},
      {key:"sendinput_max_step",label:"SendInput Max Step",type:"number",step:1,min:1},
      {key:"raw_max_step_y",label:"Raw Max Step Y",type:"number",step:1,min:1}
    ];
    const grid = document.getElementById("grid");
    const runtime = document.getElementById("runtime");
    const message = document.getElementById("message");
    const form = document.getElementById("form");
    function setMessage(text){ message.textContent = text; }
    function buildFields(){
      for(const field of fields){
        const wrap=document.createElement("div");
        wrap.className=field.type==="bool"?"field toggle":"field";
        const label=document.createElement("label");
        label.htmlFor=field.key; label.textContent=field.label; wrap.appendChild(label);
        let input;
        if(field.type==="select"){ input=document.createElement("select"); for(const opt of field.options){ const o=document.createElement("option"); o.value=opt.value; o.textContent=opt.label; input.appendChild(o);} }
        else {
          input=document.createElement("input");
          input.type=field.type==="bool"?"checkbox":"number";
          if(field.type==="number"){
            input.step=field.step ?? "any";
            if(field.min !== undefined) input.min=String(field.min);
            if(field.max !== undefined) input.max=String(field.max);
          }
        }
        input.id=field.key; wrap.appendChild(input); grid.appendChild(wrap);
      }
    }
    function applyConfig(cfg){
      for(const field of fields){
        const input=document.getElementById(field.key); if(!input) continue;
        if(field.type==="bool") input.checked=Boolean(cfg[field.key]);
        else input.value=cfg[field.key];
      }
    }
    function renderStatus(status){
      const settle=status.target_found ? `${status.pid_settled ? "settled" : "gating"} ${Number(status.pid_settle_error_metric_px||0).toFixed(1)}/${Number(status.pid_settle_threshold_px||0).toFixed(1)}` : "n/a";
      runtime.textContent=`Runtime ${status.running ? "running" : "stopped"} | mode ${status.mode_label} | aim ${status.aimmode_label} | preview ${status.debug_preview_enable ? "ON" : "OFF"} | overlay ${status.debug_overlay_enable ? "ON" : "OFF"} | F5 X1 custom sequence ${status.side_button_key_sequence_enabled ? "ON" : "OFF"} | track ${status.tracking_strategy} | target ${status.target_found ? "locked" : "none"} | speed ${Number(status.target_speed).toFixed(1)} | settle ${settle} | cmd (${status.aim_dx}, ${status.aim_dy})`;
    }
    function collectPayload(){
      const payload={};
      for(const field of fields){
        const input=document.getElementById(field.key); if(!input) continue;
        if(field.type==="bool") payload[field.key]=Boolean(input.checked);
        else if(field.type==="select") payload[field.key]=String(input.value);
        else payload[field.key]=Number(input.value);
      }
      return payload;
    }
    async function requestJson(path, options={}){
      const res=await fetch(path, options); const data=await res.json();
      if(!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      return data;
    }
    async function loadAll(){
      const payload=await requestJson("/api/pid");
      applyConfig(payload.config); renderStatus(payload.status); setMessage("Loaded runtime values.");
    }
    async function refreshStatus(){
      try { const payload=await requestJson("/api/pid/status"); renderStatus(payload.status); } catch(err) { runtime.textContent=err.message; }
    }
    form.addEventListener("submit", async (event)=>{ event.preventDefault(); try { const payload=await requestJson("/api/pid",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(collectPayload())}); applyConfig(payload.config); renderStatus(payload.status); setMessage("Applied runtime changes."); } catch(err) { setMessage(err.message); } });
    document.getElementById("reload").addEventListener("click", ()=>loadAll().catch(err=>setMessage(err.message)));
    document.getElementById("reset").addEventListener("click", async ()=>{ try { const payload=await requestJson("/api/pid/reset",{method:"POST"}); renderStatus(payload.status); setMessage("PID reset requested."); } catch(err) { setMessage(err.message); } });
    buildFields();
    loadAll().catch(err=>setMessage(err.message));
    window.setInterval(refreshStatus, 1000);
  </script>
</body>
</html>)HTML";
}

std::string buildRuntimePageHtml() {
    return std::string(R"HTML(<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Delta Runtime</title><style>
body{margin:0;padding:18px;background:#0f1518;color:#eef6fa;font:14px "Segoe UI",sans-serif}main{max-width:1100px;margin:0 auto}section{background:#162126;border:1px solid #2b3b42;border-radius:16px;padding:16px;margin:0 0 16px}h1,h2{margin:0 0 10px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}.field{display:grid;gap:6px}.toggle{display:flex;align-items:center;justify-content:space-between;gap:8px}.bar{padding:10px 12px;border:1px solid #2b3b42;border-radius:12px;background:#10181c;margin:0 0 10px;color:#c7d7de}.chip{padding:9px 10px;border:1px solid #34464f;border-radius:10px;background:#0c1215;color:#eef6fa}input,select,a,button{box-sizing:border-box}input,select,a{width:100%;padding:9px 10px;border-radius:10px;border:1px solid #34464f;background:#0c1215;color:#eef6fa;text-decoration:none}button{padding:10px 14px;border:0;border-radius:999px;cursor:pointer}button.primary{background:#f0b35a;color:#1e1408;font-weight:700}button.secondary{background:#223038;color:#eef6fa;border:1px solid #34464f}.actions{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px}
</style></head><body><main>
<section><h1>Delta Native Runtime</h1><div id="runtime" class="bar">Connecting...</div><div id="message" class="bar">Loading configuration...</div><div id="strategy-note" class="bar">Loading strategy note...</div></section>
<section><h2>Mode Toggles</h2><div class="grid"><div class="field"><label for="aim-mode">F4 Aim Mode</label><select id="aim-mode"><option value="head">Head</option><option value="body">Body</option><option value="hybrid">Hybrid</option></select></div><div class="field"><label>XBUTTON2 Mode</label><div id="toggle-mode" class="chip">Loading...</div></div><div class="field"><label>F5 X1 Sequence</label><div id="toggle-f5" class="chip">Loading...</div></div><div class="field"><label>F6 Left-Hold Engage</label><div id="toggle-f6" class="chip">Loading...</div></div><div class="field"><label>F7 Recoil Fallback</label><div id="toggle-f7" class="chip">Loading...</div></div><div class="field"><label>F8 TriggerBot</label><div id="toggle-f8" class="chip">Loading...</div></div></div></section>
<section><h2>Advanced Recoil</h2><div class="grid"><div class="field"><label for="recoil-mode">Recoil Mode</label><select id="recoil-mode"><option value="legacy">Legacy</option><option value="advanced_profile">Advanced Profile</option></select></div><div class="field"><label for="recoil-profile">Saved Profiles</label><select id="recoil-profile"></select></div><div class="field"><label for="recoil-editor-link">Profile Editor</label><a id="recoil-editor-link" href="http://127.0.0.1:8766/" target="_blank" rel="noreferrer">Open Python Recoil Editor</a></div></div><div id="recoil-status" class="bar">Loading recoil status...</div><div id="recoil-debug" class="bar">Loading recoil debug...</div><div class="actions"><button class="primary" type="button" id="recoil-apply">Apply Recoil Selection</button><button class="secondary" type="button" id="recoil-refresh">Refresh Recoil Profiles</button></div></section>
<section><h2>Runtime Tuning</h2><form id="form"><div class="grid" id="grid"></div><div class="actions"><button class="primary" type="submit">Apply Runtime Changes</button><button class="secondary" type="button" id="reload">Reload</button><button class="secondary" type="button" id="reset">Reset PID</button></div></form></section>
<script>
const F=[{k:"pid_enable",l:"PID Enabled",t:"b"},{k:"tracking_enabled",l:"Tracking Enabled",t:"b"},{k:"debug_preview_enable",l:"Debug Preview",t:"b"},{k:"debug_overlay_enable",l:"Debug Overlay",t:"b"},{k:"capture_cached_timeout_ms",l:"Cached Capture Timeout (ms)",t:"n",s:1,n:0},{k:"body_y_ratio",l:"Body Aim Y Ratio",t:"n",s:0.01,n:0,x:1},{k:"head_y_ratio",l:"Head Aim Y Ratio",t:"n",s:0.01,n:0,x:1},{k:"tracking_strategy",l:"Tracking Strategy",t:"s",o:[["raw","Raw Detection"],["raw_delta","Raw + Velocity"],["legacy_pid","Legacy PID"]]},{k:"tracking_velocity_alpha",l:"Velocity Beta",t:"n",s:0.001},{k:"kp",l:"Kp (X/Y)",t:"n",s:0.001},{k:"ki",l:"Ki (X/Y)",t:"n",s:0.001},{k:"kd",l:"Kd (X/Y)",t:"n",s:0.001},{k:"integral_limit",l:"Integral Limit",t:"n",s:1},{k:"anti_windup_gain",l:"Anti-Windup Gain",t:"n",s:0.001},{k:"derivative_alpha",l:"Derivative Alpha",t:"n",s:0.001},{k:"output_limit",l:"PID Output Limit",t:"n",s:1},{k:"pid_settle_enable",l:"PID Settle Gate",t:"b"},{k:"pid_settle_error_px",l:"PID Settle Error (px)",t:"n",s:0.1,n:0},{k:"pid_settle_threshold_min_scale",l:"PID Settle Min Scale",t:"n",s:0.001,n:0},{k:"pid_settle_threshold_max_scale",l:"PID Settle Max Scale",t:"n",s:0.001,n:0},{k:"pid_settle_stable_frames",l:"PID Settle Stable Frames",t:"n",s:1,n:1},{k:"pid_settle_error_delta_px",l:"PID Settle Error Delta (px)",t:"n",s:0.1,n:0},{k:"pid_settle_pre_output_scale",l:"PID Pre-Settle Scale",t:"n",s:0.001,n:0,x:1},{k:"legacy_pid_lock_error_px",l:"Legacy PID Lock Error (px)",t:"n",s:0.1,n:0},{k:"legacy_pid_speed_multiplier",l:"Legacy PID Speed Multiplier",t:"n",s:0.001},{k:"legacy_pid_threshold_min_scale",l:"Legacy PID Min Scale",t:"n",s:0.001,n:0},{k:"legacy_pid_threshold_max_scale",l:"Legacy PID Max Scale",t:"n",s:0.001,n:0},{k:"legacy_pid_transition_sharpness",l:"Legacy PID Transition Sharpness",t:"n",s:0.001,n:0},{k:"legacy_pid_transition_midpoint",l:"Legacy PID Transition Midpoint",t:"n",s:0.001},{k:"legacy_pid_stable_frames",l:"Legacy PID Stable Frames",t:"n",s:1,n:1},{k:"legacy_pid_error_delta_px",l:"Legacy PID Error Delta (px)",t:"n",s:0.1,n:0},{k:"legacy_pid_prelock_scale",l:"Legacy PID Prelock Scale",t:"n",s:0.001,n:0,x:1},{k:"sticky_bias_px",l:"Sticky Bias (px)",t:"n",s:1},{k:"target_guard_enable",l:"Target Guard",t:"b"},{k:"target_guard_commit_frames",l:"Target Guard Commit Frames",t:"n",s:1,n:1},{k:"target_guard_hold_frames",l:"Target Guard Hold Frames",t:"n",s:1,n:0},{k:"target_guard_window_scale",l:"Target Guard Window Scale",t:"n",s:0.01,n:0.1},{k:"target_guard_min_window_px",l:"Target Guard Min Window (px)",t:"n",s:1,n:1},{k:"target_lead_enable",l:"Target Lead",t:"b"},{k:"target_lead_commit_frames",l:"Target Lead Commit Frames",t:"n",s:1,n:1},{k:"target_lead_auto_latency_enable",l:"Target Lead Auto Latency",t:"b"},{k:"target_lead_max_time_s",l:"Target Lead Max Time (s)",t:"n",s:0.001,n:0},{k:"target_lead_min_speed_px_s",l:"Target Lead Min Speed (px/s)",t:"n",s:1,n:0},{k:"target_lead_max_offset_box_scale",l:"Target Lead Max Box Scale",t:"n",s:0.01,n:0},{k:"target_lead_smoothing_alpha",l:"Target Lead Smoothing",t:"n",s:0.001,n:0,x:1},{k:"prediction_time",l:"Target Lead Manual Bias (s)",t:"n",s:0.001},{k:"target_max_lost_frames",l:"Max Lost Frames",t:"n",s:1,n:1},{k:"model_conf",l:"Model Conf",t:"n",s:0.001,n:0,x:1},{k:"detection_min_conf",l:"Detection Min Conf",t:"n",s:0.001,n:0,x:1},{k:"ego_motion_comp_enable",l:"Ego Motion Comp",t:"b"},{k:"ego_motion_comp_gain_x",l:"Ego Comp Gain X",t:"n",s:0.001},{k:"ego_motion_comp_gain_y",l:"Ego Comp Gain Y",t:"n",s:0.001},{k:"ego_motion_error_gate_enable",l:"Ego Error Gate",t:"b"},{k:"ego_motion_error_gate_px",l:"Ego Gate Error (px)",t:"n",s:1,n:0},{k:"ego_motion_error_gate_normalize_by_box",l:"Ego Gate Normalize By Box",t:"b"},{k:"ego_motion_error_gate_norm_threshold",l:"Ego Gate Norm Threshold",t:"n",s:0.01,n:0},{k:"ego_motion_reset_on_switch",l:"Reset Ego On Switch",t:"b"},{k:"triggerbot_click_hold_s",l:"Trigger Hold (s)",t:"n",s:0.001,n:0},{k:"triggerbot_click_cooldown_s",l:"Trigger Cooldown (s)",t:"n",s:0.001,n:0},{k:"side_button_key_sequence_use_right_click",l:"F5 X1 Step 1: Use Right Click",t:"b"},{k:"side_button_key_sequence_right_click_hold_ms",l:"F5 X1 Step 1: Right Click Hold (ms)",t:"n",s:0.001,n:0},{k:"side_button_key_sequence_use_left_click",l:"F5 X1 Step 2: Use Left Click",t:"b"},{k:"side_button_key_sequence_left_click_hold_ms",l:"F5 X1 Step 2: Left Click Hold (ms)",t:"n",s:0.001,n:0},{k:"side_button_key_sequence_use_key3",l:"F5 X1 Step 3: Use Key 3",t:"b"},{k:"side_button_key_sequence_key3_press_time_ms",l:"F5 X1 Step 3: Key 3 Hold (ms)",t:"n",s:0.001,n:0},{k:"side_button_key_sequence_use_key1",l:"F5 X1 Step 4: Use Key 1",t:"b"},{k:"side_button_key_sequence_key1_press_time_ms",l:"F5 X1 Step 4: Key 1 Hold (ms)",t:"n",s:0.001,n:0},{k:"side_button_key_sequence_loop_delay_ms",l:"F5 X1 Step 5: Wait Before Next Loop (ms)",t:"n",s:0.001,n:0},{k:"recoil_compensation_y_rate_px_s",l:"Legacy Recoil Y Rate (px/s)",t:"n",s:1},{k:"recoil_compensation_y_px",l:"Legacy Recoil Y (px/cmd)",t:"n",s:0.1},{k:"left_hold_engage_button",l:"F6 Engage Button",t:"s",o:[["rightkey","Right Key"],["leftkey","Left Key"],["x1","X1 Side Button"],["both","Left / Right / X1"]]},{k:"recoil_tune_fallback_ignore_mode_check",l:"F7 Ignore Mode Check",t:"b"},{k:"sendinput_gain_x",l:"SendInput Gain X",t:"n",s:0.001},{k:"sendinput_gain_y",l:"SendInput Gain Y",t:"n",s:0.001},{k:"sendinput_max_step",l:"SendInput Max Step",t:"n",s:1,n:1},{k:"raw_max_step_x",l:"Raw Max Step X",t:"n",s:1,n:1},{k:"raw_max_step_y",l:"Raw Max Step Y",t:"n",s:1,n:1}];
)HTML")
        + R"HTML(const G=id=>document.getElementById(id),grid=G("grid"),runtime=G("runtime"),recoilStatus=G("recoil-status"),recoilDebug=G("recoil-debug"),recoilMode=G("recoil-mode"),recoilProfile=G("recoil-profile"),message=G("message"),strategyNote=G("strategy-note"),form=G("form"),aimMode=G("aim-mode");
const toggleMode=G("toggle-mode"),toggleF5=G("toggle-f5"),toggleF6=G("toggle-f6"),toggleF7=G("toggle-f7"),toggleF8=G("toggle-f8");
const setMessage=t=>message.textContent=t;
function renderStrategyNote(strategy){const mode=String(strategy||"raw_delta");strategyNote.textContent=mode==="legacy_pid"?"Legacy PID keeps its dedicated controller path. Target Lead still applies when enabled, but Legacy PID ignores Velocity Beta, ego-motion compensation, anti-windup, derivative smoothing, output limit, and PID settle knobs.":"Raw/raw_delta use the modern PID + feedforward path, and Target Lead can add lock-gated aim-ahead on top.";}
function buildFields(){for(const f of F){const w=document.createElement("div");w.className=f.t==="b"?"field toggle":"field";const l=document.createElement("label");l.htmlFor=f.k;l.textContent=f.l;w.appendChild(l);let i;if(f.t==="s"){i=document.createElement("select");for(const [v,t] of f.o){const o=document.createElement("option");o.value=v;o.textContent=t;i.appendChild(o);}}else{i=document.createElement("input");i.type=f.t==="b"?"checkbox":"number";if(f.t==="n"){i.step=f.s??"any";if(f.n!==undefined)i.min=String(f.n);if(f.x!==undefined)i.max=String(f.x);}}i.id=f.k;w.appendChild(i);grid.appendChild(w);}}
function applyConfig(cfg){for(const f of F){const i=G(f.k);if(!i)continue;if(f.t==="b")i.checked=Boolean(cfg[f.k]);else if(cfg[f.k]!==undefined)i.value=cfg[f.k];}if(cfg.aim_mode!==undefined)aimMode.value=String(cfg.aim_mode);renderStrategyNote(cfg.tracking_strategy);}
function applyRecoilConfig(cfg){recoilMode.value=String(cfg.recoil_mode??"legacy");recoilProfile.value=String(cfg.selected_recoil_profile_id??"");}
function renderToggleChip(node,label,on){node.textContent=`${label}: ${on?"ON":"OFF"}`;}
function renderModeToggles(s){aimMode.value=String(s.aim_mode??"head");toggleMode.textContent=`${s.mode_label||"OFF"}`;toggleF5.textContent=`${s.side_button_key_sequence_enabled?"ON":"OFF"}`;toggleF6.textContent=`${s.left_hold_engage?"ON":"OFF"}`;toggleF7.textContent=`${s.recoil_tune_fallback?"ON":"OFF"}`;toggleF8.textContent=`${s.triggerbot_enable?"ON":"OFF"}`;}
function renderRecoilStatus(s){const loaded=s.recoil_profile_loaded?"loaded":"not loaded";const name=s.selected_profile_name||s.selected_profile_id||"none";const err=s.recoil_error?` | error ${s.recoil_error}`:"";const hScale=s.recoil_horizontal_scale_factor ?? s.recoil_scale_factor ?? 0;recoilStatus.textContent=`F7 ${s.recoil_enabled?"ON":"OFF"} | mode ${s.recoil_mode} | profile ${name} (${loaded}) | shot ${s.recoil_shot_index}/${s.recoil_shot_count} | scale V/H ${Number(s.recoil_scale_factor||0).toFixed(3)} / ${Number(hScale).toFixed(3)} | interval ${s.recoil_fire_interval_ms||0}ms${err}`;recoilDebug.textContent=`debug ${s.recoil_debug_state||"idle"} | mode-toggle ${s.recoil_mode_active?"ON":"OFF"} | F6 ${s.recoil_hold_engage_toggle?"ON":"OFF"} | ignore-mode ${s.recoil_ignore_mode_check?"ON":"OFF"} | trigger ${s.recoil_trigger_pressed?"DOWN":"UP"} | left ${s.recoil_left_pressed?"DOWN":"UP"} | x1 ${s.recoil_x1_pressed?"DOWN":"UP"} | spray ${s.recoil_spray_active?"ACTIVE":"IDLE"} | scheduled (${s.recoil_scheduled_dx||0}, ${s.recoil_scheduled_dy||0}) | last applied (${s.recoil_last_applied_dx||0}, ${s.recoil_last_applied_dy||0}) @ shot ${s.recoil_last_applied_shot_index||0} | applied count ${s.recoil_apply_count||0}`;}
function renderStatus(s){const settle=s.target_found?`${s.pid_settled?"settled":"gating"} ${Number(s.pid_settle_error_metric_px||0).toFixed(1)}/${Number(s.pid_settle_threshold_px||0).toFixed(1)}`:"n/a";const lead=s.lead_active?`lead ${Number(s.lead_time_ms||0).toFixed(1)}ms`:"lead off";runtime.textContent=`Runtime ${s.running?"running":"stopped"} | mode ${s.mode_label} | aim ${s.aim_mode_label||s.aimmode_label} | preview ${s.debug_preview_enable?"ON":"OFF"} | overlay ${s.debug_overlay_enable?"ON":"OFF"} | F8 ${s.triggerbot_enable?"ON":"OFF"} | recoil ${s.recoil_mode} | profile ${s.selected_profile_name||s.selected_profile_id||"none"} | target ${s.target_found?"locked":"none"} | speed ${Number(s.target_speed).toFixed(1)} | ${lead} | settle ${settle} | cmd (${s.aim_dx}, ${s.aim_dy})`;renderStrategyNote(s.tracking_strategy);renderModeToggles(s);renderRecoilStatus(s);}
function collectPayload(){const out={};for(const f of F){const i=G(f.k);if(!i)continue;out[f.k]=f.t==="b"?Boolean(i.checked):f.t==="s"?String(i.value):Number(i.value);}return out;}
const collectRecoilPayload=()=>({recoil_mode:String(recoilMode.value),selected_recoil_profile_id:String(recoilProfile.value||"")});
function renderProfiles(profiles,selectedId){const want=String(selectedId??recoilProfile.value??"");recoilProfile.innerHTML="";const empty=document.createElement("option");empty.value="";empty.textContent="No profile selected";recoilProfile.appendChild(empty);for(const p of profiles||[]){const o=document.createElement("option");o.value=String(p.id);o.textContent=p.valid?`${p.name} (${p.shot_count} shots)`:`${p.name} [invalid]`;o.disabled=!p.valid;recoilProfile.appendChild(o);}recoilProfile.value=want;if(recoilProfile.value!==want)recoilProfile.value="";}
async function requestJson(path,options={}){const res=await fetch(path,options);const data=await res.json();if(!res.ok)throw new Error(data.error||`HTTP ${res.status}`);return data;}
async function loadRecoil(){const p=await requestJson("/api/recoil");applyRecoilConfig(p.config);renderProfiles(p.profiles,p.config.selected_recoil_profile_id);renderRecoilStatus(p.status);}
async function loadAll(){const p=await requestJson("/api/pid");applyConfig(p.config);renderStatus(p.status);await loadRecoil();setMessage("Loaded runtime values.");}
)HTML"
        + R"HTML(async function refreshStatus(){try{renderStatus((await requestJson("/api/pid/status")).status);}catch(err){runtime.textContent=err.message;}}
form.addEventListener("submit",async e=>{e.preventDefault();try{const p=await requestJson("/api/pid",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(collectPayload())});applyConfig(p.config);renderStatus(p.status);setMessage("Applied runtime changes.");}catch(err){setMessage(err.message);}});
aimMode.addEventListener("change",async()=>{try{const p=await requestJson("/api/pid",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({aim_mode:String(aimMode.value)})});applyConfig(p.config);renderStatus(p.status);setMessage("Applied aim mode.");}catch(err){setMessage(err.message);}});
G("reload").addEventListener("click",()=>loadAll().catch(err=>setMessage(err.message)));
G("reset").addEventListener("click",async()=>{try{renderStatus((await requestJson("/api/pid/reset",{method:"POST"})).status);setMessage("PID reset requested.");}catch(err){setMessage(err.message);}});
G("recoil-apply").addEventListener("click",async()=>{try{const p=await requestJson("/api/recoil",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(collectRecoilPayload())});applyRecoilConfig(p.config);renderProfiles(p.profiles,p.config.selected_recoil_profile_id);renderRecoilStatus(p.status);setMessage("Applied recoil profile selection.");}catch(err){setMessage(err.message);}});
G("recoil-refresh").addEventListener("click",async()=>{try{renderProfiles((await requestJson("/api/recoil/profiles")).profiles,recoilProfile.value);setMessage("Reloaded recoil profiles.");}catch(err){setMessage(err.message);}});
buildFields();loadAll().catch(err=>setMessage(err.message));window.setInterval(refreshStatus,1000);
</script></main></body></html>)HTML";
}

bool applyRuntimePatch(const std::string& body, RuntimeConfig& cfg, std::string& error) {
    if (const auto value = extractJsonBool(body, "pid_enable"); value.has_value()) cfg.pid_enable = *value;
    if (const auto value = extractJsonBool(body, "tracking_enabled"); value.has_value()) cfg.tracking_enabled = *value;
    if (const auto value = extractJsonBool(body, "debug_preview_enable"); value.has_value()) cfg.debug_preview_enable = *value;
    if (const auto value = extractJsonBool(body, "debug_overlay_enable"); value.has_value()) cfg.debug_overlay_enable = *value;
    if (const auto value = extractJsonString(body, "aim_mode"); value.has_value()) cfg.aim_mode = parseAimMode(*value);
    if (const auto value = extractJsonNumber(body, "capture_cached_timeout_ms"); value.has_value()) {
        cfg.capture_cached_timeout_ms = std::max(0, static_cast<int>(std::lround(*value)));
    }
    if (const auto value = extractJsonNumber(body, "body_y_ratio"); value.has_value()) cfg.body_y_ratio = clamp(static_cast<float>(*value), 0.0F, 1.0F);
    if (const auto value = extractJsonNumber(body, "head_y_ratio"); value.has_value()) cfg.head_y_ratio = clamp(static_cast<float>(*value), 0.0F, 1.0F);
    if (const auto value = extractJsonString(body, "tracking_strategy"); value.has_value()) cfg.tracking_strategy = parseTrackingStrategy(*value);
    if (const auto value = extractJsonNumber(body, "tracking_velocity_alpha"); value.has_value()) cfg.tracking_velocity_alpha = clamp(static_cast<float>(*value), 0.0F, 1.0F);
    if (const auto value = extractJsonNumber(body, "kp"); value.has_value()) cfg.kp = static_cast<float>(*value);
    if (const auto value = extractJsonNumber(body, "ki"); value.has_value()) cfg.ki = static_cast<float>(*value);
    if (const auto value = extractJsonNumber(body, "kd"); value.has_value()) cfg.kd = static_cast<float>(*value);
    if (const auto value = extractJsonNumber(body, "integral_limit"); value.has_value()) cfg.integral_limit = std::max(0.0F, static_cast<float>(*value));
    if (const auto value = extractJsonNumber(body, "anti_windup_gain"); value.has_value()) cfg.anti_windup_gain = clamp(static_cast<float>(*value), 0.0F, 1.0F);
    if (const auto value = extractJsonNumber(body, "derivative_alpha"); value.has_value()) cfg.derivative_alpha = clamp(static_cast<float>(*value), 0.0F, 1.0F);
    if (const auto value = extractJsonNumber(body, "output_limit"); value.has_value()) cfg.output_limit = std::max(0.0F, static_cast<float>(*value));
    if (const auto value = extractJsonBool(body, "pid_settle_enable"); value.has_value()) cfg.pid_settle_enable = *value;
    if (const auto value = extractJsonNumber(body, "pid_settle_error_px"); value.has_value()) cfg.pid_settle_error_px = std::max(0.0F, static_cast<float>(*value));
    if (const auto value = extractJsonNumber(body, "pid_settle_threshold_min_scale"); value.has_value()) {
        cfg.pid_settle_threshold_min_scale = std::max(0.0F, static_cast<float>(*value));
    }
    if (const auto value = extractJsonNumber(body, "pid_settle_threshold_max_scale"); value.has_value()) {
        cfg.pid_settle_threshold_max_scale = std::max(cfg.pid_settle_threshold_min_scale, static_cast<float>(*value));
    }
    if (const auto value = extractJsonNumber(body, "pid_settle_stable_frames"); value.has_value()) {
        cfg.pid_settle_stable_frames = std::max(1, static_cast<int>(std::lround(*value)));
    }
    if (const auto value = extractJsonNumber(body, "pid_settle_error_delta_px"); value.has_value()) {
        cfg.pid_settle_error_delta_px = std::max(0.0F, static_cast<float>(*value));
    }
    if (const auto value = extractJsonNumber(body, "pid_settle_pre_output_scale"); value.has_value()) {
        cfg.pid_settle_pre_output_scale = clamp(static_cast<float>(*value), 0.0F, 1.0F);
    }
    cfg.pid_settle_threshold_max_scale = std::max(cfg.pid_settle_threshold_max_scale, cfg.pid_settle_threshold_min_scale);
    if (const auto value = extractJsonNumber(body, "legacy_pid_lock_error_px"); value.has_value()) {
        cfg.legacy_pid_lock_error_px = std::max(0.0F, static_cast<float>(*value));
    }
    if (const auto value = extractJsonNumber(body, "legacy_pid_speed_multiplier"); value.has_value()) {
        cfg.legacy_pid_speed_multiplier = static_cast<float>(*value);
    }
    if (const auto value = extractJsonNumber(body, "legacy_pid_threshold_min_scale"); value.has_value()) {
        cfg.legacy_pid_threshold_min_scale = std::max(0.0F, static_cast<float>(*value));
    }
    if (const auto value = extractJsonNumber(body, "legacy_pid_threshold_max_scale"); value.has_value()) {
        cfg.legacy_pid_threshold_max_scale = std::max(cfg.legacy_pid_threshold_min_scale, static_cast<float>(*value));
    }
    if (const auto value = extractJsonNumber(body, "legacy_pid_transition_sharpness"); value.has_value()) {
        cfg.legacy_pid_transition_sharpness = std::max(0.0F, static_cast<float>(*value));
    }
    if (const auto value = extractJsonNumber(body, "legacy_pid_transition_midpoint"); value.has_value()) {
        cfg.legacy_pid_transition_midpoint = static_cast<float>(*value);
    }
    if (const auto value = extractJsonNumber(body, "legacy_pid_stable_frames"); value.has_value()) {
        cfg.legacy_pid_stable_frames = std::max(1, static_cast<int>(std::lround(*value)));
    }
    if (const auto value = extractJsonNumber(body, "legacy_pid_error_delta_px"); value.has_value()) {
        cfg.legacy_pid_error_delta_px = std::max(0.0F, static_cast<float>(*value));
    }
    if (const auto value = extractJsonNumber(body, "legacy_pid_prelock_scale"); value.has_value()) {
        cfg.legacy_pid_prelock_scale = clamp(static_cast<float>(*value), 0.0F, 1.0F);
    }
    cfg.legacy_pid_threshold_max_scale = std::max(cfg.legacy_pid_threshold_max_scale, cfg.legacy_pid_threshold_min_scale);
    if (const auto value = extractJsonNumber(body, "sticky_bias_px"); value.has_value()) cfg.sticky_bias_px = std::max(0.0F, static_cast<float>(*value));
    if (const auto value = extractJsonBool(body, "target_guard_enable"); value.has_value()) cfg.target_guard_enable = *value;
    if (const auto value = extractJsonNumber(body, "target_guard_commit_frames"); value.has_value()) {
        cfg.target_guard_commit_frames = std::max(1, static_cast<int>(std::lround(*value)));
    }
    if (const auto value = extractJsonNumber(body, "target_guard_hold_frames"); value.has_value()) {
        cfg.target_guard_hold_frames = std::max(0, static_cast<int>(std::lround(*value)));
    }
    if (const auto value = extractJsonNumber(body, "target_guard_window_scale"); value.has_value()) {
        cfg.target_guard_window_scale = std::max(0.1F, static_cast<float>(*value));
    }
    if (const auto value = extractJsonNumber(body, "target_guard_min_window_px"); value.has_value()) {
        cfg.target_guard_min_window_px = std::max(1, static_cast<int>(std::lround(*value)));
    }
    if (const auto value = extractJsonBool(body, "target_lead_enable"); value.has_value()) cfg.target_lead_enable = *value;
    if (const auto value = extractJsonNumber(body, "target_lead_commit_frames"); value.has_value()) {
        cfg.target_lead_commit_frames = std::max(1, static_cast<int>(std::lround(*value)));
    }
    if (const auto value = extractJsonBool(body, "target_lead_auto_latency_enable"); value.has_value()) {
        cfg.target_lead_auto_latency_enable = *value;
    }
    if (const auto value = extractJsonNumber(body, "target_lead_max_time_s"); value.has_value()) {
        cfg.target_lead_max_time_s = std::max(0.0F, static_cast<float>(*value));
    }
    if (const auto value = extractJsonNumber(body, "target_lead_min_speed_px_s"); value.has_value()) {
        cfg.target_lead_min_speed_px_s = std::max(0.0F, static_cast<float>(*value));
    }
    if (const auto value = extractJsonNumber(body, "target_lead_max_offset_box_scale"); value.has_value()) {
        cfg.target_lead_max_offset_box_scale = std::max(0.0F, static_cast<float>(*value));
    }
    if (const auto value = extractJsonNumber(body, "target_lead_smoothing_alpha"); value.has_value()) {
        cfg.target_lead_smoothing_alpha = clamp(static_cast<float>(*value), 0.0F, 1.0F);
    }
    if (const auto value = extractJsonNumber(body, "prediction_time"); value.has_value()) cfg.prediction_time = std::max(0.0F, static_cast<float>(*value));
    if (const auto value = extractJsonNumber(body, "target_max_lost_frames"); value.has_value()) cfg.target_max_lost_frames = std::max(1, static_cast<int>(std::lround(*value)));
    if (const auto value = extractJsonNumber(body, "model_conf"); value.has_value()) cfg.model_conf = clamp(static_cast<float>(*value), 0.0F, 1.0F);
    if (const auto value = extractJsonNumber(body, "detection_min_conf"); value.has_value()) cfg.detection_min_conf = clamp(static_cast<float>(*value), 0.0F, 1.0F);
    if (const auto value = extractJsonBool(body, "ego_motion_comp_enable"); value.has_value()) cfg.ego_motion_comp_enable = *value;
    if (const auto value = extractJsonNumber(body, "ego_motion_comp_gain_x"); value.has_value()) cfg.ego_motion_comp_gain_x = static_cast<float>(*value);
    if (const auto value = extractJsonNumber(body, "ego_motion_comp_gain_y"); value.has_value()) cfg.ego_motion_comp_gain_y = static_cast<float>(*value);
    if (const auto value = extractJsonBool(body, "ego_motion_error_gate_enable"); value.has_value()) cfg.ego_motion_error_gate_enable = *value;
    if (const auto value = extractJsonNumber(body, "ego_motion_error_gate_px"); value.has_value()) cfg.ego_motion_error_gate_px = std::max(0.0F, static_cast<float>(*value));
    if (const auto value = extractJsonBool(body, "ego_motion_error_gate_normalize_by_box"); value.has_value()) cfg.ego_motion_error_gate_normalize_by_box = *value;
    if (const auto value = extractJsonNumber(body, "ego_motion_error_gate_norm_threshold"); value.has_value()) cfg.ego_motion_error_gate_norm_threshold = std::max(0.0F, static_cast<float>(*value));
    if (const auto value = extractJsonBool(body, "ego_motion_reset_on_switch"); value.has_value()) cfg.ego_motion_reset_on_switch = *value;
    if (const auto value = extractJsonString(body, "recoil_mode"); value.has_value()) cfg.recoil_mode = parseRecoilMode(*value);
    if (const auto value = extractJsonString(body, "selected_recoil_profile_id"); value.has_value()) cfg.selected_recoil_profile_id = *value;
    if (const auto value = extractJsonBool(body, "triggerbot_enable"); value.has_value()) cfg.triggerbot_enable = *value;
    if (const auto value = extractJsonNumber(body, "triggerbot_click_hold_s"); value.has_value()) cfg.triggerbot_click_hold_s = std::max(0.0F, static_cast<float>(*value));
    if (const auto value = extractJsonNumber(body, "triggerbot_click_cooldown_s"); value.has_value()) cfg.triggerbot_click_cooldown_s = std::max(0.0F, static_cast<float>(*value));
    if (const auto value = extractJsonBool(body, "side_button_key_sequence_use_key3"); value.has_value()) {
        cfg.side_button_key_sequence_use_key3 = *value;
    }
    if (const auto value = extractJsonNumber(body, "side_button_key_sequence_key3_press_time_ms"); value.has_value()) {
        cfg.side_button_key_sequence_key3_press_time_ms = std::max(0.0, *value);
    }
    if (const auto value = extractJsonBool(body, "side_button_key_sequence_use_key1"); value.has_value()) {
        cfg.side_button_key_sequence_use_key1 = *value;
    }
    if (const auto value = extractJsonNumber(body, "side_button_key_sequence_key1_press_time_ms"); value.has_value()) {
        cfg.side_button_key_sequence_key1_press_time_ms = std::max(0.0, *value);
    }
    if (const auto value = extractJsonBool(body, "side_button_key_sequence_use_right_click"); value.has_value()) {
        cfg.side_button_key_sequence_use_right_click = *value;
    }
    if (const auto value = extractJsonNumber(body, "side_button_key_sequence_right_click_hold_ms"); value.has_value()) {
        cfg.side_button_key_sequence_right_click_hold_ms = std::max(0.0, *value);
    }
    if (const auto value = extractJsonBool(body, "side_button_key_sequence_use_left_click"); value.has_value()) {
        cfg.side_button_key_sequence_use_left_click = *value;
    }
    if (const auto value = extractJsonNumber(body, "side_button_key_sequence_left_click_hold_ms"); value.has_value()) {
        cfg.side_button_key_sequence_left_click_hold_ms = std::max(0.0, *value);
    }
    if (const auto value = extractJsonNumber(body, "side_button_key_sequence_loop_delay_ms"); value.has_value()) {
        cfg.side_button_key_sequence_loop_delay_ms = std::max(0.0, *value);
    }
    if (const auto value = extractJsonNumber(body, "recoil_compensation_y_rate_px_s"); value.has_value()) cfg.recoil_compensation_y_rate_px_s = static_cast<float>(*value);
    if (const auto value = extractJsonNumber(body, "recoil_compensation_y_px"); value.has_value()) cfg.recoil_compensation_y_px = static_cast<float>(*value);
    if (const auto value = extractJsonString(body, "left_hold_engage_button"); value.has_value()) cfg.left_hold_engage_button = parseEngageButton(*value);
    if (const auto value = extractJsonBool(body, "recoil_tune_fallback_ignore_mode_check"); value.has_value()) cfg.recoil_tune_fallback_ignore_mode_check = *value;
    if (const auto value = extractJsonNumber(body, "sendinput_gain_x"); value.has_value()) cfg.sendinput_gain_x = static_cast<float>(*value);
    if (const auto value = extractJsonNumber(body, "sendinput_gain_y"); value.has_value()) cfg.sendinput_gain_y = static_cast<float>(*value);
    if (const auto value = extractJsonNumber(body, "sendinput_max_step"); value.has_value()) cfg.sendinput_max_step = std::max(1, static_cast<int>(std::lround(*value)));
    if (const auto value = extractJsonNumber(body, "raw_max_step_x"); value.has_value()) cfg.raw_max_step_x = std::max(1, static_cast<int>(std::lround(*value)));
    if (const auto value = extractJsonNumber(body, "raw_max_step_y"); value.has_value()) cfg.raw_max_step_y = std::max(1, static_cast<int>(std::lround(*value)));
    if (body.find('{') == std::string::npos) {
        error = "Request body must be a JSON object.";
        return false;
    }
    return true;
}

std::string buildApiPayload(RuntimeConfigStore& store, SharedState& shared_state) {
    const RuntimeConfig cfg = store.snapshot();
    std::ostringstream oss;
    oss << "{"
        << "\"config\":" << buildConfigJson(cfg, store.version(), store.resetToken()) << ","
        << "\"status\":" << buildStatusJson(cfg, shared_state)
        << "}";
    return oss.str();
}

std::string httpResponse(const std::string& body, const char* content_type, const int status = 200, const char* status_text = "OK") {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << ' ' << status_text << "\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Cache-Control: no-store\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    return oss.str();
}

std::string readRequest(SOCKET client) {
    std::string request;
    std::array<char, 4096> buffer{};
    int content_length = 0;
    bool header_complete = false;
    while (true) {
        const int received = recv(client, buffer.data(), static_cast<int>(buffer.size()), 0);
        if (received <= 0) {
            break;
        }
        request.append(buffer.data(), static_cast<size_t>(received));
        const size_t header_end = request.find("\r\n\r\n");
        if (!header_complete && header_end != std::string::npos) {
            header_complete = true;
            const std::regex content_length_pattern(R"(Content-Length:\s*([0-9]+))", std::regex::icase);
            std::smatch match;
            if (std::regex_search(request, match, content_length_pattern) && match.size() > 1) {
                content_length = std::stoi(match[1].str());
            }
            if (request.size() >= header_end + 4 + static_cast<size_t>(content_length)) {
                break;
            }
        } else if (header_complete && request.size() >= request.find("\r\n\r\n") + 4 + static_cast<size_t>(content_length)) {
            break;
        }
    }
    return request;
}

}  // namespace

RuntimeFrontendServer::RuntimeFrontendServer(const StaticConfig& config, RuntimeConfigStore& config_store, SharedState& shared_state)
    : config_(config),
      config_store_(config_store),
      shared_state_(shared_state) {}

RuntimeFrontendServer::~RuntimeFrontendServer() {
    stop();
}

void RuntimeFrontendServer::start() {
    if (running_.exchange(true)) {
        return;
    }
    thread_ = std::thread([this]() { serve(); });
}

void RuntimeFrontendServer::stop() {
    if (!running_.exchange(false)) {
        return;
    }
#if defined(_WIN32)
    SOCKET wake = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (wake != INVALID_SOCKET) {
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(config_.frontend_port);
        inet_pton(AF_INET, config_.frontend_host.c_str(), &addr.sin_addr);
        connect(wake, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        closesocket(wake);
    }
#endif
    if (thread_.joinable()) {
        thread_.join();
    }
}

void RuntimeFrontendServer::serve() {
#if defined(_WIN32)
    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        std::cout << "[frontend] WSAStartup failed.\n";
        running_ = false;
        return;
    }

    SOCKET server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server == INVALID_SOCKET) {
        WSACleanup();
        running_ = false;
        return;
    }

    u_long nonblocking = 1;
    ioctlsocket(server, FIONBIO, &nonblocking);
    const char one = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(config_.frontend_port);
    if (inet_pton(AF_INET, config_.frontend_host.c_str(), &addr.sin_addr) != 1) {
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    }

    if (bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR || listen(server, 8) == SOCKET_ERROR) {
        std::cout << "[frontend] Failed to bind http://" << config_.frontend_host << ':' << config_.frontend_port << "/\n";
        closesocket(server);
        WSACleanup();
        running_ = false;
        return;
    }

    std::cout << "[frontend] Runtime UI: http://" << config_.frontend_host << ':' << config_.frontend_port << "/\n";
    const std::string html = buildRuntimePageHtml();

    while (running_) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(server, &readfds);
        timeval timeout{};
        timeout.tv_sec = 0;
        timeout.tv_usec = 200000;
        const int ready = select(0, &readfds, nullptr, nullptr, &timeout);
        if (ready <= 0 || !FD_ISSET(server, &readfds)) {
            continue;
        }

        SOCKET client = accept(server, nullptr, nullptr);
        if (client == INVALID_SOCKET) {
            continue;
        }
        u_long client_blocking = 0;
        ioctlsocket(client, FIONBIO, &client_blocking);

        const std::string request = readRequest(client);
        const size_t line_end = request.find("\r\n");
        const std::string request_line = line_end == std::string::npos ? request : request.substr(0, line_end);
        const size_t method_end = request_line.find(' ');
        const size_t path_end = method_end == std::string::npos ? std::string::npos : request_line.find(' ', method_end + 1);
        const std::string method = method_end == std::string::npos ? std::string{} : request_line.substr(0, method_end);
        const std::string path = path_end == std::string::npos ? std::string{} : request_line.substr(method_end + 1, path_end - method_end - 1);
        const size_t body_pos = request.find("\r\n\r\n");
        const std::string body = body_pos == std::string::npos ? std::string{} : request.substr(body_pos + 4);

        std::string response;
        if (method == "GET" && rootPathMatches(path)) {
            response = httpResponse(html, "text/html; charset=utf-8");
        } else if (method == "GET" && (pathMatches(path, "/api/pid/status") || pathMatches(path, "/api/pidf/status"))) {
            response = httpResponse(
                std::string("{\"status\":") + buildStatusJson(config_store_.snapshot(), shared_state_) + "}",
                "application/json; charset=utf-8");
        } else if (method == "GET" && pathMatches(path, "/api/recoil/profiles")) {
            response = httpResponse(buildRecoilProfilesPayload(config_), "application/json; charset=utf-8");
        } else if (method == "GET" && pathMatches(path, "/api/recoil")) {
            response = httpResponse(buildRecoilPayload(config_, config_store_, shared_state_), "application/json; charset=utf-8");
        } else if (method == "POST" && pathMatches(path, "/api/recoil")) {
            RuntimeConfig next = config_store_.snapshot();
            std::string error;
            if (applyRecoilPatch(body, next, error)) {
                config_store_.update(next);
                response = httpResponse(buildRecoilPayload(config_, config_store_, shared_state_), "application/json; charset=utf-8");
            } else {
                response = httpResponse(std::string("{\"error\":\"") + jsonEscape(error) + "\"}", "application/json; charset=utf-8", 400, "Bad Request");
            }
        } else if (method == "POST" && (pathMatches(path, "/api/pid/reset") || pathMatches(path, "/api/pidf/reset"))) {
            config_store_.requestReset();
            response = httpResponse(buildApiPayload(config_store_, shared_state_), "application/json; charset=utf-8");
        } else if (method == "GET" && (pathMatches(path, "/api/pid") || pathMatches(path, "/api/pidf"))) {
            response = httpResponse(buildApiPayload(config_store_, shared_state_), "application/json; charset=utf-8");
        } else if (method == "POST" && (pathMatches(path, "/api/pid") || pathMatches(path, "/api/pidf"))) {
            RuntimeConfig next = config_store_.snapshot();
            std::string error;
            if (applyRuntimePatch(body, next, error)) {
                config_store_.update(next);
                response = httpResponse(buildApiPayload(config_store_, shared_state_), "application/json; charset=utf-8");
            } else {
                response = httpResponse(std::string("{\"error\":\"") + jsonEscape(error) + "\"}", "application/json; charset=utf-8", 400, "Bad Request");
            }
        } else {
            response = httpResponse("{\"error\":\"Not found.\"}", "application/json; charset=utf-8", 404, "Not Found");
        }

        send(client, response.data(), static_cast<int>(response.size()), 0);
        closesocket(client);
    }

    closesocket(server);
    WSACleanup();
#else
    running_ = false;
#endif
}

std::unique_ptr<RuntimeFrontendServer> makeRuntimeFrontend(const StaticConfig& config, RuntimeConfigStore& config_store, SharedState& shared_state) {
    return std::make_unique<RuntimeFrontendServer>(config, config_store, shared_state);
}

}  // namespace delta
