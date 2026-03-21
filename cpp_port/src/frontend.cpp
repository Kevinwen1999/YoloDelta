#include "delta/frontend.hpp"

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

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

namespace delta {

namespace {

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

const char* trackingStrategyName(const TrackingStrategy strategy) {
    return strategy == TrackingStrategy::Raw ? "raw" : "raw_delta";
}

TrackingStrategy parseTrackingStrategy(const std::string& value) {
    return value == "raw" ? TrackingStrategy::Raw : TrackingStrategy::RawDelta;
}

const char* engageButtonName(const LeftHoldEngageButton button) {
    switch (button) {
    case LeftHoldEngageButton::Left: return "leftkey";
    case LeftHoldEngageButton::Both: return "both";
    case LeftHoldEngageButton::Right:
    default: return "rightkey";
    }
}

LeftHoldEngageButton parseEngageButton(const std::string& value) {
    if (value == "leftkey") return LeftHoldEngageButton::Left;
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

std::string buildConfigJson(const RuntimeConfig& cfg, const std::uint64_t version, const std::uint64_t reset_token) {
    std::ostringstream oss;
    oss << "{"
        << "\"pid_enable\":" << (cfg.pid_enable ? "true" : "false") << ","
        << "\"tracking_enabled\":" << (cfg.tracking_enabled ? "true" : "false") << ","
        << "\"debug_preview_enable\":" << (cfg.debug_preview_enable ? "true" : "false") << ","
        << "\"capture_cached_timeout_ms\":" << cfg.capture_cached_timeout_ms << ","
        << "\"body_y_ratio\":" << cfg.body_y_ratio << ","
        << "\"tracking_strategy\":\"" << trackingStrategyName(cfg.tracking_strategy) << "\","
        << "\"tracking_velocity_alpha\":" << cfg.tracking_velocity_alpha << ","
        << "\"kp\":" << cfg.kp << ","
        << "\"ki\":" << cfg.ki << ","
        << "\"kd\":" << cfg.kd << ","
        << "\"integral_limit\":" << cfg.integral_limit << ","
        << "\"anti_windup_gain\":" << cfg.anti_windup_gain << ","
        << "\"derivative_alpha\":" << cfg.derivative_alpha << ","
        << "\"output_limit\":" << cfg.output_limit << ","
        << "\"sticky_bias_px\":" << cfg.sticky_bias_px << ","
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
        << "\"triggerbot_enable\":" << (cfg.triggerbot_enable ? "true" : "false") << ","
        << "\"triggerbot_click_hold_s\":" << cfg.triggerbot_click_hold_s << ","
        << "\"triggerbot_click_cooldown_s\":" << cfg.triggerbot_click_cooldown_s << ","
        << "\"recoil_compensation_y_rate_px_s\":" << cfg.recoil_compensation_y_rate_px_s << ","
        << "\"recoil_compensation_y_px\":" << cfg.recoil_compensation_y_px << ","
        << "\"left_hold_engage_button\":\"" << engageButtonName(cfg.left_hold_engage_button) << "\","
        << "\"recoil_tune_fallback_ignore_mode_check\":"
        << (cfg.recoil_tune_fallback_ignore_mode_check ? "true" : "false") << ","
        << "\"sendinput_gain_x\":" << cfg.sendinput_gain_x << ","
        << "\"sendinput_gain_y\":" << cfg.sendinput_gain_y << ","
        << "\"sendinput_max_step\":" << cfg.sendinput_max_step << ","
        << "\"raw_max_step_y\":" << cfg.raw_max_step_y << ","
        << "\"version\":" << version << ","
        << "\"reset_token\":" << reset_token
        << "}";
    return oss.str();
}

std::string buildStatusJson(const SharedState& shared_state) {
    std::lock_guard<std::mutex> lock(shared_state.mutex);
    std::ostringstream oss;
    oss << "{"
        << "\"running\":" << (shared_state.running ? "true" : "false") << ","
        << "\"mode\":" << shared_state.toggles.mode << ","
        << "\"mode_label\":\"" << (shared_state.toggles.mode == 0 ? "OFF" : "ACTIVE") << "\","
        << "\"aimmode\":" << shared_state.toggles.aimmode << ","
        << "\"aimmode_label\":\"" << (shared_state.toggles.aimmode == 0 ? "HEAD" : "BODY") << "\","
        << "\"tracking_strategy\":\"" << jsonEscape(shared_state.tracking_strategy) << "\","
        << "\"target_found\":" << (shared_state.target_found ? "true" : "false") << ","
        << "\"target_speed\":" << shared_state.target_speed << ","
        << "\"target_cls\":" << shared_state.target_cls << ","
        << "\"aim_dx\":" << shared_state.aim_dx << ","
        << "\"aim_dy\":" << shared_state.aim_dy
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
      <p>Live tuning for the native raw/raw-delta tracking runtime.</p>
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
      {key:"capture_cached_timeout_ms",label:"Cached Capture Timeout (ms)",type:"number",step:1,min:0},
      {key:"body_y_ratio",label:"Body Aim Y Ratio",type:"number",step:0.01,min:0,max:1},
      {key:"tracking_strategy",label:"Tracking Strategy",type:"select",options:[{value:"raw",label:"Raw Detection"},{value:"raw_delta",label:"Raw + Velocity"}]},
      {key:"tracking_velocity_alpha",label:"Velocity Beta",type:"number",step:0.001},
      {key:"kp",label:"Kp (X/Y)",type:"number",step:0.001},
      {key:"ki",label:"Ki (X/Y)",type:"number",step:0.001},
      {key:"kd",label:"Kd (X/Y)",type:"number",step:0.001},
      {key:"integral_limit",label:"Integral Limit",type:"number",step:1},
      {key:"anti_windup_gain",label:"Anti-Windup Gain",type:"number",step:0.001},
      {key:"derivative_alpha",label:"Derivative Alpha",type:"number",step:0.001},
      {key:"output_limit",label:"PID Output Limit",type:"number",step:1},
      {key:"sticky_bias_px",label:"Sticky Bias (px)",type:"number",step:1},
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
      {key:"recoil_compensation_y_rate_px_s",label:"Recoil Comp Y Rate (px/s)",type:"number",step:1},
      {key:"recoil_compensation_y_px",label:"Legacy Recoil Comp Y (px/cmd)",type:"number",step:0.1},
      {key:"left_hold_engage_button",label:"F6 Engage Button",type:"select",options:[{value:"rightkey",label:"Right Key"},{value:"leftkey",label:"Left Key"},{value:"both",label:"Either Key"}]},
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
      runtime.textContent=`Runtime ${status.running ? "running" : "stopped"} | mode ${status.mode_label} | aim ${status.aimmode_label} | track ${status.tracking_strategy} | target ${status.target_found ? "locked" : "none"} | speed ${Number(status.target_speed).toFixed(1)} | cmd (${status.aim_dx}, ${status.aim_dy})`;
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

bool applyRuntimePatch(const std::string& body, RuntimeConfig& cfg, std::string& error) {
    if (const auto value = extractJsonBool(body, "pid_enable"); value.has_value()) cfg.pid_enable = *value;
    if (const auto value = extractJsonBool(body, "tracking_enabled"); value.has_value()) cfg.tracking_enabled = *value;
    if (const auto value = extractJsonBool(body, "debug_preview_enable"); value.has_value()) cfg.debug_preview_enable = *value;
    if (const auto value = extractJsonNumber(body, "capture_cached_timeout_ms"); value.has_value()) {
        cfg.capture_cached_timeout_ms = std::max(0, static_cast<int>(std::lround(*value)));
    }
    if (const auto value = extractJsonNumber(body, "body_y_ratio"); value.has_value()) cfg.body_y_ratio = clamp(static_cast<float>(*value), 0.0F, 1.0F);
    if (const auto value = extractJsonString(body, "tracking_strategy"); value.has_value()) cfg.tracking_strategy = parseTrackingStrategy(*value);
    if (const auto value = extractJsonNumber(body, "tracking_velocity_alpha"); value.has_value()) cfg.tracking_velocity_alpha = clamp(static_cast<float>(*value), 0.0F, 1.0F);
    if (const auto value = extractJsonNumber(body, "kp"); value.has_value()) cfg.kp = static_cast<float>(*value);
    if (const auto value = extractJsonNumber(body, "ki"); value.has_value()) cfg.ki = static_cast<float>(*value);
    if (const auto value = extractJsonNumber(body, "kd"); value.has_value()) cfg.kd = static_cast<float>(*value);
    if (const auto value = extractJsonNumber(body, "integral_limit"); value.has_value()) cfg.integral_limit = std::max(0.0F, static_cast<float>(*value));
    if (const auto value = extractJsonNumber(body, "anti_windup_gain"); value.has_value()) cfg.anti_windup_gain = clamp(static_cast<float>(*value), 0.0F, 1.0F);
    if (const auto value = extractJsonNumber(body, "derivative_alpha"); value.has_value()) cfg.derivative_alpha = clamp(static_cast<float>(*value), 0.0F, 1.0F);
    if (const auto value = extractJsonNumber(body, "output_limit"); value.has_value()) cfg.output_limit = std::max(0.0F, static_cast<float>(*value));
    if (const auto value = extractJsonNumber(body, "sticky_bias_px"); value.has_value()) cfg.sticky_bias_px = std::max(0.0F, static_cast<float>(*value));
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
    if (const auto value = extractJsonBool(body, "triggerbot_enable"); value.has_value()) cfg.triggerbot_enable = *value;
    if (const auto value = extractJsonNumber(body, "triggerbot_click_hold_s"); value.has_value()) cfg.triggerbot_click_hold_s = std::max(0.0F, static_cast<float>(*value));
    if (const auto value = extractJsonNumber(body, "triggerbot_click_cooldown_s"); value.has_value()) cfg.triggerbot_click_cooldown_s = std::max(0.0F, static_cast<float>(*value));
    if (const auto value = extractJsonNumber(body, "recoil_compensation_y_rate_px_s"); value.has_value()) cfg.recoil_compensation_y_rate_px_s = static_cast<float>(*value);
    if (const auto value = extractJsonNumber(body, "recoil_compensation_y_px"); value.has_value()) cfg.recoil_compensation_y_px = static_cast<float>(*value);
    if (const auto value = extractJsonString(body, "left_hold_engage_button"); value.has_value()) cfg.left_hold_engage_button = parseEngageButton(*value);
    if (const auto value = extractJsonBool(body, "recoil_tune_fallback_ignore_mode_check"); value.has_value()) cfg.recoil_tune_fallback_ignore_mode_check = *value;
    if (const auto value = extractJsonNumber(body, "sendinput_gain_x"); value.has_value()) cfg.sendinput_gain_x = static_cast<float>(*value);
    if (const auto value = extractJsonNumber(body, "sendinput_gain_y"); value.has_value()) cfg.sendinput_gain_y = static_cast<float>(*value);
    if (const auto value = extractJsonNumber(body, "sendinput_max_step"); value.has_value()) cfg.sendinput_max_step = std::max(1, static_cast<int>(std::lround(*value)));
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
        << "\"status\":" << buildStatusJson(shared_state)
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
    const std::string html = buildPageHtml();

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
            response = httpResponse(std::string("{\"status\":") + buildStatusJson(shared_state_) + "}", "application/json; charset=utf-8");
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
