#ifndef ARDUINO_ARCH_RP2040
#error "Not building for RP2040 - fix Tools > Board"
#endif

#include "pio_usb.h"
#include "Adafruit_TinyUSB.h"
#include "usbh_helper.h"   // provides: Adafruit_USBH_Host USBHost;
                           // and rp2040_configure_pio_usb() (pins, 5V enable, pio cfg)
#include "pico/mutex.h"    // cross-core hardware mutex

// ---------- device side (core 0) ----------
uint8_t const desc_hid_report[] = { TUD_HID_REPORT_DESC_MOUSE() };

Adafruit_USBD_HID usb_hid(desc_hid_report, sizeof(desc_hid_report),
                          HID_ITF_PROTOCOL_MOUSE, 1, false);

// ---------- sensitivity (physical mouse only) ----------
// 1.0 = unchanged, 2.0 = twice as fast, 0.5 = half speed.
// Applied ONLY to physical mouse deltas; PC-injected deltas are never scaled.
volatile float mouse_sensitivity = 10.0f;

// Fractional remainder carried between reports so slow/fractional scaling
// doesn't lose sub-pixel motion (e.g. 0.5 sensitivity on a delta of 1).
float sens_residual_x = 0.0f;
float sens_residual_y = 0.0f;

// ---------- shared state between cores ----------
auto_init_mutex(accum_mutex);

int32_t shared_dx = 0;
int32_t shared_dy = 0;
int32_t shared_wheel = 0;
uint8_t shared_buttons = 0;      // last known physical button state

// Tracks what buttons the PC last saw, so button-only changes get sent.
uint8_t last_sent_buttons = 0;

// ---------- serial protocol ----------
static const uint8_t SYNC = 0xAA;

enum { WAIT_SYNC, READ_BODY };
uint8_t state = WAIT_SYNC;
uint8_t buf[5];
uint8_t idx = 0;

// ---------- debug counters ----------
volatile uint32_t loop0_count = 0;
volatile uint32_t ready_true = 0, ready_false = 0;

//====================================================================
// CORE 0 : native USB device + serial injector + sole HID reporter
//====================================================================
void setup() {
  usb_hid.begin();
  Serial.begin(921600);
  while (!TinyUSBDevice.mounted()) delay(1);
}

// The single point that talks to the PC.
static inline void sendReport(uint8_t buttons, int8_t dx, int8_t dy, int8_t wheel) {
  if (usb_hid.ready()) { ready_true++;  usb_hid.mouseReport(0, buttons, dx, dy, wheel, 0); }
  else                 { ready_false++; }
}

// Grab + clear the accumulated deltas (under the lock), then emit outside the lock.
// Sends when there is motion OR when the button state changed. One report per flush;
// carries leftover forward.
void flushAccumulated() {
  if (!usb_hid.ready()) return;   // HID busy — don't spin, try next loop

  mutex_enter_blocking(&accum_mutex);
  int32_t dx = shared_dx, dy = shared_dy, w = shared_wheel;
  uint8_t btn = shared_buttons;
  shared_dx = 0; shared_dy = 0; shared_wheel = 0;
  mutex_exit(&accum_mutex);

  bool has_motion = (dx || dy || w);
  bool buttons_changed = (btn != last_sent_buttons);
  if (!has_motion && !buttons_changed) return;

  int8_t sx = (dx> 127)?127:(dx<-127)?-127:(int8_t)dx;
  int8_t sy = (dy> 127)?127:(dy<-127)?-127:(int8_t)dy;
  int8_t sw = (w > 127)?127:(w <-127)?-127:(int8_t)w;
  sendReport(btn, sx, sy, sw);
  last_sent_buttons = btn;

  // push remainder back so it goes out next loop iterations
  int32_t rem_dx = dx - sx, rem_dy = dy - sy, rem_w = w - sw;
  if (rem_dx || rem_dy || rem_w) {
    mutex_enter_blocking(&accum_mutex);
    shared_dx += rem_dx; shared_dy += rem_dy; shared_wheel += rem_w;
    mutex_exit(&accum_mutex);
  }
}

void loop() {
  loop0_count++;

  // Debug print lives here (core 0) because it touches Serial, the same USB
  // CDC connection core 0's parser below reads. TinyUSB's CDC/HID state is
  // NOT safe to touch from both cores concurrently — calling Serial.printf
  // from core 1 (as loop1() used to, once a second) could race the read
  // below and wedge the whole USB device: PC-injected movement stops being
  // consumed AND, since loop1() also drives USBHost.task() for the physical
  // mouse, the physical mouse passthrough stalls too. Keep all Serial access
  // on core 0.
  static uint32_t debug_print_tick = 0;
  if (millis() - debug_print_tick > 1000) {
    debug_print_tick = millis();
    Serial.printf("loop0=%lu , ready_true = %lu, ready_false = %lu\n",
                  loop0_count, ready_true, ready_false);
  }

  // ----- serial parser: injection merges via shared deltas (NEVER scaled) -----
  while (Serial.available()) {
    uint8_t b = Serial.read();

    if (state == WAIT_SYNC) {
      if (b == SYNC) {
        idx = 0;
        state = READ_BODY;
      }
    } else {
      buf[idx++] = b;
      if (idx == 5) {
        uint8_t chk = buf[0] ^ buf[1] ^ buf[2] ^ buf[3];
        if (chk == buf[4]) {
          int16_t dx = (int16_t)((buf[0] << 8) | buf[1]);
          int16_t dy = (int16_t)((buf[2] << 8) | buf[3]);
          mutex_enter_blocking(&accum_mutex);
          shared_dx += dx;      // injected deltas added raw, no sensitivity
          shared_dy += dy;
          mutex_exit(&accum_mutex);
        }
        state = WAIT_SYNC;
      }
    }
  }

  flushAccumulated();
}

//====================================================================
// CORE 1 : PIO-USB host — reads the physical mouse on the USB-A port
//====================================================================
void setup1() {
  rp2040_configure_pio_usb();

  pinMode(PIN_5V_EN, OUTPUT);
  digitalWrite(PIN_5V_EN, PIN_5V_EN_STATE);

  USBHost.begin(1);
}

void loop1() {
  // No Serial access here — see the comment in loop() (core 0) for why.
  USBHost.task();
}

//--------------------------------------------------------------------
// Host HID callbacks
//--------------------------------------------------------------------

void tuh_hid_mount_cb(uint8_t dev_addr, uint8_t instance,
                      uint8_t const* desc_report, uint16_t desc_len) {
  (void)desc_report;
  (void)desc_len;
  // Report protocol (not boot) so the wheel byte is present.
  tuh_hid_receive_report(dev_addr, instance);
}

void tuh_hid_umount_cb(uint8_t dev_addr, uint8_t instance) {
  (void)dev_addr;
  (void)instance;
  mutex_enter_blocking(&accum_mutex);
  shared_buttons = 0;
  mutex_exit(&accum_mutex);
}

// Called each time the physical mouse sends a report.
void tuh_hid_report_received_cb(uint8_t dev_addr, uint8_t instance,
                                uint8_t const* report, uint16_t len) {
  // Standard mouse report: report[0]=buttons, [1]=dx, [2]=dy, [3]=wheel
  if (len >= 3) {
    int8_t raw_dx = (int8_t)report[1];
    int8_t raw_dy = (int8_t)report[2];
    int8_t w      = (len >= 4) ? (int8_t)report[3] : 0;   // wheel not scaled

    // ---- apply sensitivity to PHYSICAL dx/dy only ----
    // Multiply, add the leftover fraction from last time, then split into
    // an integer part (sent now) and a new fractional remainder (carried).
    float fx = raw_dx * mouse_sensitivity + sens_residual_x;
    float fy = raw_dy * mouse_sensitivity + sens_residual_y;

    int32_t out_dx = (int32_t)fx;   // truncates toward zero
    int32_t out_dy = (int32_t)fy;

    sens_residual_x = fx - out_dx;  // keep the sub-integer part
    sens_residual_y = fy - out_dy;

    mutex_enter_blocking(&accum_mutex);
    shared_dx += out_dx;            // scaled physical motion
    shared_dy += out_dy;
    shared_wheel += w;              // wheel unscaled
    shared_buttons = report[0];
    mutex_exit(&accum_mutex);
  }

  tuh_hid_receive_report(dev_addr, instance);
}