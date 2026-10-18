// Browser smoke check using Node's built-in WebSocket and Chrome DevTools.
// Start a disposable headless Chrome with --remote-debugging-port=9229 first.
// node check_benchmark_report.cjs path/to/report.html [screenshot.png]
const fs = require('node:fs');
const { pathToFileURL } = require('node:url');
const path = require('node:path');

(async () => {
  const pages = await (await fetch('http://127.0.0.1:9229/json/list')).json();
  const page = pages.find(p => p.type === 'page');
  if (!page) throw new Error('No Chrome page available');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  const pending = new Map(), errors = [];
  let sequence = 0;
  ws.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const request = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) request.reject(new Error(JSON.stringify(message.error)));
      else request.resolve(message.result);
    }
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
  };
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++sequence;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => {
    const value = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (value.exceptionDetails) throw new Error(JSON.stringify(value.exceptionDetails));
    return value.result.value;
  };
  try {
    await send('Page.enable'); await send('Runtime.enable');
    await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false });
    await send('Page.navigate', { url: pathToFileURL(path.resolve(process.argv[2])).href });
    let ready = false;
    for (let i = 0; i < 100; i++) {
      ready = await evaluate("document.readyState === 'complete' && document.getElementById('scenario')?.options.length > 0");
      if (ready) break;
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    if (!ready) throw new Error('Report failed to initialize');
    const result = await evaluate(`(() => {
      const check = (condition, message) => { if (!condition) throw Error(message); };
      check(document.querySelectorAll('#results tr').length === run.runs.length, 'Missing result rows');
      const select = document.getElementById('scenario');
      select.value = run.runs.length - 1; select.dispatchEvent(new Event('change'));
      check(selected === run.runs.length - 1, 'Scenario selection failed');
      const scrub = document.getElementById('scrub');
      scrub.value = 140; scrub.dispatchEvent(new Event('input'));
      check(frame === 140, 'Scrubber failed');
      document.getElementById('forward').click(); check(frame === 141, 'Frame step failed');
      const event = document.querySelector('#events button');
      if (event) { event.click(); check(frame !== 141, 'Failure interval seek failed'); }
      document.getElementById('play').click(); check(playing, 'Playback did not start');
      document.getElementById('play').click(); check(!playing, 'Playback did not pause');
      check(Boolean(base) === !document.getElementById('baseline-panel').hidden, 'Comparison panel visibility incorrect');
      const pixels = document.getElementById('candidate').getContext('2d').getImageData(0,0,900,506).data;
      check(pixels.some(value => value !== 0), 'Replay canvas is empty');
      setFrame(140);
      return {scenarios:run.runs.length, comparison:Boolean(base), status:document.getElementById('status').textContent};
    })()`);
    if (errors.length) throw new Error(JSON.stringify(errors));
    if (process.argv[3]) {
      const screenshot = await send('Page.captureScreenshot', { format: 'png' });
      fs.writeFileSync(process.argv[3], Buffer.from(screenshot.data, 'base64'));
    }
    console.log(JSON.stringify(result, null, 2));
  } finally { ws.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
