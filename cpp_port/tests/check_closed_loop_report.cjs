// Requires disposable Chrome on --remote-debugging-port=9229.
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
(async () => {
  const pages = await (await fetch('http://127.0.0.1:9229/json/list')).json();
  const ws = new WebSocket(pages.find(p => p.type === 'page').webSocketDebuggerUrl);
  const pending = new Map(), errors = []; let sequence = 0;
  ws.onmessage = event => {
    const msg = JSON.parse(event.data);
    if (msg.id) { const p = pending.get(msg.id); pending.delete(msg.id); msg.error ? p.reject(msg.error) : p.resolve(msg.result); }
    if (msg.method === 'Runtime.exceptionThrown') errors.push(msg.params.exceptionDetails);
  };
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++sequence; pending.set(id, { resolve, reject }); ws.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  try {
    await send('Page.enable'); await send('Runtime.enable');
    await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false });
    await send('Page.navigate', { url: pathToFileURL(path.resolve(process.argv[2])).href });
    let ready = false;
    for (let i = 0; i < 100; ++i) {
      ready = await evaluate("document.readyState === 'complete' && document.getElementById('cases')?.options.length > 0 && document.getElementById('baseline')?.width > 300");
      if (ready) break;
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    if (!ready) throw Error('Comparison failed to initialize images');
    const result = await evaluate(`(async () => {
      const check=(v,m)=>{if(!v)throw Error(m);};
      const select=document.getElementById('cases');
      check(select.options.length===new Set(runs.map(r=>r.case)).size && select.options.length>=4,
        'Suite must show all paired cases');
      select.value='crossing-1500';select.dispatchEvent(new Event('change'));
      const scrub=document.getElementById('scrub');scrub.value=65;scrub.dispatchEvent(new Event('input'));
      document.getElementById('next').click();check(frame===66,'Synchronized frame step failed');
      document.getElementById('back').click();check(frame===65,'Backward frame step failed');
      await new Promise(resolve=>setTimeout(resolve,250));
      for(const variant of variants){
        check(document.getElementById(variant).width>300,'Missing scene image');
        check(document.getElementById(variant+'-state').textContent.includes('Scene 66'),'Panels are not synchronized');
        check(document.getElementById(variant+'-link').getAttribute('href').includes('crossing-1500'),'Report link points to wrong case');
      }
      check(document.querySelectorAll('#metrics tr').length===8,'Missing selection/following/ambiguity comparison metrics');
      if(runs[0].mode==='visible')check(document.querySelector('main>p').textContent.includes('real-time scheduling'),
        'Visible results must not be described as controlled-time latency');
      return {cases:select.options.length,frame:frame+1,metrics:document.querySelectorAll('#metrics tr').length};
    })()`);
    if (errors.length) throw Error(JSON.stringify(errors));
    if (process.argv[3]) {
      const screenshot = await send('Page.captureScreenshot', { format: 'png' });
      fs.writeFileSync(process.argv[3], Buffer.from(screenshot.data, 'base64'));
    }
    const fullReport = await evaluate("runs[0].mode==='visible' ? new URL(document.getElementById('motion-identity-link').getAttribute('href'),location.href).href : null");
    if(fullReport) {
      await send('Page.navigate',{url:fullReport});
      let valid=false;
      for(let i=0;i<100;++i) {
        valid=await evaluate("document.querySelector('h1')?.textContent==='Visible-screen GPU closed loop' && document.getElementById('number')?.textContent.includes(' / ') && document.querySelector('pre')?.textContent.includes('dispatch_to_simulated_application')");
        if(valid)break;
        await new Promise(resolve=>setTimeout(resolve,50));
      }
      if(!valid)throw Error('Visible report must display measured timing stages and loaded playback');
    }
    console.log(JSON.stringify(result));
  } finally { await send('Browser.close'); ws.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
