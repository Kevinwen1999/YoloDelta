param([Parameter(Mandatory=$true)][string]$SuiteRoot)
$ErrorActionPreference = 'Stop'
$SuiteRoot = (Resolve-Path -LiteralPath $SuiteRoot).Path
$rows = Get-Content -LiteralPath (Join-Path $SuiteRoot 'suite.json') -Raw | ConvertFrom-Json
$references = @{}
$runs = foreach ($row in $rows) {
    $directory = $row.Case + '-' + $row.Variant
    $run = Get-Content -LiteralPath (Join-Path $SuiteRoot "$directory/closed-loop.json") -Raw | ConvertFrom-Json
    if ($run.schema_version -lt 2 -or !$run.validity.valid) { throw "Comparison requires valid schema 2 results: $directory" }
    if ($run.correctness -and !$run.correctness.passed) { throw "Comparison contains correctness violations: $directory" }
    # Setup duration is measured evidence, not a different hardware layout.
    $hardware = $run.capture_hardware | Select-Object * -ExcludeProperty capture_warmup_ms
    $signature = ConvertTo-Json -InputObject @($run.scope,$hardware,$run.fingerprints,$run.scene,$run.gpu.name,$run.gpu.ordinal,$run.build.source_hash) -Depth 12 -Compress
    if ($references.ContainsKey($row.Case) -and $references[$row.Case] -ne $signature) {
        throw "Paired runs differ in scene, assets, model, hardware or source: $($row.Case)"
    }
    $references[$row.Case] = $signature
    $unattributed = @($run.applied_movements | Where-Object { $_.selected_truth_id -lt 0 }).Count
    $run.summary | Add-Member -NotePropertyName 'unattributed_movements' -NotePropertyValue $unattributed -Force
    [pscustomobject]@{ case=$row.Case; variant=$row.Variant; directory=$directory;
        frames=$run.frames; summary=$run.summary; gpu=$run.gpu; scene=$run.scene; mode=$run.mode; desktop=$(if($run.scene.viewport){$run.scene.viewport}else{$run.capture_hardware.desktop_size}) }
}
$dataJson = ConvertTo-Json -InputObject @($runs) -Depth 25 -Compress
$dataJson = $dataJson.Replace('<','\u003c')
$html = @'
<!doctype html><meta charset="utf-8"><title>Delta paired closed-loop comparison</title>
<style>body{margin:0;background:#101820;color:#e9f2f8;font:15px system-ui}main{max-width:1500px;margin:auto;padding:24px}select,button{font:inherit;padding:7px}input{width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}section{background:#1b2933;padding:15px;border-radius:10px;margin:16px 0}canvas{width:100%;background:#16232d}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:8px;text-align:left;border-bottom:1px solid #425765}a{color:#81c8ff}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<main><h1>Paired closed-loop comparison</h1><p>Controlled-time GPU runs using backup imagery and simulated camera movement. Camera sensitivity and cutout geometry remain exploratory. Inference wall time is measured; logical observation age is simulated.</p>
<label>Scenario <select id="cases"></select></label><section><label>Frame <b id="number"></b><input id="scrub" type="range" min="0" value="0"></label><button id="back">Previous</button> <button id="next">Next</button></section>
<div class="pair"><section><h2>Default baseline</h2><canvas id="baseline"></canvas><p id="baseline-state"></p><a id="baseline-link">Full report</a></section><section><h2>Motion candidate</h2><canvas id="motion-identity"></canvas><p id="motion-identity-state"></p><a id="motion-identity-link">Full report</a></section></div>
<section><table><thead><tr><th>Metric</th><th>Baseline</th><th>Motion candidate</th></tr></thead><tbody id="metrics"></tbody></table></section>
<script id="runs" type="application/json">__RUN_DATA__</script><script>
const runs=JSON.parse(document.getElementById('runs').textContent),variants=['baseline','motion-identity'];
if(runs[0].mode==='visible')document.querySelector('main>p').textContent='Visible-screen GPU runs with production DXGI capture and real-time scheduling. No mouse or serial output. Paired frames align by observation index; each run has its own actual timestamps. Camera response and cutouts remain exploratory.';
const cases=document.getElementById('cases'),scrub=document.getElementById('scrub');let frame=0;
for(const name of [...new Set(runs.map(r=>r.case))]){const o=document.createElement('option');o.textContent=name;o.value=name;cases.append(o);}
function selected(v){return runs.find(r=>r.case===cases.value&&r.variant===v);}
function draw(v){const run=selected(v),idx=frame,name=cases.value,r=run.frames[idx],img=new Image();img.onload=()=>{if(frame!==idx||cases.value!==name)return;const canvas=document.getElementById(v);canvas.width=r.capture[2];canvas.height=r.capture[3];const ctx=canvas.getContext('2d');ctx.drawImage(img,0,0);function box(b,color){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.strokeRect(b[0]-r.capture[0],b[1]-r.capture[1],b[2]-b[0],b[3]-b[1]);}for(const t of r.truth_boxes)box(t.bbox,t.id===1?'#42f59b':'#df8dff');for(const d of r.detector_boxes)box(d.bbox,d.selected?'#ffbd52':'#67b7ff');const x=(run.desktop?.[0]||800)/2-r.capture[0],y=(run.desktop?.[1]||800)/2-r.capture[1];ctx.strokeStyle='white';ctx.beginPath();ctx.moveTo(x-9,y);ctx.lineTo(x+9,y);ctx.moveTo(x,y-9);ctx.lineTo(x,y+9);ctx.stroke();};img.src=`${run.directory}/${r.image}`;
document.getElementById(`${v}-state`).textContent=`Scene ${r.scene_state_id} | render ${r.render_ms} ms | available ${r.available_ms} ms | detections ${r.detections} | output ready ${r.output_ready} | ${r.motion_decision||'legacy policy'} | selected identity ${r.selected_truth_id}`;document.getElementById(`${v}-link`).href=`${run.directory}/report.html`;}
function update(){
  const a=selected('baseline'),b=selected('motion-identity');
  scrub.max=Math.min(a.frames.length,b.frames.length)-1;frame=Math.min(frame,Number(scrub.max));scrub.value=frame;
  document.getElementById('number').textContent=`${frame+1} / ${Number(scrub.max)+1}`;variants.forEach(draw);
  const tbody=document.getElementById('metrics');tbody.replaceChildren();
  const metrics=[['Output-ready coverage','output_ready_coverage',true],['Frames within labeled target','fraction_frames_within_target_region',true],['Mean normalized aim error','mean_normalized_error_all_frames',false],['Attributed wrong-target movements','movements_with_wrong_selected_target',false],['Movements with ambiguous/unattributed identity','unattributed_movements',false],['Wrong moves before intended acquisition','identity_scoring.wrong_movements_before_intended_acquisition',false],['Wrong moves after intended acquisition','identity_scoring.wrong_movements_after_intended_acquisition',false],['Movements after loss deadline','movements_after_loss_deadline',false]];
  for(const [label,key,pct] of metrics){
    const tr=document.createElement('tr');
    const values=[a,b].map(r=>{const n=key.split('.').reduce((value,k)=>value?.[k],r.summary);return n==null?'n/a':pct?`${(n*100).toFixed(1)}%`:Number(n).toFixed(Number.isInteger(n)?0:3);});
    for(const value of [label,...values]){const td=document.createElement('td');td.textContent=value;tr.append(td);}tbody.append(tr);
  }
}
cases.onchange=()=>{frame=0;update();};scrub.oninput=()=>{frame=Number(scrub.value);update();};document.getElementById('back').onclick=()=>{frame=Math.max(0,frame-1);update();};document.getElementById('next').onclick=()=>{frame=Math.min(Number(scrub.max),frame+1);update();};update();
</script></main>
'@
$html.Replace('__RUN_DATA__',$dataJson) | Set-Content -LiteralPath (Join-Path $SuiteRoot 'comparison.html') -Encoding utf8
Write-Host "Comparison: $(Join-Path $SuiteRoot 'comparison.html')"
