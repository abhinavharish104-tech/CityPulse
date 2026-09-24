"""Live strip-chart recorder: a self-contained HTML/JS component.

The whole week for one neighbourhood is sent to the browser once; the browser then draws it reading by
reading with Plotly.extendTraces, a moving pen and a sliding window, like a hospital monitor or a
seismograph. Nothing is re-run on the Streamlit server while it plays, so there is no page refresh.
"""
from __future__ import annotations

import json
import os

import pandas as pd

import citypulse_core as cc
import cp_ui as ui

FEED_COL = {"rainfall": "#7CB7FF", "traffic_congestion": ui.MARIGOLD, "incidents_30m": ui.ROSE, "aqi": "#B79CFF"}


def payload(zd: pd.DataFrame, zone_name: str, now: pd.Timestamp, thr: dict, events: list, state_words: dict,
            offline: str | None, window_h: int = 6) -> dict:
    zd = zd.sort_values("timestamp")
    ms = (zd.timestamp.astype("datetime64[ns]").astype("int64") // 10**6).tolist()   # naive -> shown as-is
    feeds = []
    for f in cc.FEATURES:
        lo = (zd[f + "_typical"] - 2 * zd[f + "_sigma"]).clip(lower=0)
        hi = zd[f + "_typical"] + 2 * zd[f + "_sigma"]
        top = float(max(zd[f].max(), hi.max())) * 1.08 + 1e-6
        bot = 0.0 if f != "aqi" else float(min(zd[f].min(), lo.min())) * 0.9
        feeds.append(dict(key=f, name=cc.NAMES[f], unit=cc.UNITS[f], color=FEED_COL[f], offline=(f == offline),
                          dec=1 if f == "rainfall" else 0, pct=(f == "traffic_congestion"),
                          v=zd[f].round(2).tolist(), typ=zd[f + "_typical"].round(2).tolist(),
                          lo=lo.round(2).tolist(), hi=hi.round(2).tolist(), z=zd[f"z_{f}"].round(2).tolist(),
                          range=[bot, top]))
    start = int((zd.timestamp <= now).sum()) - 1
    evs = [dict(s=int(pd.Timestamp(e["start"]).value // 10**6), e=int(pd.Timestamp(e["end"]).value // 10**6)
                + 5 * 60 * 1000, state=e["peak_state"], label=e["pattern"]) for e in events]
    return dict(zone=zone_name, t=ms, feeds=feeds, h=zd.h.round(1).tolist(), st=zd.st_v.tolist(),
                pulse=zd.pulse_v.astype(int).tolist(), thr=thr, colors=ui.STATE, words=state_words, events=evs,
                start=max(start, 0), window_h=window_h)


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700;12..96,800&family=Atkinson+Hyperlegible:wght@400;700&display=swap" rel="stylesheet">
<script src="__PLOTLY__"></script>
<style>
 :root{--ink:__INK__;--panel:__PANEL__;--line:__LINE__;--text:__TEXT__;--muted:__MUTED__;--mari:__MARI__}
 *{box-sizing:border-box} body{margin:0;background:transparent;color:var(--text);font-family:'Atkinson Hyperlegible','Segoe UI',sans-serif}
 .wrap{background:#0F0C24;border:1px solid var(--line);border-radius:18px;padding:12px 14px 4px}
 .bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:10px}
 button,select{font:inherit;color:var(--text);background:#28234F;border:1px solid var(--line);border-radius:999px;padding:5px 14px;cursor:pointer}
 button:hover,select:hover{border-color:var(--mari)} button.main{background:var(--mari);color:#241a00;border-color:var(--mari);font-weight:700;min-width:96px}
 button:focus-visible,select:focus-visible{outline:2px solid var(--mari);outline-offset:2px}
 .clock{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:1.35rem;margin-left:auto;font-variant-numeric:tabular-nums}
 .chip{display:inline-flex;gap:6px;align-items:center;border:1px solid currentColor;border-radius:999px;padding:2px 10px;font-weight:700}
 .pulse{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:1.35rem;font-variant-numeric:tabular-nums}
 .read{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-bottom:6px}
 .cell{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:6px 10px;min-width:0}
 .cell .n{color:var(--muted);font-size:.8rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 .cell .v{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.25rem;font-variant-numeric:tabular-nums}
 .cell .u{color:var(--muted);font-size:.78rem}
 .cell.hot{border-color:currentColor}
 @media(max-width:700px){.read{grid-template-columns:repeat(2,1fr)}.clock{margin-left:0}}
</style></head><body><div class="wrap">
 <div class="bar">
  <button class="main" id="play" aria-label="Play or pause">❚❚ Pause</button>
  <button id="back" title="Go back 6 hours">↺ 6 h back</button>
  <label>Speed <select id="speed"><option value="4">slow</option><option value="12" selected>normal</option><option value="36">fast</option><option value="120">very fast</option></select></label>
  <label>Show <select id="win"><option value="3">3 h</option><option value="6" selected>6 h</option><option value="24">24 h</option></select></label>
  <span class="chip" id="chip">●</span><span class="pulse" id="pulse"></span>
  <span class="clock" id="clock"></span>
 </div>
 <div class="read" id="read"></div>
 <div id="plot" style="height:600px"></div>
</div>
<script>
const D = __DATA__;
const F = D.feeds, T = D.t, N = T.length, H = 60*60*1000;
const S = ms => new Date(ms).toISOString().slice(0,19).replace('T',' ');  // naive local time string
const TX = T.map(S);
const rm = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let cur = Math.max(0, D.start - 72), playing = !rm, acc = 0, winH = D.window_h;
const dom = [[0.83,1],[0.64,0.80],[0.45,0.61],[0.26,0.42],[0,0.22]];
const fmt = (f,v) => (v==null?'–':(f.dec? v.toFixed(1): Math.round(v)) + (f.pct?'%':''));

// ---------- traces (all share one x axis; one y axis per pane)
const traces = [];
F.forEach((f,k) => {
  const ya = k===0 ? 'y' : 'y'+(k+1);
  traces.push({x:[],y:[],yaxis:ya,mode:'lines',line:{width:0},hoverinfo:'skip',showlegend:false});
  traces.push({x:[],y:[],yaxis:ya,mode:'lines',line:{width:0},fill:'tonexty',fillcolor:'rgba(166,159,199,.13)',hoverinfo:'skip',showlegend:false});
  traces.push({x:[],y:[],yaxis:ya,mode:'lines',line:{color:'__MUTED__',width:1,dash:'dot'},name:'usual',hovertemplate:'usual %{y:.1f}<extra></extra>',showlegend:false});
  traces.push({x:[],y:[],yaxis:ya,mode:'lines',line:{color:f.offline?'__MUTED__':f.color,width:2.2,dash:f.offline?'dash':'solid'},name:f.name,hovertemplate:f.name+' %{y:.1f}<extra></extra>',showlegend:false});
});
const HY = traces.length;
traces.push({x:[],y:[],yaxis:'y5',mode:'lines',line:{color:'__TEXT__',width:2.2},name:'combined',hovertemplate:'combined %{y:.0f}<extra></extra>',showlegend:false});
const FL = traces.length;
traces.push({x:[],y:[],yaxis:'y5',mode:'markers',marker:{size:7,color:[]},hoverinfo:'skip',showlegend:false});
const PEN = traces.length;
F.forEach((f,k) => traces.push({x:[],y:[],yaxis:k===0?'y':'y'+(k+1),mode:'markers',hoverinfo:'skip',showlegend:false,
  marker:{size:11,color:f.offline?'__MUTED__':f.color,line:{color:'#fff',width:1.5}}}));
traces.push({x:[],y:[],yaxis:'y5',mode:'markers',hoverinfo:'skip',showlegend:false,marker:{size:12,color:'#fff'}});

const ann = F.map((f,k)=>({text:f.name+' ('+f.unit+')'+(f.offline?' — offline':''),xref:'paper',yref:'paper',x:0,y:dom[k][1]+0.005,xanchor:'left',yanchor:'bottom',showarrow:false,font:{color:'__MUTED__',size:12}}));
ann.push({text:'Combined anomaly score',xref:'paper',yref:'paper',x:0,y:dom[4][1]+0.005,xanchor:'left',yanchor:'bottom',showarrow:false,font:{color:'__MUTED__',size:12}});
const band = (a,b,c)=>({type:'rect',xref:'paper',yref:'y5',x0:0,x1:1,y0:a,y1:b,fillcolor:c,opacity:.10,line:{width:0},layer:'below'});
const base = [band(D.thr.watch,D.thr.confirmed,D.colors.Watch),band(D.thr.confirmed,D.thr.critical,D.colors.Confirmed),band(D.thr.critical,100,D.colors.Critical)];
const axis = {gridcolor:'rgba(166,159,199,.12)',zeroline:false,linecolor:'__LINE__',tickfont:{size:11}};
const layout = {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:'Atkinson Hyperlegible, Segoe UI, sans-serif',color:'__TEXT__'},
  margin:{l:44,r:14,t:14,b:30},hovermode:'x unified',hoverlabel:{bgcolor:'#28234F',bordercolor:'__LINE__'},annotations:ann,shapes:base,
  xaxis:Object.assign({type:'date',tickformat:'%a %H:%M',anchor:'y5'},axis)};
F.forEach((f,k)=>{ layout[k===0?'yaxis':'yaxis'+(k+1)] = Object.assign({domain:dom[k],range:f.range,fixedrange:true},axis); });
layout.yaxis5 = Object.assign({domain:dom[4],range:[0,100],fixedrange:true},axis);
Plotly.newPlot('plot', traces, layout, {displayModeBar:false,responsive:true});

// ---------- readouts
const read = document.getElementById('read');
read.innerHTML = F.map((f,k)=>`<div class="cell" id="c${k}"><div class="n">${f.name}</div><div class="v" id="v${k}"></div><div class="u" id="u${k}"></div></div>`).join('')
  + `<div class="cell" id="c4"><div class="n">Combined score</div><div class="v" id="v4"></div><div class="u" id="u4"></div></div>`;

function cols(i0, i1){  // arrays for readings i0..i1 inclusive
  const x = TX.slice(i0, i1+1), out = {x:[], y:[]};
  F.forEach(f=>{ ['hi','lo','typ','v'].forEach(key=>{ out.x.push(x); out.y.push(f[key].slice(i0,i1+1)); }); });
  out.x.push(x); out.y.push(D.h.slice(i0,i1+1));
  return out;
}
function flagged(i0,i1){
  const y=[],c=[];
  for(let i=i0;i<=i1;i++){ const s=D.st[i]; y.push(s==='Normal'?null:D.h[i]); c.push(D.colors[s]); }
  return {x:[TX.slice(i0,i1+1)], y:[y], 'marker.color':[c]};
}
const idx = [...Array(HY+1).keys()];
function draw(i0, i1){
  const c = cols(i0,i1); Plotly.extendTraces('plot', c, idx);
  Plotly.extendTraces('plot', flagged(i0,i1), [FL]);
}
function frame(){
  const t = T[cur], w = winH*H;
  const px = [], py = [];
  F.forEach(f=>{px.push([TX[cur]]);py.push([f.v[cur]]);}); px.push([TX[cur]]); py.push([D.h[cur]]);
  const ev = D.events.filter(e=>e.s<=t).map(e=>({type:'rect',xref:'x',yref:'paper',x0:S(e.s),x1:S(Math.min(e.e,t)),y0:0,y1:1,
     fillcolor:D.colors[e.state],opacity:.12,line:{width:0},layer:'below'}));
  Plotly.update('plot', {x:px,y:py}, {'xaxis.range':[S(t-w), S(t+w*0.04)], shapes: base.concat(ev)}, [PEN,PEN+1,PEN+2,PEN+3,PEN+4]);
  // readout
  const s = D.st[cur], col = D.colors[s];
  document.getElementById('clock').textContent = new Date(t).toLocaleString('en-GB',{timeZone:'UTC',weekday:'short',day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'});
  const chip = document.getElementById('chip'); chip.style.color = col; chip.textContent = '● ' + D.words[s];
  const p = document.getElementById('pulse'); p.style.color = col; p.textContent = 'pulse ' + D.pulse[cur];
  F.forEach((f,k)=>{
    const hot = !f.offline && f.z[cur] >= 2, el = document.getElementById('c'+k);
    el.className = 'cell' + (hot?' hot':''); el.style.color = hot ? D.colors.Watch : '';
    document.getElementById('v'+k).textContent = f.offline ? 'offline' : fmt(f, f.v[cur]);
    document.getElementById('v'+k).style.color = hot ? D.colors.Watch : '__TEXT__';
    document.getElementById('u'+k).textContent = 'usually ' + fmt(f, f.typ[cur]);
  });
  document.getElementById('v4').textContent = Math.round(D.h[cur]); document.getElementById('v4').style.color = col;
  document.getElementById('u4').textContent = 'alert levels ' + Math.round(D.thr.watch) + ' / ' + Math.round(D.thr.confirmed) + ' / ' + Math.round(D.thr.critical);
}
function reset(to){
  cur = Math.max(0, to);
  const blank = {x:idx.map(()=>[]), y:idx.map(()=>[])};
  Plotly.restyle('plot', blank, idx); Plotly.restyle('plot', {x:[[]],y:[[]],'marker.color':[[]]}, [FL]);
  draw(Math.max(0,cur-24*12), cur); frame();
}
reset(cur);

const btn = document.getElementById('play');
const setBtn = ()=> btn.textContent = playing ? '❚❚ Pause' : (cur>=N-1 ? '↻ Replay' : '▶ Play');
btn.onclick = ()=>{ if(cur>=N-1){ reset(D.start-72); playing = true; } else { playing = !playing; } setBtn(); };
document.getElementById('back').onclick = ()=>{ reset(cur-72); };
document.getElementById('win').onchange = e=>{ winH = +e.target.value; frame(); };
setBtn();
const TICK = 80;
setInterval(()=>{
  if(!playing || cur>=N-1) { if(cur>=N-1 && playing){ playing=false; setBtn(); } return; }
  acc += (+document.getElementById('speed').value) * TICK/1000;
  const n = Math.floor(acc); if(n<1) return; acc -= n;
  const to = Math.min(N-1, cur+n); draw(cur+1, to); cur = to; frame();
}, TICK);
</script></body></html>"""


def recorder_html(data: dict) -> str:
    s = TEMPLATE
    for k, v in {"__INK__": ui.INK, "__PANEL__": ui.PANEL, "__LINE__": ui.LINE, "__TEXT__": ui.TEXT,
                 "__MUTED__": ui.MUTED, "__MARI__": ui.MARIGOLD}.items():
        s = s.replace(k, v)
    s = s.replace("__PLOTLY__", os.environ.get("CP_PLOTLY_SRC", "https://cdn.plot.ly/plotly-2.35.2.min.js"))
    return s.replace("__DATA__", json.dumps(data, separators=(",", ":")))
