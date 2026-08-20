/* Execute the explorer's real JS against the real export in a stub DOM.
   The point is to catch runtime errors -- undefined identifiers, bad property
   access on a missing speaker, filter states that throw -- which no amount of
   reading the diff will find. */
import fs from "fs";
import path from "path";

const ROOT = process.argv[2];
const html = fs.readFileSync(path.join(ROOT, "index.html"), "utf8");

// ------------------------------------------------------------- DOM stubs
const IDS = [...html.matchAll(/id="([^"]+)"/g)].map(m => m[1]);
const mk = (id) => {
  const el = {
    id, _html: "", value: "", textContent: "", style: {setProperty(){}, removeProperty(){}},
    dataset: {}, classList: {_s:new Set(),
      add(x){this._s.add(x)}, remove(x){this._s.delete(x)},
      toggle(x,on){on?this._s.add(x):this._s.delete(x)}, contains(x){return this._s.has(x)}},
    children: [],
    get innerHTML(){ return this._html; },
    set innerHTML(v){ this._html = String(v); },
    querySelector(){ return mk("child"); },
    querySelectorAll(sel){ return collect(this._html, sel); },
    getContext(){ return CTX; },
    getBoundingClientRect(){ return {left:0, top:0, width:1000, height:400}; },
    appendChild(c){ this.children.push(c); },
    focus(){}, play(){ return Promise.resolve(); }, pause(){}, load(){},
    addEventListener(){}, removeEventListener(){},
    clientWidth: 1000, clientHeight: 400, readyState: 1, duration: 1802, currentTime: 0,
  };
  return el;
};
// Elements built from generated innerHTML, so handlers attached to them run.
function collect(htmlStr, sel){
  const attr = sel.match(/\[([a-z-]+)(?:="([^"]*)")?\]/);
  const out = [];
  if (attr){
    const re = new RegExp(`${attr[1]}="([^"]*)"`, "g");
    let m; while ((m = re.exec(htmlStr))){
      if (attr[2] !== undefined && m[1] !== attr[2]) continue;
      const e = mk(""); e.dataset[camel(attr[1].replace(/^data-/,""))] = m[1];
      e.querySelector = () => mk("child"); out.push(e);
    }
  } else if (sel === "th" || sel === "tr" || sel.includes("tr")){
    const n = (htmlStr.match(/<tr/g)||[]).length; for(let i=0;i<n;i++) out.push(mk(""));
  }
  return out;
}
const camel = s => s.replace(/-([a-z])/g, (_,c)=>c.toUpperCase());

const CTX = new Proxy({}, {get:(t,k)=>{
  if(k==="canvas") return {width:1000,height:400};
  if(k==="measureText") return ()=>({width:10});
  if(k==="setLineDash") return ()=>{};
  return typeof k === "string" ? ()=>{} : undefined; },
  set:()=>true});

const store = new Map();
for (const id of IDS) store.set(id, mk(id));
const ERRORS = [];
global.document = {
  querySelector(sel){
    if(sel.startsWith("#")) return store.get(sel.slice(1)) || mk(sel.slice(1));
    if(sel===".grid") return mk("grid");
    return mk("x");
  },
  querySelectorAll(sel){ return collect(html, sel); },
  getElementById(id){ return store.get(id) || mk(id); },
  body:{classList:{add(){},remove(){}}}, addEventListener(){},
};
global.window = global; global.localStorage = {getItem:()=>null,setItem(){},removeItem(){}};
global.devicePixelRatio = 2;
global.requestAnimationFrame = () => 0;
global.addEventListener = () => {};
global.innerWidth = 1600;
global.fetch = async (u) => ({ json: async () =>
  JSON.parse(fs.readFileSync(path.join(ROOT, u), "utf8")) });

// ---------------------------------------------------------------- run it
let js = html.match(/<script>([\s\S]*)<\/script>/)[1];
js = js.replace(/^\s*boot\(\);\s*$/m, "");           // drive boot manually
const mod = new Function(`${js}\nreturn {boot, select, refresh, draw, regionList,
  chipBars, summary, inspect, seek, scriptTable, sorted, render,
  get TYPES(){return TYPES}, set TYPES(v){TYPES=v},
  get FOCUS(){return FOCUS}, set FOCUS(v){FOCUS=v},
  get DATA(){return DATA}, get INDEX(){return INDEX},
  get MODEL(){return MODEL}, set MODEL(v){MODEL=v}};`)();

const step = async (name, fn) => {
  try { await fn(); process.stdout.write(`  ok    ${name}\n`); }
  catch (e) { ERRORS.push([name, e]); process.stdout.write(`  FAIL  ${name}\n        ${e.message}\n`); }
};

await step("boot()", () => mod.boot());
const clips = mod.INDEX.clips.map(c => c.clip_id);
await step(`select first clip`, () => mod.select(clips[0]));

for (const t of ["MISS","FA","CONFUSION","OVERLAP"]){
  await step(`isolate ${t}`, () => { mod.TYPES = new Set([t]); mod.refresh(); });
}
await step("combine MISS+CONFUSION", () => { mod.TYPES = new Set(["MISS","CONFUSION"]); mod.refresh(); });
mod.TYPES = null;

// focus every speaker of every model on a handful of clips, incl. spurious ones
const sample = ["QuA_B6IZ6Ls__61_1863", clips[0], clips[5], clips[20]].filter(x => x && clips.includes(x));
for (const id of sample){
  await step(`clip ${id}: focus every speaker x model`, async () => {
    await mod.select(id);
    for (const m of mod.INDEX.models){
      mod.MODEL = m; mod.refresh();
      const spk = [...new Set([...mod.DATA.ref_speakers,
                   ...mod.DATA.models[m].hypothesis.map(t=>t.mapped)])];
      for (const s of spk){
        mod.FOCUS = s; mod.refresh();
        for (const t of ["MISS","FA","CONFUSION","OVERLAP"]){ mod.TYPES=new Set([t]); mod.refresh(); }
        mod.TYPES = null;
      }
      mod.FOCUS = null;
    }
  });
}

// the crash I patched: focus a speaker, then move to a clip without them
await step("stale focus across clips", async () => {
  await mod.select(sample[0]); mod.FOCUS = mod.DATA.ref_speakers.at(-1); mod.refresh();
  for (const id of clips.slice(0, 12)) { await mod.select(id); }
});
await step("by-script table", () => mod.scriptTable());
await step("seek + inspect", () => { mod.seek(12.5); mod.seek(-5); mod.seek(1e9); });
await step("sweep ALL clips (both models, draw+regions)", async () => {
  for (const id of clips){ await mod.select(id);
    for (const m of mod.INDEX.models){ mod.MODEL=m; mod.refresh(); } }
});
console.log(ERRORS.length ? `\n  ${ERRORS.length} FAILURES` : "\n  all steps clean");
if (ERRORS.length) { console.log(ERRORS[0][1].stack.split("\n").slice(0,6).join("\n")); process.exit(1); }
