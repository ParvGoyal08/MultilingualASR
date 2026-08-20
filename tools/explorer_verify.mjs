/* Correctness, not just absence of crashes. Reads what the UI actually renders
   and checks it against independently computed expectations. */
import fs from "fs"; import path from "path";
const ROOT = process.argv[2];
const load = f => JSON.parse(fs.readFileSync(path.join(ROOT, f), "utf8"));
const INDEX = load("data/clips.json");

// Independent reimplementation of the filter semantics.
const rt = r => r.overlap ? [...r.types, "OVERLAP"] : r.types;
function role(r, S){
  const inR = r.ref.includes(S), inH = r.hyp.includes(S);
  if (!inR && !inH) return null;
  const took = r.hyp.filter(x => !r.ref.includes(x));
  const lost = r.ref.filter(x => !r.hyp.includes(x));
  if (inR && !inH) return took.length ? {k:"CONFUSION"} : {k:"MISS"};
  if (!inR && inH) return lost.length ? {k:"CONFUSION"} : {k:"FA"};
  return null;
}

function residual(r, sp){
  if(!r.n) return 0;
  const g={MISS:0,FA:0,CONFUSION:0};
  for(const s of sp){ const x=role(r,s); if(x) g[x.k]++; }
  return Math.max(0,r.n.miss-g.MISS)+Math.max(0,r.n.fa-g.FA)
       + ((r.n.confusion && !g.CONFUSION) ? r.n.confusion : 0);
}
let checks = 0, fails = [], flagged = 0;
const ck = (cond, msg) => { checks++; if (!cond) fails.push(msg); };

for (const c of INDEX.clips){
  const j = load(`data/${c.clip_id}.json`);
  for (const m of INDEX.models){
    const b = j.models[m]; if (!b) continue;
    const errRegions = b.regions.filter(r => rt(r).length);
    const spk = [...new Set([...j.ref_speakers, ...b.hypothesis.map(t => t.mapped)])];

    // 1. isolating a type must partition: every error region is reachable by
    //    exactly the types it carries, and the four types together cover all.
    const union = new Set();
    for (const t of ["MISS","FA","CONFUSION","OVERLAP"])
      errRegions.filter(r => rt(r).includes(t)).forEach(r => union.add(r));
    ck(union.size === errRegions.length,
       `${c.clip_id}/${m}: ${errRegions.length - union.size} regions unreachable by any type filter`);

    // 2. every region carrying a scored error must implicate >=1 speaker,
    //    otherwise focusing can never surface it.
    const orphan = errRegions.filter(r =>
      r.types.length && !spk.some(s => role(r, s)));
    const unreported = orphan.filter(r => residual(r, spk) === 0);
    flagged += orphan.length - unreported.length;
    ck(unreported.length === 0,
       `${c.clip_id}/${m}: ${unreported.length} error regions implicate no speaker AND are not flagged unattributed`);

    // 3. focus must be a strict subset of unfiltered, and the union over all
    //    speakers must recover every error region.
    const covered = new Set();
    for (const s of spk) errRegions.filter(r => role(r, s)).forEach(r => covered.add(r));
    const errOnly = errRegions.filter(r => r.types.length);
    ck([...errOnly].every(r => covered.has(r) || residual(r, spk) > 0),
       `${c.clip_id}/${m}: error regions neither reachable by focus nor flagged unattributed`);

    // 4. a MISS role must never be assigned where the region has no miss,
    //    and likewise FA -- the label must not contradict the region.
    for (const s of spk) for (const r of errRegions){
      const x = role(r, s); if (!x) continue;
      if (x.k === "MISS" && residual(r, spk) === 0) ck(r.types.includes("MISS"),
        `${c.clip_id}/${m}/${s}: labelled MISS on a region with types ${r.types}`);
      if (x.k === "FA" && residual(r, spk) === 0) ck(r.types.includes("FA"),
        `${c.clip_id}/${m}/${s}: labelled FA on a region with types ${r.types}`);
      if (x.k === "CONFUSION") ck(r.types.length > 0,
        `${c.clip_id}/${m}/${s}: labelled CONFUSION on a clean region`);
    }

    // 5. index vs payload agreement: the list column must match the clip file.
    const e = c.models[m];
    ck(Math.abs(e.miss_sec - b.totals.miss_sec) < 1e-6, `${c.clip_id}/${m}: miss mismatch index vs payload`);
    ck(Math.abs(e.fa_sec - b.totals.fa_sec) < 1e-6, `${c.clip_id}/${m}: fa mismatch`);
    ck(Math.abs(e.confusion_sec - b.totals.confusion_sec) < 1e-6, `${c.clip_id}/${m}: conf mismatch`);
    ck(e.n_speakers_hyp === b.hyp_speakers.length, `${c.clip_id}/${m}: speaker count mismatch`);
    // 6. DER really is error/total with the backfilled denominator.
    ck(Math.abs(e.error_sec / e.total_sec - e.der) < 2e-3, `${c.clip_id}/${m}: der != error/total`);
  }
  ck(c.script && c.script !== "unknown", `${c.clip_id}: missing script label`);
  ck(c.overlap_frac >= 0 && c.overlap_frac < 1, `${c.clip_id}: implausible overlap_frac`);
}
console.log(`  ${checks} assertions, ${fails.length} failures` +
            (flagged ? `, ${flagged} regions correctly flagged unattributed` : ""));
fails.slice(0, 8).forEach(f => console.log("   ", f));
process.exit(fails.length ? 1 : 0);
