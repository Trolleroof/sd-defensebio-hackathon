"use client";
import { useState } from "react";
import { RESULTS } from "@/lib/data";
const VIEWS = ["suite", "pulse", "heat", "load", "process"] as const;
type View = typeof VIEWS[number];
const CRITERIA = ["evidence", "experiment", "feedback", "repro", "selection"] as const;
const LABELS = {suite:"Overall suite",pulse:"Pulse · bpm",heat:"Heat · °C",load:"Load · W",process:"Process review"};
export function Leaderboard() {
 const [view,setView]=useState<View>("suite");
 const rows=[...RESULTS.models].sort((a,b)=>{
  const av=view==="suite"?a.suite_elo:view==="process"?a.reviews:a.tasks[view].elo;
  const bv=view==="suite"?b.suite_elo:view==="process"?b.reviews:b.tasks[view].elo;
  return (bv??-Infinity)-(av??-Infinity)||a.model.localeCompare(b.model);
 });
 return <div>
  <div className="board-bar"><div className="tabs" role="group" aria-label="Leaderboard view">{VIEWS.map(v=><button key={v} aria-pressed={view===v} onClick={()=>setView(v)}>{LABELS[v]}</button>)}</div></div>
  <p className="board-note" style={{marginBottom:20}}>{view==="suite"?"Overall rank requires a valid three-task repeat. No confidence intervals estimated.":view==="process"?"Separate 0–4 criteria · mean (observable count) · 65/72 episode reviews available":"Lower MAE is better. Mean MAE includes valid attempts only; validity is shown alongside it."}</p>
  <div className="results-scroll" tabIndex={0} role="region" aria-label={LABELS[view]}>
   <table className="results-table"><thead><tr><th scope="col">Model</th>{view==="process"? <>{CRITERIA.map(c=><th scope="col" key={c}>{c==="repro"?"Reproducibility":c.charAt(0).toUpperCase()+c.slice(1)}</th>)}<th scope="col">Reviews</th></>:<><th scope="col">Prediction Elo</th><th scope="col">Valid attempts</th>{view!=="suite"&&<th scope="col">Mean MAE</th>}</>}</tr></thead>
    <tbody>{rows.map(r=><tr key={r.model}><th scope="row">{r.model}<small>{r.effort} effort</small></th>{view==="process"?<>{CRITERIA.map(c=><td key={c}>{r.criteria[c].mean===null?"Unobserved":`${r.criteria[c].mean.toFixed(2)} (${r.criteria[c].count})`}</td>)}<td>{r.reviews}/9</td></>:<><td>{(view==="suite"?r.suite_elo:r.tasks[view].elo)?.toFixed(1)??"Unranked"}</td><td>{view==="suite"?`${r.valid}/9`:`${r.tasks[view].valid}/3`}</td>{view!=="suite"&&<td>{r.tasks[view].mean_loss?.toFixed(view==="heat"?3:2)??"Unavailable"}</td>}</>}</tr>)}</tbody>
   </table>
  </div>
  <p className="board-note" style={{marginTop:18}}>Elo is relative to this eight-model pool. Failed attempts remain included in matched outcomes. <a href="/results/compact-coding-v1.json" download>Download scores and coverage ↗</a></p>
 </div>;
}
