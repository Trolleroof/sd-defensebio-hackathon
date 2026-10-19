import results from "./results.json";
export const RESULTS = results;
export const DASHBOARD_URL = process.env.NEXT_PUBLIC_DASHBOARD_URL ?? "http://127.0.0.1:8765";
export const RELEASE = {version: "compact-coding-v1", status: "Retrospective pilot", episodes: results.episodes, updated: "October 18, 2026"};
export const TASKS = [
 {id:"pulse",name:"Motion-robust pulse estimation",family:"Causal estimation",data:"GalaxyPPG · 23 people",budget:"13 / 5 / 5 training / development / test people",summary:"Estimate trailing eight-second heart rate every two seconds from wrist PPG and acceleration. ECG-derived references remain provisional.",phases:["Train","Evaluate","Select"]},
 {id:"heat",name:"Core-temperature forecasting",family:"Forecasting",data:"CORE cycling · 24 people",budget:"14 / 5 / 5 training / development / test people",summary:"Forecast measured rectal temperature at +5, +10 and +20 minutes from causal sensor history and demographics. Error is reported in °C.",phases:["Train","Evaluate","Select"]},
 {id:"load",name:"Load-carriage metabolic cost",family:"Unseen-person prediction",data:"EVOBREATH · 48 people",budget:"28 / 10 / 10 training / development / test people",summary:"Predict gross metabolic power in watts for level walking at 4 km/h with 0, 5, 10 or 15 kg load. All 189 available measurements are preserved.",phases:["Train","Evaluate","Select"]},
];
export const POST_SECTIONS = [
 {id:"introduction",n:"01",title:"Introduction",sub:"What the pilot tests"},
 {id:"environment",n:"02",title:"The environment",sub:"Three measured-data tasks"},
 {id:"methodology",n:"03",title:"Methodology",sub:"Budget, trials and scoring"},
 {id:"results",n:"04",title:"Results",sub:"Sealed outcomes by model"},
 {id:"failures",n:"05",title:"Failure analysis",sub:"Validity and protocol failures"},
 {id:"limitations",n:"06",title:"Limitations",sub:"Coverage and known gaps"},
];
