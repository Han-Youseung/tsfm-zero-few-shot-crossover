// Run with the bundled @oai/artifact-tool installed; source data already verified on CPU.
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const out = process.argv[2];
if (!out) throw new Error('Expected writing-kit output directory');
const data = JSON.parse(await fs.readFile(path.join(out, 'tables/workbook_inputs.json'), 'utf8'));
const refs = JSON.parse(await fs.readFile(path.join(out, 'references.json'), 'utf8'));
const wb = Workbook.create();
const sheets = Object.fromEntries(['Summary','Curves','Conditions','Channels','Data','Sources'].map(n=>[n,wb.worksheets.add(n)]));
const col = n => {let s='';for(n++;n>0;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
const near=(a,b)=>{if(typeof a!=='number'||Math.abs(a-b)>1e-11)throw new Error(`Mismatch ${a} != ${b}`);};
function make(name,title,note,headers,rows,widths) {
  const s=sheets[name], last=col(headers.length-1), end=rows.length+4;
  s.showGridLines=false;
  s.getRange(`A1:${last}${end}`).format.font={name:'Arial',size:10,color:'#1F3044'};
  s.getRange(`A1:${last}${end}`).format.rowHeight=23;
  s.getRange(`A1:${last}1`).merge(); s.getRange('A1').values=[[title]];
  s.getRange('A1').format.font={name:'Arial',size:17,bold:true,color:'#123D63'};
  s.getRange('A1').format.rowHeight=32;
  s.getRange(`A2:${last}2`).merge(); s.getRange('A2').values=[[note]];
  s.getRange(`A2:${last}2`).format.wrapText=true;s.getRange('A2').format.rowHeight=44;
  s.getRange(`A4:${last}4`).values=[headers];
  s.getRange(`A4:${last}4`).format.fill='#173D64';
  s.getRange(`A4:${last}4`).format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};
  s.getRange(`A4:${last}4`).format.wrapText=true;s.getRange(`A4:${last}4`).format.rowHeight=34;
  s.getRange(`A5:${last}${end}`).values=rows;
  s.getRange(`A5:${last}${end}`).format.verticalAlignment='center';
  s.getRange(`A5:${last}${end}`).setNumberFormat('0.00000');
  for(let c=0;c<headers.length;c++)s.getRange(`${col(c)}1:${col(c)}${end}`).format.columnWidth=widths[c]||14;
  for(let r=5;r<=end;r+=2)s.getRange(`A${r}:${last}${r}`).format.fill='#F0F5FA';
  s.freezePanes.freezeRows(4);
  s.tables.add(`A4:${last}${end}`,true,`Table${name}`);
  return s;
}
const input = data.conditions;
const ins = make('Conditions','Verified final tests · 270 conditions',
  'Source: evidence/final_results.json. No new model execution. Blank = not applicable or not recorded. Time is not a controlled hardware benchmark.',
  ['Model','Dataset','Seed','Train pool %','NMAE','NMSE','MAE (raw)','MSE (raw)','Selected pool','Steps','Best step','Unique visited','Train sec','Test sec','GPU'],
  input.map(r=>[r.model,r.dataset,r.seed,r.requested_rate,r.normalized_mae,r.normalized_mse,r.mae,r.mse,r.selected_pool,r.actual_optimizer_steps,r.best_validation_step,r.actual_unique_windows,r.train_seconds,r.test_inference_seconds,r.gpu]),
  [12,12,10,12,12,12,12,13,12,10,10,12,14,14,30]);
ins.getRange('C5:C274').setNumberFormat('0');ins.getRange('D5:D274').setNumberFormat('0.0%');
ins.getRange('I5:L274').setNumberFormat('#,##0');ins.getRange('M5:N274').setNumberFormat('#,##0.0');
const idx=new Map(input.map((r,i)=>[[r.model,r.dataset,r.seed,r.requested_rate].join('|'),i+5]));
const curves=data.curves.filter(r=>r.metric==='normalized_mae');
const cs=make('Curves','Paired improvement · all eight fine-tuning rates',
  'Mean/min/max are computed from three paired seed ratios. Not confidence intervals. A positive improvement means lower error. Source cells link to Conditions.',
  ['Model','Dataset','Pool %','ZS mean','FT mean','Mean gain','Min gain','Max gain','Seed 1729','Seed 2718','Seed 31415'],
  curves.map(r=>[r.model,r.dataset,r.rate,null,null,null,null,null,null,null,null]),
  [12,12,10,12,12,12,12,12,13,13,13]);
const formulas=curves.map((r,i)=>{
  const n=i+5, zero=[],few=[],im=[];
  for(const seed of [1729,2718,31415]) {
    const a=idx.get([r.model,r.dataset,seed,0].join('|')),b=idx.get([r.model,r.dataset,seed,r.rate].join('|'));
    if(!a||!b)throw new Error('Missing paired condition');
    zero.push(`Conditions!E${a}`);few.push(`Conditions!E${b}`);
    im.push(`=(${zero.at(-1)}-${few.at(-1)})/${zero.at(-1)}`);
  }
  return [`=AVERAGE(${zero.join(',')})`,`=AVERAGE(${few.join(',')})`,`=AVERAGE(I${n}:K${n})`,`=MIN(I${n}:K${n})`,`=MAX(I${n}:K${n})`,...im];
});
cs.getRange('D5:K84').formulas=formulas;
cs.getRange('C5:C84').setNumberFormat('0.0%');cs.getRange('F5:K84').setNumberFormat('0.00%;[Red]-0.00%;0.00%');
const ends=data.endpoints.filter(r=>r.metric==='normalized_mae');
const ss=make('Summary','Paper table · two models / five dataset variants / H96',
  '270 final tests verified. C–G are formula-linked to Curves. H–I are fixed analysis snapshots, NOT live recalculated boundaries. Unmet ≠ missing. Full pool ≠ convergence.',
  ['Model','Dataset','ZS NMAE','100% NMAE','Mean gain','Seed min','Seed max','Mean sustained','All-seed sustained'],
  ends.map(e=>[e.model,e.dataset,null,null,null,null,null,e.mean_sustained_interval,e.all_seed_sustained_interval]),
  [12,12,12,12,12,12,12,20,34]);
const endFormula=ends.map(e=>{
  const row=curves.findIndex(r=>r.model===e.model&&r.dataset===e.dataset&&r.rate===1)+5;
  return ['D','E','F','G','H'].map(c=>`=Curves!${c}${row}`);
});
ss.getRange('C5:G14').formulas=endFormula;
ss.getRange('E5:G14').setNumberFormat('0.00%;[Red]-0.00%;0.00%');
ss.getRange('H5:I14').format.wrapText=true;ss.getRange('A5:I14').format.rowHeight=39;
const ch=make('Channels','Channel-level endpoint · 100% selected pool',
  'Fixed verified analysis snapshot. Channel-macro primary results can hide heterogeneous channel behavior. Seed ranges are descriptive.',
  ['Model','Dataset','Channel','ZS NMAE','FT NMAE','Mean gain','Seed min','Seed max','Improved seeds'],
  data.channels.map(r=>[r.model,r.dataset,r.channel,r.zero_mean,r.few_mean,r.mean_relative_improvement,r.seed_min_relative_improvement,r.seed_max_relative_improvement,r.improved_seed_count]),
  [12,12,35,12,12,12,12,12,14]);
ch.getRange('F5:H76').setNumberFormat('0.00%;[Red]-0.00%;0.00%');ch.getRange('I5:I76').setNumberFormat('0');
const ds=make('Data','Dataset composition · five variants / three source groups',
  'Sources: tables/datasets.csv; train-only descriptors and original fingerprints in evidence. All eight numeric Tetouan channels are targets, not only three power channels.',
  ['Dataset','Rows','Channels','Interval min','Train rows','Val rows','Test rows','H96 hours','Train windows','Test windows','Lag-1 corr','Trend / std'],
  data.datasets.map(r=>[r.dataset,r.rows,r.channels,r.frequency_minutes,r.train_rows,r.validation_rows,r.test_rows,r.horizon_hours,r.train_candidate_windows,r.test_windows,r.train_lag1_channel_mean,r.train_abs_trend_in_std_channel_mean]),
  [12,12,10,12,12,12,12,12,14,14,12,14]);
ds.getRange('B5:J9').setNumberFormat('#,##0');ds.getRange('K5:L9').setNumberFormat('0.000');
const src=make('Sources','References and provenance',
  'Published paper metadata and official model/software/data sources. Repository access dates are NOT publication years. Full annotations and BibTeX are included alongside this workbook.',
  ['No.','Citation key','Year','Reference title','URL'],
  refs.map(r=>[r.number,r.key,r.year,r.title,r.url]),[7,25,9,75,65]);
src.getRange('A5:A14').setNumberFormat('0');src.getRange('C5:C14').setNumberFormat('0');
src.getRange('D5:E14').format.wrapText=true;src.getRange('A5:E14').format.rowHeight=65;
wb.recalculate();
for(let i=0;i<curves.length;i++) {
  const v=cs.getRange(`D${i+5}:H${i+5}`).values[0],r=curves[i];
  [r.zero_mean,r.few_mean,r.mean_relative_improvement,r.seed_min_relative_improvement,r.seed_max_relative_improvement].forEach((e,j)=>near(v[j],e));
}
for(let i=0;i<ends.length;i++)near(ss.getRange(`E${i+5}`).values[0][0],ends[i].mean_relative_improvement);
// Perturb one imported input, verify dependency recalculation, then restore original exactly.
const target=idx.get([curves[0].model,curves[0].dataset,1729,curves[0].rate].join('|'));
const original=ins.getRange(`E${target}`).values[0][0],before=cs.getRange('F5').values[0][0];
ins.getRange(`E${target}`).values=[[original*1.01]];wb.recalculate();
const after=cs.getRange('F5').values[0][0];if(!(after<before))throw new Error('Formula perturbation failed');
ins.getRange(`E${target}`).values=[[original]];wb.recalculate();near(cs.getRange('F5').values[0][0],before);
const inspect=await wb.inspect({kind:'region',sheetId:'Summary',range:'A4:I14',tableMaxRows:11,tableMaxCols:9,maxChars:6000});
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#N/A',options:{useRegex:true,maxResults:50},maxChars:2500});
await fs.mkdir(path.join(out,'qa'),{recursive:true});
await fs.writeFile(path.join(out,'qa/workbook_inspection.json'),JSON.stringify({summary:inspect,errors},null,2));
const ranges={Summary:'A1:I14',Curves:'A1:K13',Conditions:'A1:O13',Channels:'A1:I13',Data:'A1:L9',Sources:'A1:E8'};
for(const [name,range]of Object.entries(ranges)){
  const image=await wb.render({sheetName:name,range,scale:1.4,format:'png'});
  await fs.writeFile(path.join(out,`qa/sheet-${name}.png`),new Uint8Array(await image.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(wb);await file.save(path.join(out,'paper_tables.xlsx'));
await fs.writeFile(path.join(out,'qa/workbook_checks.json'),JSON.stringify({paired_curve_rows_checked:80,summary_rows_checked:10,formula_perturbation_restored:true,rendered_sheets:Object.keys(sheets)},null,2));
console.log('Exported verified paper_tables.xlsx; all six sheets rendered.');
