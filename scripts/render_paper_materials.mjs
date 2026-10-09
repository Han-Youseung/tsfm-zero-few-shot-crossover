// CPU-only scientific figure export and HTML visual QA. No Colab/browser session changes.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import sharp from 'sharp';
import {chromium} from 'playwright';

const out=process.argv[2];
if(!out)throw new Error('Expected output directory');
const data=JSON.parse(await fs.readFile(path.join(out,'tables/workbook_inputs.json'),'utf8'));
const datasets=data.datasets.map(r=>r.dataset), fig=path.join(out,'figures');
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="855" viewBox="0 0 1060 855"><rect width="1060" height="855" fill="white"/>'];
for(let i=0;i<datasets.length;i++){
  const file=path.join(fig,`relative-${datasets[i]}.svg`),src=await fs.readFile(file,'utf8');
  const inner=src.slice(src.indexOf('>')+1,src.lastIndexOf('</svg>'));
  parts.push(`<g transform="translate(${(i%2)*530},${Math.floor(i/2)*280})">${inner}</g>`);
  await sharp(Buffer.from(src)).resize({width:1590}).withMetadata({density:300}).png().toFile(file.replace('.svg','.png'));
}
parts.push('<g transform="translate(580,600)" font-family="Arial,sans-serif" font-size="16" fill="#243743">');
for(const [i,t]of ['Two fixed model versions / H96','Line: mean paired relative improvement','Band: three-seed min–max (not CI)','Positive values: improvement over Zero-Shot','Panel y-axis ranges differ','100% pool does not imply full traversal'].entries())parts.push(`<text x="0" y="${i*28}">${esc(t)}</text>`);
parts.push('</g></svg>');
await fs.writeFile(path.join(fig,'relative-all.svg'),parts.join(''));
await sharp(Buffer.from(parts.join(''))).resize({width:3180}).withMetadata({density:300}).png().toFile(path.join(fig,'relative-all.png'));
const main=data.endpoints.filter(r=>r.metric==='normalized_mae');
const y=v=>355-v/0.6*260,svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="425" viewBox="0 0 1060 425"><rect width="1060" height="425" fill="white"/><g font-family="Arial,sans-serif" fill="#243743"><text x="65" y="26" font-size="20" font-weight="bold">Absolute normalized MAE · three-seed mean</text><text x="65" y="53" font-size="13">Lower is better. Endpoint: 100% candidate pool, not convergence.</text>'];
for(let t=0;t<=6;t++){
  const v=t/10;svg.push(`<line x1="65" x2="1030" y1="${y(v)}" y2="${y(v)}" stroke="#DCE3EA"/><text x="52" y="${y(v)+4}" text-anchor="end" font-size="13">${v.toFixed(1)}</text>`);
}
const series=[['ttm','zero_mean','#2166ac','TTM ZS'],['ttm','few_mean','#2166ac','TTM 100%'],['moirai1','zero_mean','#b34d20','MOIRAI ZS'],['moirai1','few_mean','#b34d20','MOIRAI 100%']];
for(let j=0;j<4;j++){
  const [model,key,color,label]=series[j],xx=420+j*151;
  svg.push(`<rect x="${xx}" y="67" width="16" height="13" fill="${key==='zero_mean'?'white':color}" stroke="${color}" stroke-width="2"/><text x="${xx+23}" y="79" font-size="12">${label}</text>`);
  for(let i=0;i<5;i++){
    const row=main.find(r=>r.model===model&&r.dataset===datasets[i]),v=row[key],x=90+i*193+j*36;
    svg.push(`<rect x="${x}" y="${y(v)}" width="27" height="${355-y(v)}" fill="${key==='zero_mean'?'white':color}" stroke="${color}" stroke-width="1.5"/><text x="${x+13.5}" y="${y(v)-7}" text-anchor="middle" font-size="11">${v.toFixed(3)}</text>`);
  }
}
for(let i=0;i<5;i++)svg.push(`<text x="${157+i*193}" y="382" text-anchor="middle" font-size="16">${datasets[i]}</text>`);
svg.push('<text x="65" y="413" font-size="12">Training-std normalization; channel macro. No error bars; see paired seed tables for variation.</text></g></svg>');
await fs.writeFile(path.join(fig,'absolute-nmae.svg'),svg.join(''));
await sharp(Buffer.from(svg.join(''))).resize({width:3180}).withMetadata({density:300}).png().toFile(path.join(fig,'absolute-nmae.png'));
let index=await fs.readFile(path.join(out,'index.html'),'utf8');
const block='<h2>다운로드용 통합 그림</h2><p><a href="figures/relative-all.png">상대 개선 5패널 PNG</a> · <a href="figures/relative-all.svg">벡터 SVG</a></p><img src="figures/absolute-nmae.svg" alt="절대 normalized MAE 비교"><p><a href="figures/absolute-nmae.png">절대 오차 PNG</a> · <a href="figures/absolute-nmae.svg">벡터 SVG</a></p>';
if(!index.includes('다운로드용 통합 그림'))index=index.replace('</main>',block+'</main>');
await fs.writeFile(path.join(out,'index.html'),index);
await fs.mkdir(path.join(out,'qa'),{recursive:true});
const browser=await chromium.launch({headless:true,channel:'msedge'});
const checks=[];
try{
  const page=await browser.newPage({viewport:{width:1360,height:1080},deviceScaleFactor:1});
  for(const name of ['index.html','paper_draft_ko.html']){
    await page.goto(pathToFileURL(path.resolve(out,name)).href);
    await page.screenshot({path:path.join(out,'qa',name.replace('.html','-top.png'))});
    const result=await page.evaluate(()=>({title:document.title,overflow:document.documentElement.scrollWidth>innerWidth,unresolved:document.body.innerText.includes('{{RESULTS}}'),brokenImages:[...document.images].filter(i=>!i.complete||i.naturalWidth===0).length}));
    if(result.overflow||result.unresolved||result.brokenImages)throw new Error(JSON.stringify(result));
    checks.push({name,...result});
  }
  await page.setViewportSize({width:390,height:844});
  await page.goto(pathToFileURL(path.resolve(out,'index.html')).href);
  if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw new Error('Mobile overflow');
  checks.push({mobile_overflow:false,figures:7,png_density:300});
}finally{await browser.close();}
await fs.writeFile(path.join(out,'qa/render_checks.json'),JSON.stringify(checks,null,2));
console.log('Seven vector/PNG figures and two HTML views verified.');
