// Six-use-case A6 includes two CLI runs; this checks four real browser submissions.
const {chromium}=require('playwright');
const fs=require('fs'),path=require('path');
const root=path.resolve(process.argv[2]), repo=process.cwd();
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const tasks=new Map(fs.readFileSync(path.join(repo,'outputs/extensions/A/data/jrc-tasks-v2/tasks.jsonl'),'utf8').trim().split('\n').map(s=>{const t=JSON.parse(s);return[t.task_id,t]}));
const targets=[['jrc-09','en','rag'],['jrc-22','zh','agent'],['jrc-40','en','agent'],['jrc-39','zh','rag']];
(async()=>{
 fs.mkdirSync(path.join(root,'browser'),{recursive:true});
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe'});
 const context=await browser.newContext({viewport:{width:1440,height:1100},permissions:['clipboard-read','clipboard-write']});
 const page=await context.newPage(), errors=[], records=[];
 page.on('pageerror',e=>errors.push(String(e)));
 try{
 await page.goto('http://127.0.0.1:7861');
 await page.getByRole('textbox',{name:'英文问题',exact:true}).fill(tasks.get('jrc-02').question_en);
 await page.getByRole('button',{name:'检索',exact:true}).click();
 await page.getByText(/检索及 JSON 准备/).waitFor({timeout:90000});
 const dl=page.waitForEvent('download');await page.locator('a[href*="results-"]').first().click();
 const refpath=path.join(root,'browser','retrieval-download.json');await(await dl).saveAs(refpath);
 const reference=JSON.parse(fs.readFileSync(path.join(root,'retrieval-reference.json')));
 if(JSON.stringify(JSON.parse(fs.readFileSync(refpath)))!==JSON.stringify(reference))throw Error('CLI/browser retrieval JSON differs');
 await page.screenshot({path:path.join(root,'browser','retrieval.png'),fullPage:true});
 await page.getByRole('tab',{name:'文档问答',exact:true}).click();
 let previousEnd=performance.now();
 const cli=JSON.parse(fs.readFileSync(path.join(root,'cli-processes.json')));
 let firstDelay=Math.max(0,65000-(Date.now()-cli[cli.length-1].finished_at*1000));
 for(let i=0;i<targets.length;i++){
  if(i) {const deadline=previousEnd+65000;while(performance.now()<deadline)await delay(Math.min(1000,deadline-performance.now()));}
  else if(firstDelay)await delay(firstDelay);
  const [tid,lang,mode]=targets[i], before=new Set(fs.readdirSync(path.join(root,'ui-session')).filter(n=>/^qa-.*\.json$/.test(n)));
  await page.getByRole('textbox',{name:'英文或中文问题',exact:true}).fill(tasks.get(tid)['question_'+lang]);
  await page.getByRole('radio',{name:mode,exact:true}).check();
  const started=performance.now(), startedAt=new Date().toISOString();
  await page.getByRole('button',{name:'向百炼提交问答',exact:true}).click();
  let file;
  const deadline=performance.now()+360000;
  while(performance.now()<deadline){file=fs.readdirSync(path.join(root,'ui-session')).find(n=>/^qa-.*\.json$/.test(n)&&!before.has(n));if(file)break;await delay(500)}
  if(!file)throw Error('No terminal UI record; inspect live process and ledger before another submission');
  const result=JSON.parse(fs.readFileSync(path.join(root,'ui-session',file)));
  await page.getByRole('heading',{name:'回答与引用',exact:true}).waitFor({timeout:20000});
  for(const c of result.answer.claims)await page.getByText(c.text,{exact:false}).first().waitFor({timeout:20000});
  console.log('buttons',JSON.stringify(await page.locator('button').evaluateAll(es=>es.map(e=>({name:e.getAttribute('aria-label'),title:e.getAttribute('title'),text:e.textContent.trim()})).filter(x=>/copy|复制/i.test(JSON.stringify(x))))));
  const copy=page.getByRole('button',{name:/copy|复制/i});
  if(await copy.count()!==1)throw Error('Expected one JSON copy control; terminal answer retained');
  await copy.click();const displayed=JSON.parse(await page.evaluate(()=>navigator.clipboard.readText()));
  if(JSON.stringify(displayed)!==JSON.stringify(result))throw Error('UI copied JSON differs from saved result');
  const links=page.locator('a[href^="#qa-page-"]');
  let citation=false;
  if(await links.count()){
   const href=await links.first().getAttribute('href');await links.first().click();
   citation=await page.evaluate(h=>{const target=document.getElementById(decodeURIComponent(h.slice(1)));return !!target&&target.querySelector('img')?.naturalWidth>0},href);
   if(!citation)throw Error('Citation did not resolve to loaded original page');
  }
  await page.screenshot({path:path.join(root,'browser',tid+'-'+lang+'.png'),fullPage:true});
  const record={task_id:tid,language:lang,mode,started_at:startedAt,saved_result:file,completed:result.completed,termination_reason:result.termination_reason,clipboard_matches_saved:true,citation_checked:citation,claims:result.answer.claims.length,browser_response_ms:performance.now()-started,timing_scope:'Actual click through response, JSON copying and citation/image verification; not percentile latency'};
  records.push(record);fs.writeFileSync(path.join(root,'browser','results.json'),JSON.stringify({records,page_errors:errors},null,2));
  console.log(tid,lang,result.termination_reason);previousEnd=performance.now();
 }
 if(errors.length)throw Error(errors.join('\n'));
 console.log('Four real UI submissions verified.');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
