// Actual browser clicks. Mock records remain separate from paid functional cases.
const {chromium}=require('playwright');
const fs=require('fs'),path=require('path');
const root=path.resolve(process.argv[2]), phase=process.argv[3]||'mock';
const port=phase==='real'?7864:7865;
const demo=path.resolve('outputs/releases/v0.1.0-candidate/demo-bundle');
const cases=JSON.parse(fs.readFileSync(path.join(demo,'qa-examples.json')));
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 fs.mkdirSync(path.join(root,'browser'),{recursive:true});
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe'});
 const context=await browser.newContext({viewport:{width:1440,height:1100},permissions:['clipboard-read','clipboard-write']});
 const page=await context.newPage(),errors=[],records=[];
 page.on('pageerror',e=>errors.push(String(e)));
 try{
  await page.goto('http://127.0.0.1:'+port);
  await page.getByRole('tab',{name:'文档问答',exact:true}).click();
  let lastEnd=0;
  const count=phase==='real'?cases.length:2;
  for(let i=0;i<count;i++){
   if(phase==='real'&&lastEnd){const deadline=lastEnd+65000;while(performance.now()<deadline)await sleep(Math.min(500,deadline-performance.now()));}
   if(i===2||i===4)await page.getByRole('button',{name:'新建会话',exact:true}).click();
   await page.getByRole('textbox',{name:'英文或中文问题',exact:true}).fill(cases[i].question);
   await page.getByRole('radio',{name:cases[i].mode,exact:true}).check();
   const before=new Set(fs.readdirSync(root).filter(n=>/^qa-.*\.json$/.test(n)));
   await page.getByRole('button',{name:'向百炼提交问答',exact:true}).click();
   await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='向百炼提交问答'&&b.disabled));
   await page.locator('#agent-progress').waitFor();
   const earlySteps=await page.locator('#progress-panel').innerText();
   const earlyHasTerminal=earlySteps.includes('任务结束');
   await page.screenshot({path:path.join(root,'browser',`turn${i+1}-running.png`)});
   let file;
   const deadline=performance.now()+240000;
   while(performance.now()<deadline){file=fs.readdirSync(root).find(n=>/^qa-.*\.json$/.test(n)&&!before.has(n));if(file)break;await sleep(300)}
   if(!file)throw Error('No terminal record; inspect process before repeating a submission');
   const result=JSON.parse(fs.readFileSync(path.join(root,file)));
   await page.getByRole('button',{name:'向百炼提交问答',exact:true}).waitFor();
   await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='向百炼提交问答'&&!b.disabled));
   await page.locator('#answer-panel h3').waitFor();
   await page.getByText('完整JSON、工具记录与费用',{exact:true}).click();
   const copy=page.locator('#qa-json').getByRole('button',{name:/copy|复制/i});
   await copy.click();
   const shown=JSON.parse(await page.evaluate(()=>navigator.clipboard.readText()));
   if(JSON.stringify(shown)!==JSON.stringify(result))throw Error('Saved and displayed JSON differ');
   await page.getByText('完整JSON、工具记录与费用',{exact:true}).click();
   const links=page.locator('#answer-panel a[href^="#qa-page-"]');
   let citation=null;
   if(await links.count()){
    const href=await links.first().getAttribute('href');await links.first().click();
    citation=await page.evaluate(h=>{const e=document.getElementById(decodeURIComponent(h.slice(1)));return !!e&&e.querySelector('img')?.naturalWidth>0},href);
    if(!citation)throw Error('Citation preview not resolved');
   }
   await page.locator('#conversation').scrollIntoViewIfNeeded();
   await page.screenshot({path:path.join(root,'browser',`turn${i+1}.png`)});
   records.push({case:i+1,question:cases[i].question,mode:cases[i].mode,file,
    completed:result.completed,termination_reason:result.termination_reason,conversation_id:result.conversation_id,
    turn_index:result.turn_index,context_turns:result.context_turns,live_progress_observed:!earlyHasTerminal,
    progress_events:result.progress.length,citation_checked:citation,json_matches:true,
    known_cny:result.estimated_cost_cny,claims:result.answer.claims.length});
   fs.writeFileSync(path.join(root,'browser/result.json'),JSON.stringify({phase,records,page_errors:errors},null,2));
   console.log('case',i+1,result.termination_reason,'context',result.context_turns);
   lastEnd=performance.now();
  }
  if(records[1].conversation_id!==records[0].conversation_id||records[1].context_turns!==1)throw Error('Follow-up context missing');
  if(phase==='real'&&(records[2].conversation_id===records[1].conversation_id||records[4].context_turns!==0))throw Error('Reset leaked conversation');
  if(phase==='mock'){
   const isolated=await context.browser().newContext(),other=await isolated.newPage();
   await other.goto('http://127.0.0.1:'+port);
   await other.getByRole('tab',{name:'文档问答',exact:true}).click();
   if(await other.locator('#conversation').innerText().then(s=>s.includes(cases[0].question)))throw Error('Cross-session transcript leak');
   await isolated.close();
   await page.getByRole('button',{name:'新建会话',exact:true}).click();
   await page.waitForFunction(()=>!document.getElementById('conversation').innerText.includes('Which figure'));
  }
  if(errors.length)throw Error(errors.join('\n'));
  fs.writeFileSync(path.join(root,'browser/complete.json'),JSON.stringify({passed:true,phase,records,page_errors:errors},null,2));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
