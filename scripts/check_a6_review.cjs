// Browser interaction checks use an isolated context; never create human ratings.
const {chromium}=require('playwright');
const fs=require('fs'), path=require('path');
(async()=>{
 const out=path.resolve(process.argv[2]);fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe'});
 try{
 const page=await browser.newPage({viewport:{width:1450,height:1050}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));
 await page.goto('http://127.0.0.1:7862');
 await page.locator('.card').first().waitFor();
 if(await page.locator('#task option').count()!==10||await page.locator('.card').count()!==3)throw Error('Incorrect task or candidate count');
 await page.locator('.card select').first().selectOption('0.5');
 await page.reload();
 if(await page.locator('.card select').first().inputValue()!=='0.5')throw Error('Progress lost after reload');
 await page.locator('details').first().click();
 await page.locator('.pages button').first().click();
 await page.locator('dialog[open] img').waitFor();
 await page.waitForFunction(()=>document.querySelector('dialog img').complete&&document.querySelector('dialog img').naturalWidth>0);
 await page.screenshot({path:path.join(out,'original-page.png')});
 await page.getByRole('button',{name:'关闭原页'}).click();
 const download=page.waitForEvent('download');await page.getByRole('button',{name:'导出JSON'}).click();
 const file=path.join(out,'synthetic-draft.json');await (await download).saveAs(file);
 await page.locator('#file').setInputFiles(file);await page.getByText('已导入，原有浏览器进度已替换为该文件。').waitFor();
 if(await page.locator('.card select').first().inputValue()!=='0.5')throw Error('Import changed score');
 const bad=JSON.parse(fs.readFileSync(file));bad.reviews.push(bad.reviews[0]);const badfile=path.join(out,'synthetic-duplicate.json');fs.writeFileSync(badfile,JSON.stringify(bad));
 await page.locator('#file').setInputFiles(badfile);await page.getByText(/导入失败/).waitFor();
 await page.screenshot({path:path.join(out,'review.png'),fullPage:true});
 if(errors.length)throw Error(errors.join('\n'));
 fs.writeFileSync(path.join(out,'result.json'),JSON.stringify({passed:true,tasks:10,anonymous_answers:30,reload_preserved:true,json_roundtrip:true,duplicate_rejected:true,page_zoom_loaded:true,page_errors:errors,human:'not_checked',scope:'Synthetic isolated browser draft, not a human review'},null,2));
 console.log('Review browser checks passed; no human ratings generated.');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
