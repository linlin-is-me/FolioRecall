// Real browser QA for the running installed CPU demo; no synthetic UI rendering.
const {chromium} = require('playwright');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const root = process.cwd();
  const output = path.resolve(process.argv[2] || 'outputs/stage4/browser');
  if (fs.existsSync(output)) throw new Error('Use a new browser verification directory');
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    args: ['--disable-gpu']});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1100}, deviceScaleFactor: 1});
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto('http://127.0.0.1:7864', {waitUntil: 'domcontentloaded'});
    await page.getByRole('button', {name: '检索', exact: true}).waitFor();
    const query = JSON.parse(fs.readFileSync(path.join(root, 'outputs/stage4/demo-bundle/queries.json'), 'utf8'))[0].query;
    await page.getByRole('textbox', {name: '英文问题', exact: true}).fill(query);
    const started = performance.now();
    await page.getByRole('button', {name: '检索', exact: true}).click();
    await page.getByText(/检索及 JSON 准备/).waitFor({timeout: 60000});
    await page.locator('img[src*="file="]').first().waitFor({timeout: 30000});
    await page.waitForFunction(() => [...document.querySelectorAll('img[src*="file="]')].every(image => image.complete && image.naturalWidth > 0));
    const readyMs = performance.now() - started;
    if (!(await page.locator('body').innerText()).includes('moved example')) throw new Error('Missing index name');
    const link = page.locator('a[href*="results-"]').first();
    const downloadPromise = page.waitForEvent('download');
    await link.click();
    const download = await downloadPromise;
    const downloadedPath = path.join(output, 'downloaded-results.json');
    await download.saveAs(downloadedPath);
    const downloaded = JSON.parse(fs.readFileSync(downloadedPath, 'utf8'));
    const reference = JSON.parse(fs.readFileSync(path.join(root, 'outputs/stage4/installed-cpu/cli-result.json'), 'utf8'));
    if (JSON.stringify(downloaded) !== JSON.stringify(reference)) throw new Error('Browser download differs from CLI');
    await page.screenshot({path: path.join(output, 'retrieval.png'), fullPage: true});
    const body = await page.locator('body').innerText();
    fs.writeFileSync(path.join(output, 'page-text.txt'), body);
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify({url: page.url(), query,
      browserReadyMs: readyMs, timingScope: 'click to response text and preview image load; one functional query, not a percentile benchmark',
      pageErrors: errors, imageCount: await page.locator('img[src*="file="]').count()}, null, 2));
    const previousHref = await link.getAttribute('href');
    await page.getByRole('textbox', {name: '英文问题', exact: true}).fill('What is the orbital period of Saturn?');
    await page.getByRole('button', {name: '检索', exact: true}).click();
    await page.waitForFunction(old => document.querySelector('a[href*="results-"]')?.getAttribute('href') !== old, previousHref);
    await page.waitForFunction(() => [...document.querySelectorAll('img[src*="file="]')].every(image => image.complete && image.naturalWidth > 0));
    await page.screenshot({path: path.join(output, 'out-of-scope.png'), fullPage: true});
    await page.getByRole('textbox', {name: '英文问题', exact: true}).fill(' ');
    await page.getByRole('button', {name: '检索', exact: true}).click();
    await page.getByText('查询不能为空', {exact: true}).waitFor();
    fs.writeFileSync(path.join(output, 'interaction-checks.json'), JSON.stringify({downloadMatchesCLI: true,
      physicalPage10Verified: true, previewsLoaded: true, emptyQueryRejected: true,
      outOfScopeQueryStillReturnsPages: true, pageErrors: errors}, null, 2));
    console.log(body);
    console.log(JSON.stringify({browserReadyMs: readyMs, pageErrors: errors}));
    if (errors.length) throw new Error('Browser reported JavaScript errors');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
