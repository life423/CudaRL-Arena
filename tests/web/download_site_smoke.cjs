const {chromium}=require('playwright');const fs=require('fs');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CUDARL_CHROME ? {executablePath:process.env.CUDARL_CHROME} : {})});
 fs.mkdirSync('build/site-qa',{recursive:true});
 const page=await browser.newPage();let checks=0;const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const check=(ok,message)=>{checks++;if(!ok)throw Error(message)};
 for(const [width,height] of [[1440,1000],[1280,800],[768,1024],[390,844],[320,760]]){
  await page.setViewportSize({width,height});await page.goto(process.env.CUDARL_SITE_URL || 'http://127.0.0.1:4173',{waitUntil:'networkidle'});
  check(await page.locator('link[rel=canonical]').getAttribute('href')==='https://cudarl.drewclark.io/','Canonical domain wrong');
  check((await page.locator('.download-link').evaluateAll(links=>links.every(link=>link.href.endsWith('/v0.1.0-preview.2/TadpolePondSetup.exe')))),'Player download is not the setup executable');
  check(!(await page.locator('.getting-started').textContent()).match(/extract|ZIP/i),'Player flow still requires archive extraction');
  check(await page.locator('#learning-panel').isHidden(),'Inspector should start collapsed');
  const button=page.locator('#learning-toggle');const first=await button.boundingBox(),second=await page.locator('.source-button').boundingBox();
  check(Math.abs(first.width-second.width)<0.5 && Math.abs(first.height-second.height)<0.5,'Learning/source buttons differ in size');
  await button.click();await page.waitForTimeout(250);
  check(await page.locator('#learning-panel').isVisible(),'Explanation button failed');
  check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Expanded horizontal overflow '+width);
  check(await page.locator('.stage-tabs button').count()===4,'Learning stages missing');
  await page.locator('#learning-tab-0').focus();await page.keyboard.press('ArrowRight');
  check(await page.locator('#learning-tab-1').getAttribute('aria-selected')==='true','Keyboard stage navigation failed');
  check((await page.locator('#stage-title').textContent()).includes('Choose a goal'),'Explanation did not change');
  await page.keyboard.press('End');check(await page.locator('#learning-tab-3').getAttribute('aria-selected')==='true','End key failed');
  check((await page.locator('#stage-copy').textContent()).includes('compared'),'Validation explanation missing');
  await page.locator('input[value=cpu]').check();check(await page.locator('#practice-count').textContent()==='16','CPU path is wrong');
  await page.locator('input[value=cuda]').check();check(await page.locator('#practice-count').textContent()==='1,024','CUDA path is wrong');
  await page.locator('#inside').screenshot({path:`build/site-qa/inspector-${width}.png`});
  await button.click();check(await page.locator('#learning-panel').isHidden(),'Explanation did not close');
  check(await button.getAttribute('aria-expanded')==='false','Disclosure accessibility state is stale');
  check(await page.locator('.questions').evaluate(el=>getComputedStyle(el).display==='block'),'FAQ keeps the empty sidebar layout');
  await page.screenshot({path:`build/site-qa/revised-${width}.png`,fullPage:true});
 }
 await page.emulateMedia({reducedMotion:'reduce'});await page.locator('#learning-toggle').click();
 check(await page.locator('#learning-panel').evaluate(el=>el.getAnimations().length===0),'Reduced motion ignored');
 check(errors.length===0,errors.join(';'));
 console.log(JSON.stringify({checks,errors,result:'PASS'},null,2));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
