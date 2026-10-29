// Static site only. Learning and save data stay inside the downloaded app.
fetch('release.json').then(response => {
  if (!response.ok) throw new Error('Release metadata unavailable');
  return response.json();
}).then(release => {
  if (typeof release.downloadSize === 'string') {
    document.querySelector('#download-size').textContent = `${release.downloadSize} ZIP · CPU + optional CUDA`;
  }
}).catch(() => {}); // Versioned, working download links remain usable without JS.
