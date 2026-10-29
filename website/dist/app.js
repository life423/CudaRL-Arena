// Static site only. Learning and save data stay inside the downloaded app.
fetch('release.json').then(response => {
  if (!response.ok) throw new Error('Release metadata unavailable');
  return response.json();
}).then(release => {
  if (typeof release.downloadSize === 'string') {
    document.querySelector('#download-size').textContent = `${release.downloadSize} ZIP · CPU + optional CUDA`;
  }
}).catch(() => {}); // Versioned, working download links remain usable without JS.

// A walkthrough of the native application, never a simulated browser AI.
const learningToggle = document.querySelector('#learning-toggle');
const learningPanel = document.querySelector('#learning-panel');
const stageTabs = [...document.querySelectorAll('[data-stage]')];
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const stages = [
  {
    label: '01 / OBSERVE', title: 'Start with what the creature can sense.',
    copy: 'The controllers receive numerical observations of food and body state. The swimming controller also receives target motion, so sinking and drifting food can influence its actions.',
    detail: 'These are observations from the native physical world. The browser diagram itself does not run a neural policy.', active: 0
  },
  {
    label: '02 / SELECT', title: 'Choose a goal—not a scripted route.',
    copy: 'The goal controller scores active pellets and chooses a target. It starts with learned food-selection weights, which stay fixed in this preview while the swimming controller practices.',
    detail: 'Choosing a target does not move the creature. It still has to produce the right physical movement to reach it.', active: 1
  },
  {
    label: '03 / ACT', title: 'Turn observations into tail commands.',
    copy: 'The body controller produces four tail-joint commands. Articulated-body dynamics, buoyancy and simplified water drag determine what happens next. The swimmer is not moved along a preset animation path.',
    detail: 'The same physical equations run in the CUDA and CPU versions. Rendering is separate from the learning backend.', active: 2
  },
  {
    label: '04 / PRACTICE', title: 'Practice a copy. Check it. Keep a real improvement.',
    copy: 'Background worlds practice your recent feeding locations and current setting using PPO. Candidate motor weights are compared with the working controller and checked for retention of calm-water feeding before they can reach the visible creature.',
    detail: 'PPO updates and adopted updates are different. A candidate can be rejected; improvement after every interaction is not guaranteed.', active: 3
  }
];
function selectStage(index, moveFocus = false) {
  const stage = stages[index];
  stageTabs.forEach((tab, i) => {
    tab.setAttribute('aria-selected', String(i === index));
    tab.tabIndex = i === index ? 0 : -1;
  });
  document.querySelector('#learning-stage').setAttribute('aria-labelledby', `learning-tab-${index}`);
  document.querySelector('#stage-label').textContent = stage.label;
  document.querySelector('#stage-title').textContent = stage.title;
  document.querySelector('#stage-copy').textContent = stage.copy;
  document.querySelector('#stage-detail').textContent = stage.detail;
  document.querySelectorAll('[data-flow]').forEach(node => node.classList.toggle('is-active', Number(node.dataset.flow) === stage.active));
  if (moveFocus) stageTabs[index].focus();
}
if (learningToggle && learningPanel) {
  learningToggle.addEventListener('click', () => {
    const open = learningPanel.hidden;
    learningPanel.hidden = !open;
    learningToggle.setAttribute('aria-expanded', String(open));
    if (open && !reducedMotion.matches) {
      learningPanel.animate([{ opacity: 0, transform: 'translateY(8px)' }, { opacity: 1, transform: 'translateY(0)' }], { duration: 180, easing: 'ease-out' });
    }
    if (open) learningPanel.scrollIntoView({ block: 'nearest', behavior: reducedMotion.matches ? 'auto' : 'smooth' });
  });
  stageTabs.forEach((tab, index) => {
    tab.addEventListener('click', () => selectStage(index));
    tab.addEventListener('keydown', event => {
      let next = index;
      if (event.key === 'ArrowRight') next = (index + 1) % stages.length;
      else if (event.key === 'ArrowLeft') next = (index + stages.length - 1) % stages.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = stages.length - 1;
      else return;
      event.preventDefault();
      selectStage(next, true);
    });
  });
  document.querySelectorAll('input[name="learning-backend"]').forEach(input => input.addEventListener('change', () => {
    const cpu = input.value === 'cpu';
    document.querySelector('#practice-count').textContent = cpu ? '16' : '1,024';
    document.querySelector('#backend-note').textContent = cpu
      ? 'Without usable CUDA, native CPU PPO practices the same task with a smaller population. Speed and learning outcomes vary with the computer.'
      : 'The app prefers CUDA after a device-and-kernel check. Practice stays on your GPU.';
  }));
  selectStage(0);
}
