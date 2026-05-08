// Copy BibTeX
(function () {
  const btn = document.getElementById('bib-copy');
  const text = document.getElementById('bib-text');
  if (!btn || !text) return;
  btn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(text.innerText);
      const old = btn.textContent;
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => {
        btn.textContent = old;
        btn.classList.remove('copied');
      }, 1400);
    } catch (e) {
      // Fallback: select text
      const range = document.createRange();
      range.selectNodeContents(text);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
  });
})();

// Active section highlight in nav
(function () {
  const links = Array.from(document.querySelectorAll('.nav-links a'));
  const map = new Map();
  links.forEach(a => {
    const id = a.getAttribute('href').replace('#', '');
    const el = document.getElementById(id);
    if (el) map.set(el, a);
  });
  if (!map.size) return;
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      const link = map.get(e.target);
      if (!link) return;
      if (e.isIntersecting) {
        links.forEach(l => l.classList.remove('active'));
        link.classList.add('active');
      }
    });
  }, { rootMargin: '-45% 0px -50% 0px', threshold: 0 });
  map.forEach((_, el) => obs.observe(el));
})();
