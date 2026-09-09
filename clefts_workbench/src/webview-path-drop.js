function pathFromDroppedValue(value) {
  const candidate = String(value || '').trim().replace(/^['"]|['"]$/g, '');
  if (!candidate) return '';
  if (/^(?:file|vscode-remote):\/\//i.test(candidate)) {
    try {
      const uri = new URL(candidate);
      let pathname = decodeURIComponent(uri.pathname);
      if (uri.protocol === 'file:' && uri.host && uri.host !== 'localhost') pathname = `//${uri.host}${pathname}`;
      if (/^\/[A-Za-z]:\//.test(pathname)) pathname = pathname.slice(1);
      return pathname;
    } catch (_) { return ''; }
  }
  return /^(?:\/|[A-Za-z]:[\\/]|\\\\)/.test(candidate) ? candidate : '';
}

function pathFromDataTransfer(transfer) {
  if (!transfer) return '';
  const filePath = transfer.files && transfer.files[0] && transfer.files[0].path;
  if (filePath) return String(filePath);
  const read = type => {
    try { return transfer.getData(type) || ''; } catch (_) { return ''; }
  };
  const values = [read('application/vnd.code.uri-list'), read('text/uri-list'), read('text/plain')];
  for (const value of values) {
    for (const line of value.split(/\r?\n/)) {
      if (!line.trim() || line.trim().startsWith('#')) continue;
      const result = pathFromDroppedValue(line);
      if (result) return result;
    }
  }
  return '';
}

function installPathDrop() {
  const inputFor = target => target && target.closest
    ? target.closest('input[data-path-kind]') : null;
  let active;
  document.addEventListener('dragover', event => {
    const input = inputFor(event.target);
    if (!input || input.disabled) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
    if (active && active !== input) active.classList.remove('path-drop-active');
    active = input;
    input.classList.add('path-drop-active');
  });
  document.addEventListener('dragleave', event => {
    const input = inputFor(event.target);
    if (input && !input.contains(event.relatedTarget)) input.classList.remove('path-drop-active');
  });
  document.addEventListener('drop', event => {
    const input = inputFor(event.target);
    if (!input || input.disabled) return;
    event.preventDefault();
    input.classList.remove('path-drop-active');
    active = undefined;
    const value = pathFromDataTransfer(event.dataTransfer);
    if (!value) return;
    input.value = value;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  });
  document.addEventListener('dragend', () => {
    if (active) active.classList.remove('path-drop-active');
    active = undefined;
  });
}

function script() {
  return `${pathFromDroppedValue.toString()}${pathFromDataTransfer.toString()}(${installPathDrop.toString()})();`;
}

function css() {
  return 'input[data-path-kind]{transition:border-color .12s,box-shadow .12s,background-color .12s}input[data-path-kind].path-drop-active{border-color:var(--vscode-focusBorder,#36c5a2)!important;box-shadow:0 0 0 1px var(--vscode-focusBorder,#36c5a2);background:color-mix(in srgb,var(--vscode-focusBorder,#36c5a2) 12%,var(--vscode-input-background))}';
}

module.exports = { pathFromDroppedValue, pathFromDataTransfer, script, css };
