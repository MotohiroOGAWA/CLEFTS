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
  // Turn existing pickers into a shared drop surface, preserving their handlers.
  function decorate() {
    for (const button of document.querySelectorAll('button')) {
      if (!/^Browse[.…]*$/i.test(button.textContent.trim())) continue;
      const input = button.parentElement.querySelector('input:not([type=checkbox]):not([type=radio]):not([type=number])');
      if (input && !input.dataset.pathKind) input.dataset.pathKind = /dir|folder|run/i.test(input.name || input.id) ? 'folder' : 'file';
    }
    for (const input of document.querySelectorAll('input[data-path-kind]')) {
      const host = input.closest('.path') || input.parentElement;
      if (!host || host.querySelector('[data-path-drop-zone]')) continue;
      const picker = [...host.querySelectorAll('button')].find(button => /browse/i.test(button.textContent));
      if (!picker) continue;
      picker.dataset.pathDropZone = '';
      picker.classList.add('file-drop-zone');
      picker.textContent = 'Drop ' + (input.dataset.pathKind === 'folder' ? 'a folder' : 'a file') + ' here or click to Browse';
    }
  }
  decorate();
  new MutationObserver(decorate).observe(document.body, {childList:true, subtree:true});
  const inputFor = target => target && target.closest
    ? target.closest('input[data-path-kind]') || target.closest('[data-path-drop-zone]')?.parentElement.querySelector('input[data-path-kind]') : null;
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
  document.addEventListener('drop', async event => {
    const input = inputFor(event.target);
    if (!input || input.disabled) return;
    event.preventDefault();
    input.classList.remove('path-drop-active');
    active = undefined;
    let value = pathFromDataTransfer(event.dataTransfer);
    if (!value && input.dataset.pathKind === 'file' && event.dataTransfer?.files?.[0] && window.uploadDataset) {
      try { value = await window.uploadDataset(event.dataTransfer.files[0]); }
      catch (error) { input.setCustomValidity(error.message); input.reportValidity(); return; }
    }
    if (!value) return;
    input.setCustomValidity('');
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
  return '.file-drop-zone{border:1px dashed var(--vscode-input-border,#888);padding:18px;min-width:180px;white-space:normal}.path:has(.file-drop-zone){flex-wrap:wrap}.path .file-drop-zone{flex-basis:100%}input[data-path-kind]{transition:border-color .12s,box-shadow .12s,background-color .12s}input[data-path-kind].path-drop-active{border-color:var(--vscode-focusBorder,#36c5a2)!important;box-shadow:0 0 0 1px var(--vscode-focusBorder,#36c5a2);background:color-mix(in srgb,var(--vscode-focusBorder,#36c5a2) 12%,var(--vscode-input-background))}';
}

module.exports = { pathFromDroppedValue, pathFromDataTransfer, script, css };
