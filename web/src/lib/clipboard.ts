// Copy text to the clipboard. The async Clipboard API only works in a secure
// context (https or localhost); over plain http on a hostname (how the VM serves
// this) `navigator.clipboard` is undefined, so fall back to a hidden-textarea +
// execCommand('copy'), which works there. Returns whether the copy was issued.
export function copyText(text: string): boolean {
  if (window.isSecureContext && navigator.clipboard) {
    navigator.clipboard.writeText(text).catch(() => legacyCopy(text));
    return true;
  }
  return legacyCopy(text);
}

function legacyCopy(text: string): boolean {
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.top = '-1000px';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}
