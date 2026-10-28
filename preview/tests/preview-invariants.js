export async function checkPreviewInvariants() {
  let assertions = 0;
  const assert = (condition, message) => {
    assertions += 1;
    if (!condition) throw new Error(message);
  };
  const element = id => document.getElementById(id);
  const settle = () => new Promise(resolve => setTimeout(resolve, 50));
  const setInput = async (id, value) => {
    const input = element(id);
    const owner = input instanceof HTMLSelectElement ? HTMLSelectElement : HTMLInputElement;
    Object.getOwnPropertyDescriptor(owner.prototype, 'value').set.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
  };
  const [reference, retained] = await Promise.all(
    ['/reference-request.json', '/reference-attempt.json'].map(async path => {
      const response = await fetch(path);
      assert(response.ok, 'Cannot load ' + path);
      return response.json();
    }),
  );
  assert(element('reset'), 'Preview must finish loading before this check runs');
  const downloads = [];
  const createObjectURL = URL.createObjectURL;
  const anchorClick = HTMLAnchorElement.prototype.click;
  URL.createObjectURL = blob => {
    const url = createObjectURL.call(URL, blob);
    downloads.push({ url, blob });
    return url;
  };
  HTMLAnchorElement.prototype.click = function () {
    const download = downloads.find(item => item.url === this.href);
    if (download) download.name = this.download;
  };
  const exportedJSON = async () => JSON.parse(await downloads.at(-1).blob.text());
  const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const reset = async () => { element('reset').click(); await settle(); };
  try {
    await reset();
    assert(element('capture').textContent === '91.483%', 'Original capture must be 91.483%');
    assert(element('capture').textContent === retained.observables.capture.value.toFixed(3) + '%', 'Capture must come from the retained attempt');
    assert(element('native-nodes').textContent === '22', 'Expected 22 native nodes');
    assert(element('physical-acceptance').textContent === 'not evaluated', 'Physical acceptance changed');
    assert(element('temperature-profile'), 'Original temperature plot missing');
    assert(document.querySelectorAll('select').length === 4, 'Expected four native selections');
    assert([...element('thermo').options].slice(1).every(option => option.disabled), 'Unavailable thermodynamics enabled');
    element('export-request').click();
    assert(equal(await exportedJSON(), reference), 'Original export must preserve every request field');

    await setInput('tolerance', '0.25');
    assert(!element('capture') && !element('temperature-profile'), 'Edited inputs display the original computed values');
    assert(document.querySelector('.empty')?.textContent.includes('No calculated result for this request'), 'Draft has no explicit missing-result explanation');
    element('export-request').click();
    const expected = structuredClone(reference);
    expected.numerics.solver_settings.tol = 0.25;
    assert(equal(await exportedJSON(), expected), 'Edited export must preserve the case basis and every unrelated field');
    assert(downloads.at(-1).name === 'column-request.json', 'Request download filename differs');

    for (const [id, value] of [['tolerance', '0'], ['tolerance', ''], ['nodes', '1'], ['nodes', '2.5'], ['max-nodes', '20']]) {
      await reset();
      await setInput(id, value);
      assert(element('export-request').disabled && document.querySelector('.input-error'), 'Invalid numerical value allowed: ' + id + '=' + value);
      const count = downloads.length;
      element('export-request').click();
      assert(downloads.length === count, 'Invalid request was exported');
    }
    await reset();
    await setInput('case', '1C');
    assert(!element('capture') && !element('temperature-profile'), 'Case change relabels original result');
    await reset();
    await setInput('film', 'implicit');
    assert(!element('capture') && !element('temperature-profile'), 'Film change relabels original result');
    element('run').click();
    await settle();
    assert(element('message').textContent.includes('Live calculations are not connected'), 'Run omits the unavailable live-calculation explanation');
    assert(!element('capture') && !element('temperature-profile'), 'Run fabricates draft results');

    await reset();
    assert(element('capture')?.textContent === '91.483%' && element('temperature-profile'), 'Reset does not restore original results');
    element('export-result').click();
    assert(equal(await exportedJSON(), retained), 'Retained result export differs from its original attempt');
    assert(downloads.at(-1).name === 'retained-attempt.json', 'Result download filename differs');
    assert(document.querySelector('a[href="/temperature_profile.csv"][download]'), 'Retained temperature download missing');
    assert(document.querySelector('a[href="/provenance.json"][download="retained-temperature-provenance.json"]'), 'Temperature provenance companion download missing');
    const documentation = [...document.querySelectorAll('nav button')].find(button => button.textContent === 'Documentation');
    documentation.click();
    await settle();
    assert(document.querySelector('dialog').open && element('dialog-title').textContent === 'Documentation', 'Documentation dialog does not open');
    assert(document.querySelector('dialog').textContent.includes('execution.output_dir'), 'Documentation omits retained output-directory guidance');
    document.querySelector('dialog form button').click();
    await settle();
    assert(!document.querySelector('dialog').open, 'Documentation dialog does not close');
    return { passed: true, assertions, downloadsChecked: downloads.length, capture: '91.483%', nativeNodes: 22 };
  } finally {
    URL.createObjectURL = createObjectURL;
    HTMLAnchorElement.prototype.click = anchorClick;
    if (document.querySelector('dialog')?.open) document.querySelector('dialog').close();
    await reset();
  }
}
