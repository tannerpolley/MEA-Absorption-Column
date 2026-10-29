import { useEffect, useRef, useState } from 'react';

function download(name, text, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const Icon = ({ name }) => <img className="icon" src={`/icons/${name}.svg`} alt="" />;

export function App() {
  const [reference, setReference] = useState(null);
  const [attempt, setAttempt] = useState(null);
  const [draft, setDraft] = useState(null);
  const [message, setMessage] = useState('');
  const [loadError, setLoadError] = useState('');
  const [dialog, setDialog] = useState('');
  const dialogRef = useRef(null);
  useEffect(() => {
    let active = true;
    Promise.all(['reference-request.json', 'reference-attempt.json'].map(async path => {
      const response = await fetch(`/${path}`);
      if (!response.ok) throw new Error(`Could not load ${path}: HTTP ${response.status}`);
      return response.json();
    })).then(([request, result]) => {
      if (active) { setReference(request); setDraft(structuredClone(request)); setAttempt(result); }
    }).catch(error => { if (active) setLoadError(error.message); });
    return () => { active = false; };
  }, []);
  useEffect(() => { if (dialog) dialogRef.current?.showModal(); }, [dialog]);
  if (!draft) return <main className="loading" role="status">{loadError || 'Loading retained reference…'}</main>;
  const settings = draft.numerics.solver_settings;
  const matches = JSON.stringify(draft) === JSON.stringify(reference);
  const valid = Number.isInteger(settings.mesh_points) && settings.mesh_points >= 2 &&
    Number.isFinite(settings.tol) && settings.tol > 0 &&
    Number.isFinite(settings.bc_tol) && settings.bc_tol > 0 &&
    Number.isInteger(settings.max_nodes) && settings.max_nodes >= settings.mesh_points;
  const change = (section, field, value) => {
    setDraft(previous => ({ ...previous, [section]: { ...previous[section], [field]: value } }));
    setMessage('');
  };
  const numeric = (field, value) => {
    setDraft(previous => ({ ...previous, numerics: { ...previous.numerics,
      solver_settings: { ...previous.numerics.solver_settings, [field]: value === '' ? '' : Number(value) } } }));
    setMessage('');
  };
  const reset = () => { setDraft(structuredClone(reference)); setMessage('Reference request restored.'); };
  const exportRequest = () => {
    if (!valid) { setMessage('Enter positive tolerances and valid integer node counts.'); return; }
    download('column-request.json', JSON.stringify(draft, null, 2));
    setMessage('Request exported. Choose an unused execution.output_dir before running it in Python.');
  };
  return <>
    <header className="header">
      <div className="brand"><strong>MEA Absorber</strong><span>Research Model</span></div>
      <nav aria-label="Main navigation">
        <button className="active" onClick={() => document.querySelector('#workspace').scrollIntoView()}>Workspace</button>
        <button onClick={() => setDialog('Documentation')}>Documentation</button>
        <button onClick={() => setDialog('About')}>About</button>
      </nav>
    </header>
    <div className="workspace" id="workspace">
      <aside className="configuration">
        <h1>Configuration</h1>
        <p className="intro">Set up the absorber model and prepare a calculation.</p>
        <label>Case<select id="case" value={draft.case.id} onChange={event => change('case', 'id', event.target.value)}>
          {['1C', '2C', '3C', '4C', '5C', '6C', '7C'].map(id => <option key={id}>{id}</option>)}
        </select></label>
        <label>Thermodynamic model<select id="thermo" value={draft.model.thermo_model} onChange={event => change('model', 'thermo_model', event.target.value)}>
          <option value="ideal_henry">Henry</option>
          <option disabled value="epcsaft_neutral">ePC-SAFT neutral — outside this preview</option>
          <option disabled value="epcsaft_ionic">ePC-SAFT ionic — outside this preview</option>
          <option disabled value="epcsaft_reactive_nine">ePC-SAFT nine species — outside this preview</option>
        </select></label>
        <label>Film model<select id="film" value={draft.model.parameters.enhancement_type} onChange={event => {
          setDraft(previous => ({ ...previous, model: { ...previous.model, parameters: { ...previous.model.parameters,
            enhancement_type: event.target.value } } })); setMessage('');
        }}><option value="explicit">Explicit enhancement</option><option value="implicit">Implicit enhancement</option>
          <option disabled value="frozen">Frozen film — requires supplied profile</option>
        </select></label>
        <label>Numerical method<select id="method" value={draft.numerics.method} onChange={event => change('numerics', 'method', event.target.value)}>
          <option value="scipy-bvp">SciPy BVP</option>
          <option disabled value="single">Shooting — outside this preview</option>
          <option disabled value="finite">Finite difference — outside this preview</option>
        </select></label>
        <details className="numerics" open><summary>Numerical settings<Icon name="chevron-down" /></summary>
          <label>Initial nodes<input id="nodes" type="number" min="2" step="1" value={settings.mesh_points} onChange={event => numeric('mesh_points', event.target.value)} /></label>
          <label>Tolerance<input id="tolerance" type="number" min="0" step="any" value={settings.tol} onChange={event => numeric('tol', event.target.value)} /></label>
          <details className="additional"><summary>More settings</summary>
            <label>Boundary tolerance<input id="boundary" type="number" min="0" step="any" value={settings.bc_tol} onChange={event => numeric('bc_tol', event.target.value)} /></label>
            <label>Maximum nodes<input id="max-nodes" type="number" min={settings.mesh_points || 2} step="1" value={settings.max_nodes} onChange={event => numeric('max_nodes', event.target.value)} /></label>
          </details>
        </details>
        <div className="actions">
          <button id="run" className="primary" onClick={() => setMessage(valid ? 'Live calculations are not connected. Export the saved request format for inspection; adapt it to the current Python configuration before execution.' : 'Enter positive tolerances and valid integer node counts.')}><Icon name="player-play" />Run column</button>
          <button id="export-request" className="secondary" disabled={!valid} onClick={exportRequest}><Icon name="download" />Export request</button>
          <button id="reset" className="text-button" onClick={reset}>Reset to reference</button>
        </div>
        {!valid && <p className="input-error" role="alert">Use positive tolerances, at least 2 initial nodes, and a maximum node count at least as large.</p>}
        <p id="message" className="message" role="status">{message}</p>
      </aside>
      <main className="results">
        <div className="banner"><Icon name="info-circle" /><strong>DESIGN PREVIEW — retained result, not a live run</strong><span>Numerical verification, not physical validation.</span></div>
        <div className="result-heading"><p className="eyebrow">{matches ? 'SELECTED RESULT' : 'EDITED REQUEST'}</p><h2>{matches ? 'Retained Henry / Case3C' : `Draft Henry / Case${draft.case.id}`}</h2></div>
        {matches ? <>
          <div className="metrics">
            <div><span>CO₂ capture</span><strong id="capture">{attempt.observables.capture.value.toFixed(3)}%</strong></div>
            <div><span>Native nodes</span><strong id="native-nodes">{attempt.solver.final_mesh_nodes}</strong></div>
            <div><span>Physical acceptance</span><p id="physical-acceptance">{attempt.physical_acceptance.replaceAll('_', ' ')}</p></div>
          </div>
          <section className="profile"><h3>Temperature profile (retained result)</h3><p>Liquid and vapor temperature along the packed height (bottom to top).</p>
            <div className="chart-scroll" tabIndex="0" role="region" aria-label="Temperature plot; scroll horizontally on narrow screens"><img id="temperature-profile" className="chart" src="/temperature_profile.png" alt="Retained Case3C liquid and vapor temperatures in kelvin versus normalized packed height, bottom to top. This is the original request, not an edited calculation." /></div><p className="chart-hint">Scroll the plot horizontally to inspect the full height.</p>
          </section>
          <div className="note"><Icon name="info-circle" /><span>Numerical verification, not physical validation.</span></div>
          <details className="provenance"><summary>Result basis, diagnostics and downloads</summary>
            <p>Original request: {reference.case.source}/{reference.case.id}; molar conversion; legacy_ratio vapor composition; reported_total_wet gas flow. Capture is a percentage of {attempt.observables.capture.basis}.</p>
            <p>Seven-state enthalpy formulation; one bed; Henry thermodynamics; explicit enhancement. The plot contains {attempt.native_profile.sampled_profiles.T.data.length} sampled points, distinct from the {attempt.solver.final_mesh_nodes} native nodes.</p>
            <p>Execution: {attempt.execution.status}. Numerical acceptance: {attempt.numerical_acceptance}. Physical acceptance: {attempt.physical_acceptance}. Scientific acceptance: {attempt.scientific_acceptance}. Observation agreement: {attempt.observation_agreement}.</p>
            <p>Verification here means the configured caller preserved a repeated reference calculation. It does not establish physical accuracy or mesh independence.</p>
            <p className="fingerprint">Physical input SHA-256: {attempt.resolved_inputs.physical_input_sha256}</p>
            <button id="export-result" className="secondary" onClick={() => download('retained-attempt.json', JSON.stringify(attempt, null, 2))}>Export retained result</button>
            <a className="download-link" href="/temperature_profile.csv" download="retained-temperature-profile.csv">Download temperature CSV (K)</a>
            <a className="download-link" href="/provenance.json" download="retained-temperature-provenance.json">Download companion profile provenance</a>
          </details>
        </> : <div className="empty" role="status"><Icon name="info-circle" /><h3>No calculated result for this request</h3><p>The saved Case3C result does not belong to these edited inputs. Export the saved request for inspection, or reset to the reference to inspect the saved profile.</p><button className="secondary" onClick={reset}>Restore reference</button></div>}
      </main>
    </div>
    <dialog ref={dialogRef} onClose={() => setDialog('')} aria-labelledby="dialog-title">
      <h2 id="dialog-title">{dialog}</h2>
      {dialog === 'Documentation' ? <><p>Edit case, film and numerical settings, then export the request as JSON. No solver runs in this browser preview.</p><p>Before executing the exported file, choose a new, unused <code>execution.output_dir</code>; the original retained directory is preserved in the request. The exported request preserves the parked preview’s format, including model.parameters. Adapt it to the current Python configuration before execution; this preview does not convert it.</p><p>The Python configuration resolver owns the final validation and execution. The disabled methods are outside this preview; Engine thermodynamics are available in Python but are outside this static preview.</p><a href="https://github.com/tannerpolley/MEA-Absorption-Column/issues/52" target="_blank" rel="noreferrer">Configuration and capability record</a></> : <><p>A browser workbench for the existing steady absorber model. It presents a retained model prediction with its original input basis; changing inputs requires a new calculation.</p><p>This preview contains no live solver, stripper, tanks or plant dynamics. Numerical preservation of the reference does not imply physical validation. All acceptance assessments remain not evaluated in the saved attempt.</p></>}
      <form method="dialog"><button className="primary">Close</button></form>
    </dialog>
  </>;
}
