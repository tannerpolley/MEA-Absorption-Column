# MEA Absorber preview

The browser edits configured requests and presents the retained Henry / Case 3C result. Draft edits hide the original capture and temperature plot; reset restores the reference. No solver runs in this preview. Request exports preserve the original `execution.output_dir`; choose a new, unused directory before executing an exported file with Python.

The displayed prediction retains the molar conversion, `legacy_ratio` vapor composition and `reported_total_wet` gas flow basis. The temperature plot uses 101 sampled heights, distinct from 22 native solver nodes. Numerical, physical and scientific acceptance remain `not_evaluated`. Repeated-call preservation does not establish physical accuracy or mesh independence.

Sources: [original request](../analyses/research_options/results/configuration/json_command_return_final/request.json), [retained attempt](../analyses/research_options/results/configuration/json_command_return_final/attempt.json), and [research notebook](../analyses/research_options/notebook.qmd). Browser copies are in `public/`; [provenance.json](public/provenance.json) identifies the source attempt and its SHA-256. The temperature CSV has an associated provenance JSON download in the result details. The PNG and CSV come from its retained temperature table, without a model rerun. Local [Inter](https://github.com/rsms/inter) and [Tabler icons](https://github.com/tabler/tabler-icons) include licenses in `public/`.

From `preview/`:

```bash
npm install
npm run dev -- --host 127.0.0.1 --port 4175
npm run build
npm run test:sites
```

Run the single [browser check](tests/preview-invariants.js) in the existing open preview evaluator, after the retained result loads:

```js
async () => {
  const { checkPreviewInvariants } = await import('/tests/preview-invariants.js');
  return await checkPreviewInvariants();
}
```

It checks source values, request/result separation, input limits, exact exports, reset, the unavailable live calculation, and the Documentation dialog. It intercepts downloads and restores the original request afterward; it does not launch another browser or execute the model.
