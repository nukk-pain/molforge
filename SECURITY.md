# Security Policy

## Supported scope

Security reports are accepted for the current public release snapshot of
`molforge`. The project is research software and should be run in isolated local
or sandboxed environments when processing third-party molecular inputs.

## Do not include secrets in reports

Please do not send API keys, service-account files, private datasets, patient
data, or proprietary compound libraries in a public issue. Redact sensitive
values and share only the minimal reproduction details needed to understand the
problem.

## Reporting

Email: nukkpain@gmail.com

Use the subject prefix `[molforge security]`. Include:

- affected commit or release snapshot;
- reproduction steps using public/example data where possible;
- expected vs. actual behavior;
- impact assessment;
- whether the issue can expose credentials, local files, or private molecular
  data.

## Research-use warning

molforge outputs are computational hypotheses only. Do not use this project for
clinical decisions, prescribing, patient triage, or unreviewed wet-lab actions.
