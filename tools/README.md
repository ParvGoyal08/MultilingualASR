# Explorer regression harnesses

The error explorer is a single HTML file with no build step and no test
framework, so these two node scripts are its test suite. Both take an export
directory as their only argument and need nothing installed.

    node tools/explorer_smoke.mjs  <export_dir>   # does it run?
    node tools/explorer_verify.mjs <export_dir>   # is it right?

`explorer_smoke.mjs` executes the page's real JavaScript against a stub DOM and
drives it: boot, select every clip, switch models, isolate each error type,
combine two, and focus every speaker of every model. It exists because the
things that break a canvas UI -- an undefined identifier, a property read on a
speaker that does not exist in the next clip -- are invisible in a diff.

`explorer_verify.mjs` checks the filters are correct rather than merely
non-crashing:

  * every error region is reachable by at least one type filter;
  * every scored error region either implicates some speaker, or is flagged as
    unattributed -- error time may never silently vanish under a filter;
  * a MISS or FA label never contradicts the region it sits on;
  * the list columns agree with the per-clip payload;
  * DER really is error_sec / total_sec.

Run both after touching index.html or explorer.py. On the 99-clip two-model
export that is ~75,000 assertions and takes a couple of seconds.
