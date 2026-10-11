Checks for the page in `../space`, run from `community/trending_page`. Nothing in this directory goes to a Space.

- `node --test test/app.test.mjs`: the data functions against trimmed copies of the sources, and the page files (the CSP lists the five source URLs; the page and its card carry one name and one draft state).
- `python3 test/fixture_server.py`, then `http://127.0.0.1:8766/__checks` in a browser: the page in an iframe under each failure mode of each source.
- `node test/live_check.mjs`: the sources themselves still look the way the page assumes; exit status 1 when one does not, 2 when one cannot be reached.
