Checks for the page in `../space`. Nothing in this directory goes to the Space.

- `node --test test/app.test.mjs`: the data functions against trimmed copies of the two sources.
- `python3 test/fixture_server.py`, then `http://127.0.0.1:8765/__checks` in a browser: the page in an iframe under each failure mode.
- `node test/live_check.mjs`: the two live sources still look the way the page assumes.
