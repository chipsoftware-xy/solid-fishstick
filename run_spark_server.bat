@echo off
REM Keeps telephony/two_way_test.py running indefinitely, automatically
REM relaunching it every time it exits. This is needed because the
REM server intentionally exits on its own once it detects the
REM confirmed memory/latency correlation (see two_way_test.py for the
REM full explanation) - a full process restart is the only reliable
REM way to actually reclaim that memory.
REM
REM Save this file in the project ROOT (same folder you've been
REM running "python -m telephony.two_way_test" from) - that command
REM resolves relative to the current folder, so this needs to run from
REM the same place.
REM
REM Activates .venv itself, so this works whether you run it from an
REM already-activated terminal or double-click it directly from a
REM fresh one that hasn't activated anything yet.
REM
REM Run THIS file instead of running "python -m telephony.two_way_test"
REM directly - that way each restart happens automatically and you
REM don't need to notice the process exited and relaunch it by hand.
REM
REM Ctrl+C in this window stops the loop entirely (including the
REM currently running server).

call .venv\Scripts\activate.bat

:loop
echo ============================================================
echo Starting two_way_test.py at %date% %time%
echo ============================================================
python -m telephony.two_way_test
echo.
echo ============================================================
echo two_way_test.py exited - relaunching in 2 seconds...
echo (Ctrl+C now if you want to stop for good instead)
echo ============================================================
timeout /t 2 /nobreak > nul
goto loop